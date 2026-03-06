import os
import numpy as np
import time
import sys
import argparse
import errno
import glob
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet
from lib.model.attention_mil import AttentionMIL

_wham_available = False
process_single_video = None
_kg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _kg_root not in sys.path:
    sys.path.insert(0, _kg_root)
try:
    from core.wham_inference import process_single_video
    _wham_available = True
except Exception:
    _wham_available = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--video_root', default='', type=str, help='Optional root directory for raw videos.')
    parser.add_argument('--lma_cache_dir', default='output/wham_kineguard', type=str, help='Directory with WHAM/LMA outputs.')
    parser.add_argument('--run_wham_online', action='store_true', help='Run WHAM+LMA online if cached LMA is missing.')
    parser.add_argument('--lma_feature_dim', default=128, type=int, help='Target LMA feature dimension for fusion.')
    parser.add_argument('--mil_attn_dim', default=128, type=int, help='Attention hidden dimension for AttentionMIL.')
    parser.add_argument('--mil_branches', default=1, type=int, help='Number of MIL attention branches.')
    opts = parser.parse_args()
    return opts

def _to_fixed_dim(vec, target_dim):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape[0] > target_dim:
        vec = vec[:target_dim]
    elif vec.shape[0] < target_dim:
        vec = np.pad(vec, (0, target_dim - vec.shape[0]))
    return vec

def _load_single_lma_feature(lma_file, target_dim):
    arr = np.load(lma_file, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        if arr.dtype == object:
            arr = np.array(arr.tolist(), dtype=np.float32)
        arr = np.nan_to_num(arr.astype(np.float32))
        if arr.ndim == 1:
            vec = arr
        else:
            vec = arr.reshape(arr.shape[0], -1).mean(axis=0)
        return _to_fixed_dim(vec, target_dim)
    return np.zeros((target_dim,), dtype=np.float32)

def _resolve_video_path(video_info, video_root=''):
    candidates = []
    if isinstance(video_info, dict):
        for key in ['video_path', 'filename', 'frame_dir']:
            val = video_info.get(key, '')
            if isinstance(val, str) and len(val) > 0:
                candidates.append(val)
    elif isinstance(video_info, str):
        candidates.append(video_info)

    exts = ['', '.mp4', '.avi', '.mov', '.mkv']
    for c in candidates:
        if os.path.isfile(c):
            return c
        if video_root:
            for ext in exts:
                cand = os.path.join(video_root, c + ext)
                if os.path.isfile(cand):
                    return cand
    return ''

def _extract_lma_feature_batch(batch_video, opts, device):
    if isinstance(batch_video, dict):
        n = len(batch_video.get('video_path', []))
        video_infos = []
        for i in range(n):
            info = {k: batch_video[k][i] for k in batch_video.keys()}
            video_infos.append(info)
    else:
        video_infos = batch_video

    batch_vec = []
    os.makedirs(opts.lma_cache_dir, exist_ok=True)
    for video_info in video_infos:
        video_path = _resolve_video_path(video_info, opts.video_root)
        if video_path:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            sample_dir = os.path.join(opts.lma_cache_dir, video_name)
            lma_files = sorted(glob.glob(os.path.join(sample_dir, 'lma_features_id*.npy')))
        else:
            lma_files = []

        if len(lma_files) == 0 and opts.run_wham_online and _wham_available and process_single_video is not None and video_path:
            try:
                process_single_video(video_path=video_path, output_root=opts.lma_cache_dir, visualize=False)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                sample_dir = os.path.join(opts.lma_cache_dir, video_name)
                lma_files = sorted(glob.glob(os.path.join(sample_dir, 'lma_features_id*.npy')))
            except Exception:
                lma_files = []

        if len(lma_files) > 0:
            vectors = [_load_single_lma_feature(f, opts.lma_feature_dim) for f in lma_files]
            lma_vec = np.mean(np.stack(vectors, axis=0), axis=0)
        else:
            lma_vec = np.zeros((opts.lma_feature_dim,), dtype=np.float32)
        batch_vec.append(lma_vec)

    lma_tensor = torch.from_numpy(np.stack(batch_vec, axis=0)).to(device=device)
    return lma_tensor

def validate(test_loader, model, criterion, attention_mil, fusion_head, opts):
    model.eval()
    attention_mil.eval()
    fusion_head.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt, batch_video) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
                device = batch_input.device
            else:
                device = torch.device('cpu')

            _, mb_feat = model(batch_input, return_features=True)
            B, M, T, J, C = mb_feat.shape
            mb_feat = mb_feat.reshape(B, M * T * J, C)  # (B, S, C)
            lma_feat = _extract_lma_feature_batch(batch_video, opts, device)
            lma_feat = lma_feat.unsqueeze(1).expand(-1, mb_feat.shape[1], -1)  # (B, S, L)
            fusion_feat = torch.cat([mb_feat, lma_feat], dim=-1)  # (B, S, C+L)
            bag_feat = attention_mil(fusion_feat)
            output = fusion_head(bag_feat)
            loss = criterion(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    fused_dim = args.dim_rep + opts.lma_feature_dim
    attention_mil = AttentionMIL(in_dim=fused_dim, attn_dim=opts.mil_attn_dim, attention_branches=opts.mil_branches)
    mil_out_dim = fused_dim * opts.mil_branches
    fusion_head = nn.Linear(mil_out_dim, args.action_classes)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        attention_mil = attention_mil.cuda()
        fusion_head = fusion_head.cuda()
        criterion = criterion.cuda() 
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'data/action/%s.pkl' % args.dataset
    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_train', n_frames=args.clip_len, random_move=args.random_move, scale_range=args.scale_range_train)
    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)

    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
        if 'attention_mil' in checkpoint and checkpoint['attention_mil'] is not None:
            attention_mil.load_state_dict(checkpoint['attention_mil'], strict=True)
        if 'fusion_head' in checkpoint and checkpoint['fusion_head'] is not None:
            fusion_head.load_state_dict(checkpoint['fusion_head'], strict=True)
    
    if not opts.evaluate:
        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
                                {"params": filter(lambda p: p.requires_grad, attention_mil.parameters()), "lr": args.lr_head},
                {"params": filter(lambda p: p.requires_grad, fusion_head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            attention_mil.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt, batch_video) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                    device = batch_input.device
                else:
                    device = torch.device('cpu')

                _, mb_feat = model(batch_input, return_features=True)
                B, M, T, J, C = mb_feat.shape
                mb_feat = mb_feat.reshape(B, M * T * J, C)  # (B, S, C)
                lma_feat = _extract_lma_feature_batch(batch_video, opts, device)
                lma_feat = lma_feat.unsqueeze(1).expand(-1, mb_feat.shape[1], -1)  # (B, S, L)
                fusion_feat = torch.cat([mb_feat, lma_feat], dim=-1)  # (B, S, C+L)
                bag_feat = attention_mil(fusion_feat)
                output = fusion_head(bag_feat)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()    
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1))
                sys.stdout.flush()
                
            test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, attention_mil, fusion_head, opts)
                
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('test_top5', test_top5, epoch + 1)
            
            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'attention_mil': attention_mil.state_dict(),
                'fusion_head': fusion_head.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if test_top1 > best_acc:
                best_acc = test_top1
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'attention_mil': attention_mil.state_dict(),
                'fusion_head': fusion_head.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

    if opts.evaluate:
        test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, attention_mil, fusion_head, opts)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)