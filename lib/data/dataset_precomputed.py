import os
import re
import json
import random
import numpy as np
from torch.utils.data import Dataset

from lib.utils.utils_data import resample, crop_scale_3d


def _extract_fragment_id(path):
    name = os.path.basename(path)
    m = re.search(r'wham_fragment_id(\d+)\.npz$', name)
    if m is None:
        return None
    return int(m.group(1))


class PrecomputedWHAMFolderDataset(Dataset):
    def __init__(
        self,
        registry_path,
        dataset_keys,
        data_split,
        datasets_root='',
        n_frames=243,
        random_move=False,
        scale_range=[1, 1],
        num_joints=17,
    ):
        self.data_split = data_split
        self.is_train = data_split == 'train'
        self.n_frames = n_frames
        self.random_move = random_move
        self.scale_range = scale_range
        self.num_joints = num_joints

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        datasets_cfg = registry.get('datasets', {})
        global_seed = int(registry.get('global_seed', 0))
        datasets_root = datasets_root or ''

        if isinstance(dataset_keys, str):
            dataset_keys = [k.strip() for k in dataset_keys.split(',') if k.strip()]

        if len(dataset_keys) == 0:
            raise ValueError('dataset_keys is empty. Please provide at least one dataset key from registry.')

        video_groups = {}
        for key in dataset_keys:
            if key not in datasets_cfg:
                raise KeyError(f'dataset key {key} not found in registry: {registry_path}')
            cfg = datasets_cfg[key]
            root_dir_cfg = cfg['features_root']
            if os.path.isabs(root_dir_cfg):
                root_dir = root_dir_cfg
            elif datasets_root:
                root_dir = os.path.join(datasets_root, root_dir_cfg)
            else:
                root_dir = root_dir_cfg
            label_tier = int(cfg['label_tier'])
            include_failed = bool(cfg.get('include_failed', False))
            split_ratios = cfg.get('split_ratios', {'train': 0.8, 'val': 0.1, 'test': 0.1})
            split_seed = int(cfg.get('split_seed', global_seed))

            if not os.path.isdir(root_dir):
                raise FileNotFoundError(f'features_root not found: {root_dir}')

            video_ids = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

            rng = random.Random(split_seed)
            shuffled = video_ids[:]
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * float(split_ratios.get('train', 0.8)))
            n_val = int(n * float(split_ratios.get('val', 0.1)))
            train_set = set(shuffled[:n_train])
            val_set = set(shuffled[n_train:n_train + n_val])
            test_set = set(shuffled[n_train + n_val:])

            if data_split == 'train':
                active_ids = train_set
            elif data_split in ['val', 'valid']:
                active_ids = val_set
            else:
                active_ids = test_set

            for vid in active_ids:
                folder = os.path.join(root_dir, vid)
                summary_path = os.path.join(folder, 'summary.json')
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as sf:
                            status = json.load(sf).get('status', '')
                    except Exception:
                        status = ''
                    if (not include_failed) and status != 'success':
                        continue

                npz_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('wham_fragment_id') and f.endswith('.npz')])
                if len(npz_files) == 0:
                    continue

                samples = []
                for npz in npz_files:
                    fragment_id = _extract_fragment_id(npz)
                    if fragment_id is None:
                        continue
                    samples.append({
                        'dataset_key': key,
                        'video_id': vid,
                        'label': label_tier,
                        'npz_path': npz,
                        'fragment_id': fragment_id,
                    })

                if len(samples) > 0:
                    video_groups[(key, vid)] = samples

        self.samples = []
        for _, group in sorted(video_groups.items(), key=lambda x: (x[0][0], x[0][1])):
            self.samples.extend(group)

        if len(self.samples) == 0:
            raise RuntimeError(f'No precomputed samples found for split={data_split}, keys={dataset_keys}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample['npz_path'], allow_pickle=True)
        joints = data['joints'].astype(np.float32)  # (T, J, 3)
        if joints.ndim != 3:
            raise ValueError(f'Unexpected joints shape in {sample["npz_path"]}: {joints.shape}')

        T = joints.shape[0]
        if T <= 0:
            raise ValueError(f'Empty joints in {sample["npz_path"]}')

        joint_count = min(self.num_joints, joints.shape[1])
        resample_id = resample(ori_len=T, target_len=self.n_frames, randomness=self.is_train)
        motion = joints[resample_id, :joint_count, :]  # (T, J, 3)

        if self.scale_range is not None:
            motion = crop_scale_3d(motion, scale_range=self.scale_range)

        if joint_count < self.num_joints:
            pad = np.zeros((motion.shape[0], self.num_joints - joint_count, 3), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=1)

        fake = np.zeros_like(motion)
        motion = np.stack([motion, fake], axis=0).astype(np.float32)  # (M=2, T, J, 3)

        video_info = {
            'video_path': sample['video_id'],
            'filename': sample['video_id'],
            'frame_dir': sample['video_id'],
            'fragment_id': sample['fragment_id'],
            'dataset_key': sample['dataset_key'],
        }

        return motion, int(sample['label']), video_info
