import sys
import types
import unittest
import os
import importlib
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


if importlib.util.find_spec('tensorboardX') is None:
    tbx = types.ModuleType('tensorboardX')
    class _DummyWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
    setattr(tbx, 'SummaryWriter', _DummyWriter)
    sys.modules['tensorboardX'] = tbx
else:
    importlib.import_module('tensorboardX')

if importlib.util.find_spec('easydict') is None:
    ed = types.ModuleType('easydict')
    class _EasyDict(dict):
        def __getattr__(self, item):
            return self[item]
        def __setattr__(self, key, value):
            self[key] = value
    setattr(ed, 'EasyDict', _EasyDict)
    sys.modules['easydict'] = ed
else:
    importlib.import_module('easydict')

import train_action
from lib.model.attention_mil import AttentionMIL
from lib.model.ordinal import OrdinalHead, OrdinalCrossEntropyLoss


class DummyActionNet(nn.Module):
    def __init__(self, feat_shape=(1, 2, 3, 4), num_classes=4):
        super().__init__()
        self.feat_shape = feat_shape
        self.num_classes = num_classes

    def forward(self, x, return_features=False):
        B = x.shape[0]
        M, T, J, C = self.feat_shape
        feat = torch.randn(B, M, T, J, C)
        out = torch.randn(B, self.num_classes)
        if return_features:
            return out, feat
        return out


class TestTrainActionIntegration(unittest.TestCase):
    def test_ordinal_to_class_probs_normalized(self):
        ordinal_probs = torch.tensor([[0.8, 0.4, 0.1], [0.2, 0.1, 0.05]], dtype=torch.float32)
        class_probs = train_action._ordinal_to_class_probs(ordinal_probs)

        self.assertEqual(tuple(class_probs.shape), (2, 4))
        sums = class_probs.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_validate_raises_on_sequence_mismatch(self):
        model = DummyActionNet(feat_shape=(1, 2, 3, 4), num_classes=4)
        attention_mil = AttentionMIL(in_dim=9, attn_dim=8, attention_branches=1)
        ordinal_head = OrdinalHead(in_dim=9, num_tiers=4)
        criterion = OrdinalCrossEntropyLoss(num_tiers=4)

        batch_input = torch.randn(2, 1, 2, 3, 3)
        batch_gt = torch.tensor([1, 2], dtype=torch.long)
        batch_video = {'video_path': ['a.mp4', 'b.mp4']}
        test_loader = [(batch_input, batch_gt, batch_video)]
        opts = SimpleNamespace(print_freq=100)

        # mb seq len = M*T*J = 1*2*3 = 6, but lma seq len is 5 -> mismatch
        with patch('torch.cuda.is_available', return_value=False):
            with patch('train_action._extract_lma_feature_batch', return_value=torch.randn(2, 5, 5)):
                with self.assertRaises(ValueError):
                    _ = train_action.validate(test_loader, model, criterion, attention_mil, ordinal_head, 4, opts)

    def test_validate_runs_with_static_lma(self):
        model = DummyActionNet(feat_shape=(1, 2, 3, 4), num_classes=4)
        attention_mil = AttentionMIL(in_dim=9, attn_dim=8, attention_branches=1)
        ordinal_head = OrdinalHead(in_dim=9, num_tiers=4)
        criterion = OrdinalCrossEntropyLoss(num_tiers=4)

        batch_input = torch.randn(2, 1, 2, 3, 3)
        batch_gt = torch.tensor([1, 2], dtype=torch.long)
        batch_video = {'video_path': ['a.mp4', 'b.mp4']}
        test_loader = [(batch_input, batch_gt, batch_video)]
        opts = SimpleNamespace(print_freq=100)

        with patch('torch.cuda.is_available', return_value=False):
            with patch('train_action._extract_lma_feature_batch', return_value=torch.randn(2, 5)):
                loss, top1, top5 = train_action.validate(test_loader, model, criterion, attention_mil, ordinal_head, 4, opts)

        self.assertTrue(isinstance(loss, float))
        self.assertTrue(isinstance(top1, torch.Tensor) or isinstance(top1, float))
        self.assertTrue(isinstance(top5, torch.Tensor) or isinstance(top5, float))


if __name__ == '__main__':
    unittest.main()
