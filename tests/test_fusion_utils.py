import unittest
import torch

from lib.model.fusion_utils import align_lma_to_sequence


class TestFusionUtils(unittest.TestCase):
    def test_expand_static_lma(self):
        lma = torch.randn(2, 12)
        out = align_lma_to_sequence(lma, seq_len=5)
        self.assertEqual(tuple(out.shape), (2, 5, 12))

    def test_keep_temporal_lma(self):
        lma = torch.randn(2, 5, 12)
        out = align_lma_to_sequence(lma, seq_len=5)
        self.assertEqual(tuple(out.shape), (2, 5, 12))

    def test_mismatch_raises(self):
        lma = torch.randn(2, 4, 12)
        with self.assertRaises(ValueError):
            _ = align_lma_to_sequence(lma, seq_len=5)


if __name__ == '__main__':
    unittest.main()
