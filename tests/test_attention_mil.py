import unittest
import torch

from lib.model.attention_mil import AttentionMIL


class TestAttentionMIL(unittest.TestCase):
    def test_output_and_attention_shapes(self):
        model = AttentionMIL(in_dim=16, attn_dim=8, attention_branches=1)
        x = torch.randn(4, 10, 16)
        bag_repr, attn = model(x, return_attention=True)

        self.assertEqual(tuple(bag_repr.shape), (4, 16))
        self.assertEqual(tuple(attn.shape), (4, 1, 10))

    def test_attention_softmax_normalization(self):
        model = AttentionMIL(in_dim=16, attn_dim=8, attention_branches=1)
        x = torch.randn(2, 7, 16)
        _, attn = model(x, return_attention=True)
        sums = attn.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_has_temporal_conv(self):
        model = AttentionMIL(in_dim=16, attn_dim=8, attention_branches=1)
        self.assertTrue(hasattr(model, 'temporal_conv'))


if __name__ == '__main__':
    unittest.main()
