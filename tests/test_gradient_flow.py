import unittest
import torch

from lib.model.attention_mil import AttentionMIL
from lib.model.ordinal import OrdinalHead, OrdinalCrossEntropyLoss, labels_to_ordinal


class TestGradientFlow(unittest.TestCase):
    def test_fusion_to_ordinal_backward(self):
        B, S, D, K = 2, 8, 20, 4
        attention_mil = AttentionMIL(in_dim=D, attn_dim=8, attention_branches=1)
        ordinal_head = OrdinalHead(in_dim=D, num_tiers=K)
        criterion = OrdinalCrossEntropyLoss(num_tiers=K)

        fused_feat = torch.randn(B, S, D, requires_grad=True)
        targets = torch.tensor([2, 1])
        ordinal_targets = labels_to_ordinal(targets, num_tiers=K)

        bag_feat = attention_mil(fused_feat)
        ordinal_probs = ordinal_head(bag_feat)
        loss = criterion(ordinal_probs, ordinal_targets)
        loss.backward()

        self.assertIsNotNone(fused_feat.grad)
        grad = fused_feat.grad
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all().item())


if __name__ == '__main__':
    unittest.main()
