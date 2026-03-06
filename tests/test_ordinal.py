import unittest
import torch

from lib.model.ordinal import OrdinalHead, OrdinalCrossEntropyLoss, labels_to_ordinal


class TestOrdinal(unittest.TestCase):
    def test_labels_to_ordinal(self):
        targets = torch.tensor([2])
        out = labels_to_ordinal(targets, num_tiers=4)
        expected = torch.tensor([[1.0, 1.0, 0.0]])
        self.assertTrue(torch.equal(out, expected))

    def test_ordinal_head_shape(self):
        head = OrdinalHead(in_dim=32, num_tiers=5)
        x = torch.randn(3, 32)
        probs = head(x)
        self.assertEqual(tuple(probs.shape), (3, 4))
        self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)).item())

    def test_loss_prefers_better_predictions(self):
        loss_fn = OrdinalCrossEntropyLoss(num_tiers=4, monotonic_weight=0.0)
        targets = torch.tensor([2, 1])
        ord_targets = labels_to_ordinal(targets, num_tiers=4)

        good = torch.tensor([[0.95, 0.90, 0.05], [0.90, 0.10, 0.05]], dtype=torch.float32)
        bad = torch.tensor([[0.10, 0.10, 0.90], [0.05, 0.90, 0.80]], dtype=torch.float32)

        good_loss = loss_fn(good, ord_targets)
        bad_loss = loss_fn(bad, ord_targets)
        self.assertLess(good_loss.item(), bad_loss.item())

    def test_ordinal_head_return_logits(self):
        head = OrdinalHead(in_dim=16, num_tiers=4)
        x = torch.randn(2, 16)
        probs, logits = head(x, return_logits=True)
        self.assertEqual(tuple(probs.shape), (2, 3))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_monotonic_penalty_increases_loss(self):
        loss_wo = OrdinalCrossEntropyLoss(num_tiers=4, monotonic_weight=0.0)
        loss_w = OrdinalCrossEntropyLoss(num_tiers=4, monotonic_weight=1.0)

        targets = torch.tensor([2])
        ord_targets = labels_to_ordinal(targets, num_tiers=4)

        monotonic_ok = torch.tensor([[0.9, 0.6, 0.2]], dtype=torch.float32)
        monotonic_bad = torch.tensor([[0.2, 0.8, 0.9]], dtype=torch.float32)

        base_ok = loss_wo(monotonic_ok, ord_targets)
        penalized_ok = loss_w(monotonic_ok, ord_targets)
        base_bad = loss_wo(monotonic_bad, ord_targets)
        penalized_bad = loss_w(monotonic_bad, ord_targets)

        self.assertLessEqual((penalized_ok - base_ok).item(), 1e-6)
        self.assertGreater((penalized_bad - base_bad).item(), 0.0)


if __name__ == '__main__':
    unittest.main()
