import torch
import torch.nn as nn
import torch.nn.functional as F


def labels_to_ordinal(targets, num_tiers):
    '''
        targets: (B,) integer class labels in [0, K-1]
        return: (B, K-1) cumulative binary targets
                t_j = 1 if y > j else 0, j=0..K-2
    '''
    if num_tiers < 2:
        raise ValueError('num_tiers must be >= 2 for ordinal classification.')
    thresholds = torch.arange(num_tiers - 1, device=targets.device).unsqueeze(0)
    ordinal_targets = (targets.unsqueeze(1) > thresholds).to(dtype=torch.float32)
    return ordinal_targets


class OrdinalHead(nn.Module):
    def __init__(self, in_dim, num_tiers):
        super(OrdinalHead, self).__init__()
        if num_tiers < 2:
            raise ValueError('num_tiers must be >= 2 for ordinal classification.')
        self.num_tiers = num_tiers
        self.fc = nn.Linear(in_dim, num_tiers - 1)

    def forward(self, x, return_logits=False):
        logits = self.fc(x)
        probs = torch.sigmoid(logits)
        if return_logits:
            return probs, logits
        return probs


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_tiers, reduction='mean', distance_weighted=True, monotonic_weight=0.1, eps=1e-7):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_tiers = num_tiers
        self.reduction = reduction
        self.distance_weighted = distance_weighted
        self.monotonic_weight = monotonic_weight
        self.eps = eps

    def forward(self, probs, ordinal_targets):
        '''
            probs: (B, K-1), sigmoid outputs from OrdinalHead
            ordinal_targets: (B, K-1), cumulative binary targets
        '''
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0 - self.eps, neginf=self.eps)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        bce = F.binary_cross_entropy(probs, ordinal_targets, reduction='none')

        if self.distance_weighted:
            true_tier = ordinal_targets.sum(dim=1)  # (B,)
            threshold_ids = torch.arange(self.num_tiers - 1, device=probs.device, dtype=probs.dtype).unsqueeze(0)
            weights = 1.0 + torch.abs(true_tier.unsqueeze(1) - threshold_ids) / max(self.num_tiers - 1, 1)
            bce = bce * weights

        if self.reduction == 'sum':
            base_loss = bce.sum()
        else:
            base_loss = bce.mean()

        # Encourage ordinal consistency: p(y>k+1) <= p(y>k)
        mono_violation = F.relu(probs[:, 1:] - probs[:, :-1])
        mono_penalty = mono_violation.mean() if mono_violation.numel() > 0 else probs.new_tensor(0.0)
        return base_loss + self.monotonic_weight * mono_penalty
