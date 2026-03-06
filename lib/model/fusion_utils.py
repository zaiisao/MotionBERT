import torch


def align_lma_to_sequence(lma_feat, seq_len):
    '''
        Align LMA features to MotionBERT sequence length.

        Inputs:
            lma_feat: (B, L) or (B, S_lma, L)
            seq_len: int, target sequence length
        Returns:
            (B, seq_len, L)
    '''
    if lma_feat.ndim == 2:
        return lma_feat.unsqueeze(1).expand(-1, seq_len, -1)

    if lma_feat.ndim == 3:
        if lma_feat.shape[1] != seq_len:
            raise ValueError(f"Sequence mismatch: MB={seq_len} vs LMA={lma_feat.shape[1]}")
        return lma_feat

    raise ValueError(f"Unexpected LMA shape: {tuple(lma_feat.shape)}")
