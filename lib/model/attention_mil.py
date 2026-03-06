import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=128, attention_branches=1):
        super(AttentionMIL, self).__init__()
        self.in_dim = in_dim
        self.attn_dim = attn_dim
        self.attention_branches = attention_branches

        self.temporal_conv = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=True)
        self.temporal_norm = nn.LayerNorm(self.in_dim)

        self.attention_V = nn.Sequential(
            nn.Linear(self.in_dim, self.attn_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.in_dim, self.attn_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.attn_dim, self.attention_branches)

    def forward(self, x, mask=None, return_attention=False):
        '''
            Input:
                x: (B, N, D)
                mask: (B, N), optional valid-instance mask
            Output:
                bag_repr: (B, D) if attention_branches==1 else (B, K*D)
        '''
        x = x + self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.temporal_norm(x)

        A_V = self.attention_V(x)                      # (B, N, L)
        A_U = self.attention_U(x)                      # (B, N, L)
        A = self.attention_w(A_V * A_U)               # (B, N, K)
        A = A.transpose(1, 2)                         # (B, K, N)

        if mask is not None:
            mask = mask.unsqueeze(1).to(dtype=torch.bool)  # (B, 1, N)
            A = A.masked_fill(~mask, float('-inf'))

        A = F.softmax(A, dim=-1)                      # softmax over instances (N)
        Z = torch.bmm(A, x)                           # (B, K, D)

        if self.attention_branches == 1:
            bag_repr = Z.squeeze(1)                   # (B, D)
        else:
            bag_repr = Z.reshape(x.shape[0], -1)      # (B, K*D)

        if return_attention:
            return bag_repr, A
        return bag_repr
