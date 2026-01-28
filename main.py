import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    """
    - Learned embeddings don't generalize to longer sequences than training
    - Better extrapolation to unseen sequence lengths
    - Position m rotates query/key by angle m*θ
    - Attention score depends on relative position (m-n) via rotation difference
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        # Compute rotation angles for each dimension pair
        # θ_i = 10000^(-2i/d) where i ∈ [0, d/2)
        self.inv_freq = 1.0/(10000 ** (torch.arange(0, dim, 2).float()/dim))
        self.register_buffer("inv_freq", self.inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(x, cos, sin):
    """
    :param x: Input tensor [batch, seq_len, n_heads, head_dim]
    :param cos, sin: Rotation matrices [seq_len, head_dim]

    The rotation formula:
        [x1, x2] rotated by θ = [x1*cos(θ) - x2*sin(θ), x1*sin(θ) + x2*cos(θ)]
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    # This implements complex number rotation: (a+bi) * e^(iθ) = (a+bi)(cos(θ)+i*sin(θ))
    rotated_x1 = x1*cos - x2*sin
    rotated_x2 = x1*sin + x2*cos

    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

