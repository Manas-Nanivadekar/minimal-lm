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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        
        self.c_attn = nn.Linear(config.d_model, 3*config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_head

        self.rope = RoPE(self.head_dim, max_seq_len=config.max_seq_len)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        # Reshape for multi-head attention: [B, T, C] -> [B, T, n_head, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        cos, sin = self.rope(T, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Transpose for attention computation: [B, n_head, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # [B, n_head, T, head_dim]

        # Concatenate heads: [B, n_head, T, head_dim] -> [B, T, C]
        y = y.transpose(1,2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    Linear -> GELU -> Linear -> Dropout
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4*config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.d_model,config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x