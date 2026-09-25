"""Post-normalized standard/FC-masked SET blocks."""
import math
import torch
from torch import nn
from .graphs import safe_attention


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout=0., norm_eps=1e-5, bias=True):
        super().__init__()
        if width % heads:
            raise ValueError("Heads must divide hidden width")
        self.heads, self.per_head = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.output = nn.Linear(width, width, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(width, eps=norm_eps)
        self.norm2 = nn.LayerNorm(width, eps=norm_eps)
        self.ffn = nn.Sequential(nn.Linear(width, width, bias=bias), nn.Mish(), nn.Dropout(dropout), nn.Linear(width, width, bias=bias))

    def contribution(self, h, a=None):
        b, n, d = h.shape
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, self.per_head).permute(2, 0, 3, 1, 4).unbind(0)
        weights = safe_attention(q @ k.transpose(-1, -2) / math.sqrt(self.per_head), None if a is None else a[:, None])
        u = (self.dropout(weights) @ v).transpose(1, 2).reshape(b, n, d)
        u = self.dropout(self.output(u))
        if a is not None:
            u = u.masked_fill(~a.any(-1, keepdim=True), 0)  # Includes output bias.
        return u

    def forward(self, h, a=None):
        h = self.norm1(h + self.contribution(h, a))
        return self.norm2(h + self.dropout(self.ffn(h)))

