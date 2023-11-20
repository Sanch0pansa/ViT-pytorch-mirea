import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Sequential(
            nn.Linear(in_features=dim, out_features=3 * dim, bias=qkv_bias),
            # torch.view()
            # Rearrange('b n (k h e) -> b n k h e', h=num_heads, e=head_dim, k=3)
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(in_features=dim, out_features=dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):
        (b, n, e) = x.shape
        # Attention
        qkv = self.qkv(x)
        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = torch.split(qkv, [1, 1, 1], dim=2)
        q = rearrange(q, 'b n 1 h e -> b 1 h n e')
        k = rearrange(k, 'b n 1 h e -> b 1 h n e')
        v = rearrange(v, 'b n 1 h e -> b 1 h n e')
        k = k.transpose(-2, -1)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # Out projection
        out = attn @ v
        out = rearrange(out, 'b 1 h n e -> b n (h e)')

        x = self.out_drop(self.out(out))

        return x
