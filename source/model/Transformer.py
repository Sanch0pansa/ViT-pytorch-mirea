import torch
import torch.nn as nn
from source.model.Block import Block


class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0., qkv_bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate, qkv_bias=qkv_bias)
            for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x