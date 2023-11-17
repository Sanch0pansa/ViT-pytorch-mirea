import torch
import torch.nn as nn

from source.model.MLP import MLP
from source.model.Attention import Attention


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()

        # Normalization
        self.norm1 = nn.LayerNorm([dim])

        # Attention
        self.attention = Attention(dim, num_heads)

        # Dropout
        self.drop = nn.Dropout(drop_rate)

        # Normalization
        self.norm2 = nn.LayerNorm([dim])

        # MLP
        self.MLP = MLP(dim, dim * mlp_ratio, dim)

    def forward(self, x):
        save_x = x
        x = self.norm1(x)

        # Attetnion
        x = self.attention(x)
        x += save_x
        save_x = x
        x = self.norm2(x)

        # MLP
        x = self.MLP(x)
        x += save_x

        return x