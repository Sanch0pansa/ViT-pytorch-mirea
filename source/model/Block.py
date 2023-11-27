import torch
import torch.nn as nn

from source.model.MLP import MLP
from source.model.Attention import Attention

class Block(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.,
                 qkv_bias: bool = False
                 ):
        """
        Initializes the Block module.

        Args:
        - dim (int): Dimension of the input feature.
        - num_heads (int): Number of attention heads.
        - mlp_ratio (int): Ratio of the hidden dimension in the MLP.
        - drop_rate (float): Dropout probability.
        - qkv_bias (bool): Whether to include bias in the attention module.
        """
        super().__init__()

        # Normalization
        self.norm1 = nn.LayerNorm([dim])

        # Attention
        self.attention = Attention(dim, num_heads, qkv_bias=qkv_bias)

        # Dropout
        self.drop = nn.Dropout(drop_rate)

        # Normalization
        self.norm2 = nn.LayerNorm([dim])

        # MLP
        self.MLP = MLP(dim, dim * mlp_ratio, dim)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the Block module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying attention and MLP.
        """
        save_x = x
        x = self.norm1(x)

        # Attention
        x = self.attention(x)
        x = self.drop(x)
        x += save_x
        save_x = x
        x = self.norm2(x)

        # MLP
        x = self.MLP(x)
        x += save_x

        return x