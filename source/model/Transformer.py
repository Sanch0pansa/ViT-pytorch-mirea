import torch
import torch.nn as nn
from source.model.Block import Block


class Transformer(nn.Module):
    def __init__(self,
                 depth: int,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.,
                 qkv_bias: bool = False
                 ):
        """
        Initializes the Transformer module.

        Args:
        - depth (int): Number of blocks in the Transformer.
        - dim (int): Dimension of the input features.
        - num_heads (int): Number of attention heads.
        - mlp_ratio (int): Ratio of the hidden dimension in the MLP.
        - drop_rate (float): Dropout probability.
        - qkv_bias (bool): Whether to include bias in the attention module.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate, qkv_bias=qkv_bias)
            for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after passing through the Transformer blocks.
        """
        for block in self.blocks:
            x = block(x)
        return x
