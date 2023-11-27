import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 out_drop: float = 0.
                 ):
        """
        Initializes the Attention module.

        Args:
        - dim (int): Dimension of the input feature.
        - num_heads (int): Number of attention heads.
        - qkv_bias (bool): Whether to include bias in the linear transformations for Q, K, and V.
        - attn_drop (float): Dropout probability for the attention weights.
        - out_drop (float): Dropout probability for the output.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear transformations for Q, K, and V
        self.qkv = nn.Sequential(
            nn.Linear(in_features=dim, out_features=3 * dim, bias=qkv_bias),
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(in_features=dim, out_features=dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dimension).

        Returns:
        - torch.Tensor: Output tensor after attention mechanism.
        """
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

        out = attn @ v
        out = rearrange(out, 'b 1 h n e -> b n (h e)')

        x = self.out_drop(self.out(out))

        return x
