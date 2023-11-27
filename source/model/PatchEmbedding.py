import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
import numpy as np

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(1, sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        self.patch_embeddings = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.position_embeddings = nn.Parameter(torch.tensor(get_positional_embeddings((img_size // patch_size) ** 2 + 1, embed_dim)))
        # self.position_embeddings = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patches = self.patch_embeddings(x)
        patches = torch.cat((patches, repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])), dim=1)
        patches += repeat(self.position_embeddings, '() n e -> b n e', b=x.shape[0])

        return patches