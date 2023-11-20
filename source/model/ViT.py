from torch.nn.modules.normalization import LayerNorm
import torch
import torch.nn as nn
from source.model.Transformer import Transformer
from source.model.PatchEmbedding import PatchEmbedding

class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0.,):
        super().__init__()

        # Присвоение переменных
        self.num = (img_size // patch_size) ** 2

        # Path Embeddings, CLS Token, Position Encoding
        self.embeddings = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                         embed_dim=embed_dim)

        # Transformer Encoder
        self.transformer = Transformer(depth=depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, drop_rate=drop_rate)


        # Classifier
        self.classifier = nn.Linear(in_features=embed_dim, out_features=num_classes)

    def forward(self, x):

        # Path Embeddings, CLS Token, Position Encoding
        x = self.embeddings(x)

        # Transformer Encoder
        x = self.transformer(x)
        x, _ = x.split([1, self.num], dim=1)
        x = x.squeeze(dim=1)

        # Classifier
        x = self.classifier(x)

        return x