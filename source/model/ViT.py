from torch.nn.modules.normalization import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from source.model.Transformer import Transformer
from source.model.PatchEmbedding import PatchEmbedding
import lightning as L

from torch.optim.lr_scheduler import StepLR


class ViT(L.LightningModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0., learning_rate=0.00001):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

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
        out = self.embeddings(x)

        # Transformer Encoder
        out = self.transformer(out)
        out, _ = out.split([1, self.num], dim=1)
        out = out.squeeze(dim=1)

        # Classifier
        out = self.classifier(out)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=4)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=4)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        # return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)