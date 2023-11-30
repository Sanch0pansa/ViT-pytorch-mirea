from torch.nn.modules.normalization import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from source.model.Transformer import Transformer
from source.model.PatchEmbedding import PatchEmbedding
import lightning as L

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from torch.optim import SGD


class ViT(L.LightningModule):
    """Vision Transformer with support for patch or hybrid CNN input stage"""
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 6,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 drop_rate: float = 0.0,
                 learning_rate: float = 0.00001,
                 optimizer: str = "Adam",
                 lr_scheduler: str | None = "StepLR",
                 epochs: int = 7,
                 data_loader_len: int = 1000,
                 ):
        """
        Initializes the ViT module.

        Args:
        - img_size (int): Size of the input image.
        - patch_size (int): Size of each patch.
        - in_chans (int): Number of input channels.
        - num_classes (int): Number of output classes.
        - embed_dim (int): Dimension of the embedding.
        - depth (int): Number of Transformer blocks.
        - num_heads (int): Number of attention heads.
        - mlp_ratio (float): Ratio of the hidden dimension in the MLP.
        - qkv_bias (bool): Whether to include bias in the attention module.
        - drop_rate (float): Dropout probability.
        - learning_rate (float): Learning rate for optimization.
        - optimizer (str): Optimizer type.
        - lr_scheduler (str | None): Learning rate scheduler type.
        - epochs (int): Number of epochs
        - data_loader_len (int): Steps per one epoch.
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Assign variables
        self.num = (img_size // patch_size) ** 2

        # Path Embeddings, CLS Token, Position Encoding
        self.embeddings = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                         embed_dim=embed_dim)

        # Transformer Encoder
        self.transformer = Transformer(depth=depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, drop_rate=drop_rate)

        # Classifier
        self.classifier = nn.Linear(in_features=embed_dim, out_features=num_classes)

        # Optimizer and Learning rate scheduler
        self.optimizer_type = optimizer
        self.lr_scheduler_type = lr_scheduler

        # Training parameters
        self.epochs = epochs
        self.data_loader_len = data_loader_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after passing through the ViT model.
        """
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
        """
        Configures the optimizer and scheduler for training.

        Returns:
        - torch.optim.Optimizer: Optimizer for training.
        - torch.optim.lr_scheduler.Scheduler: Learning rate scheduler.
        """

        optimizer = None
        lr_scheduler = None
        if not self.optimizer_type and not self.lr_scheduler_type:
            return None

        if self.optimizer_type == "Adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        elif self.optimizer_type == "SGD":
            optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        if self.lr_scheduler_type == "StepLR":
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.lr_scheduler_type == "OneCycleLR":
            lr_scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=self.data_loader_len, epochs=self.epochs)

        if lr_scheduler:
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

        # Example with SGD optimizer and StepLR scheduler
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
