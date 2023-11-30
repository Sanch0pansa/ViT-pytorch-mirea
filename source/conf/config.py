from dataclasses import dataclass


@dataclass
class Data:
    name: str = "BloodCells"
    path: str = "./data/data"
    seed: int = 1000,
    augmentation_ratio: int = 4,


@dataclass
class Model:
    name: str = "Vit-D3-H8-E128"
    num_classes: int = 4,
    depth: int = 3,
    embed_dim: int = 128,
    num_heads: int = 8,
    drop_rate: float = 0.3,
    mlp_ratio: float = 4


@dataclass
class Training:
    batch_size: int = 128
    batch_size_cpu: int = 128
    epochs: int = 7
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    scheduler: str | None = "StepLR"


@dataclass
class Params:
    data: Data
    model: Model
    training: Training
