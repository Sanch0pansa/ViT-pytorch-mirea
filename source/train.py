import torch
import lightning as L
from source.data_module.BloodCellsDataModule import BloodCellsDataModule
from source.model.ViT import ViT
from lightning.pytorch.callbacks import ModelCheckpoint


def train(
        depth: int = 4,
        num_heads: int = 16,
        embed_dim: int = 256,
        comment: str = "",
        filename_postfix: str = "",
        epochs: int = 5,
        learning_rate: float = 0.001,
        dataset_name: str = "BloodCells+Augmentations",
        drop_rate: float = 0.3,
        load_from_net: str = "",
        load_from_checkpoint: str = ""
):
    """
    Trains the ViT model.

    Args:
    - depth (int): Depth of the ViT model.
    - num_heads (int): Number of attention heads.
    - embed_dim (int): Embedding dimension.
    - comment (str): Comment to be added to the run name.
    - filename_postfix (str): Postfix to be added to the saved model filename.
    - epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    - dataset_name (str): Name of the dataset.
    - drop_rate (float): Dropout rate.
    - load_from_net (str): Path to a pre-trained model to load weights from.
    - load_from_checkpoint (str): Path to a checkpoint to resume training from.
    """
    architecture = f"Vit-D{depth}-H{num_heads}-E{embed_dim}"
    run_name = f"Vit-D{depth}-H{num_heads}-E{embed_dim}{'-' + comment if comment != '' else ''}"
    filename = f"../nets/{run_name}{'-' + filename_postfix if filename_postfix != '' else ''}.pth"
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="ViT",
        name=run_name,
        log_model="all",
        config={
            "learning_rate": learning_rate,
            "architecture": architecture,
            "dataset": dataset_name,
            "epochs": epochs,
        }
    )

    dm = BloodCellsDataModule()
    if load_from_checkpoint:
        model = ViT.load_from_checkpoint(f"./model-chkp/{load_from_checkpoint}")

    else:
        model = ViT(
            num_classes=dm.num_classes,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            learning_rate=learning_rate,
            drop_rate=drop_rate
        )

        if load_from_net:
            model.load_state_dict(torch.load(load_from_net))

    checkpoint_callback = ModelCheckpoint(dirpath='model-chkp/')
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    trainer.fit(model, dm)
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    # Example usage
    train(depth=6, embed_dim=128, num_heads=8, comment="SGD", epochs=7, learning_rate=0.0001)
