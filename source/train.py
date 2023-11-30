import torch
import lightning as L
from source.data_module.BloodCellsDataModule import BloodCellsDataModule
from source.model.ViT import ViT
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra
from hydra.core.config_store import ConfigStore
from source.conf.config import Params


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(
        cfg: Params,
        comment: str = "",
        filename_postfix: str = "",
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
    architecture = f"Vit-D{cfg.model.depth}-H{cfg.model.num_heads}-E{cfg.model.embed_dim}"
    run_name = f"Vit-D{cfg.model.depth}-H{cfg.model.num_heads}-E{cfg.model.embed_dim}{'-' + comment if comment != '' else ''}"
    filename = f"../nets/{run_name}{'-' + filename_postfix if filename_postfix != '' else ''}.pth"
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="ViT",
        name=run_name,
        log_model="all",
        config={
            "learning_rate": cfg.training.learning_rate,
            "architecture": architecture,
            "dataset": cfg.data.name,
            "epochs": cfg.training.epochs,
        }
    )

    dm = BloodCellsDataModule(
        data_dir=cfg.data.path,
        seed=cfg.data.seed,
        batch_size=cfg.training.batch_size,
        batch_size_cpu=cfg.training.batch_size_cpu,
        augmentation_ratio=cfg.data.augmentation_ratio
    )
    if load_from_checkpoint:
        model = ViT.load_from_checkpoint(f"./model-chkp/{load_from_checkpoint}")

    else:
        model = ViT(
            num_classes=cfg.model.num_classes,
            depth=cfg.model.depth,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            learning_rate=cfg.training.learning_rate,
            drop_rate=cfg.model.drop_rate,
            optimizer=cfg.training.optimizer,
            epochs=cfg.training.epochs,
            data_loader_len=dm.len_of_train_dataset()
        )

        if load_from_net:
            model.load_state_dict(torch.load(load_from_net))

    checkpoint_callback = ModelCheckpoint(dirpath='model-chkp/')
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
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
    train()
