import torch
import lightning as L
from source.data_module.BloodCellsDataModule import BloodCellsDataModule
from source.model.ViT import ViT
from lightning.pytorch.callbacks import ModelCheckpoint

wandb_logger = L.pytorch.loggers.WandbLogger(project="ViT")

dm = BloodCellsDataModule()
model = ViT(num_classes=dm.num_classes)
checkpoint_callback = ModelCheckpoint(dirpath='model-chkp/')
# early_stopping = EarlyStopping('val_loss')

# log gradients, parameter histogram and model topology
wandb_logger.watch(model, log_freq=1)
# model.load_state_dict(torch.load(from_load))
trainer = L.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
    callbacks=[checkpoint_callback]
)
trainer.fit(model, dm)
torch.save(model.state_dict(), "nets/nn1.pth")
