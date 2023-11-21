import torch
import lightning as L
from source.data_module.BloodCellsDataModule import BloodCellsDataModule
from source.model.ViT import ViT
from lightning.pytorch.callbacks import ModelCheckpoint

EPOCHS = 5

wandb_logger = L.pytorch.loggers.WandbLogger(
    project="ViT",
    name="Vit-D6-H12-E768",
    log_model="all",
    config={
        "learning_rate": 0.00001,
        "architecture": "Vit-D6-H12-E768",
        "dataset": "BloodCells",
        "epochs": EPOCHS,
    }
)

dm = BloodCellsDataModule()
model = ViT(num_classes=dm.num_classes)
checkpoint_callback = ModelCheckpoint(dirpath='model-chkp/')
# early_stopping = EarlyStopping('val_loss')

# log gradients, parameter histogram and model topology
wandb_logger.watch(model, log_graph=False)
# model.load_state_dict(torch.load(from_load))
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
    callbacks=[checkpoint_callback]
)
trainer.fit(model, dm)
torch.save(model.state_dict(), "nets/nn1.pth")
