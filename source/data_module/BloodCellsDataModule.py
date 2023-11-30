import os
import lightning as L
import torch
import torchvision
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from source.data_module.BloodCellsDataset import BloodCellsDataset

# Note - you must have torchvision installed for this example

# PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data/data")
# BATCH_SIZE = 128 if torch.cuda.is_available() else 16


class BloodCellsDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./",
                 seed: int = 1000,
                 batch_size: int = 128,
                 batch_size_cpu: int = 128,
                 augmentation_ratio: int = 4,
                 ):
        """
        Initializes the BloodCellsDataModule.

        Args:
        - data_dir (str): Path to the data directory.
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = A.Compose(
            [
                A.Resize(width=224, height=224),
                A.OneOf([
                    A.Blur(),
                    A.GaussNoise(),
                    A.RandomBrightnessContrast(),
                    A.RGBShift(),
                ], p=1),
                A.Rotate(limit=30, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.test_transform = A.Compose(
            [
                A.Resize(width=224, height=224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.batch_size = batch_size if torch.cuda.is_available() else batch_size_cpu
        self.seed = seed
        self.augmentation_ratio = augmentation_ratio

    def len_of_train_dataset(self):
        classes = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

        cls = [os.path.join(self.data_dir, "TRAIN", cl) for cl in classes]
        images_filepaths = []
        [images_filepaths.extend(sorted([os.path.join(cls[i], f) for f in os.listdir(cls[i])])) for i in
         range(len(classes))]
        correct_train_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]
        return (len(correct_train_images_filepaths) - 1000) * self.augmentation_ratio

    def prepare_data(self):
        """
        Prepares the dataset.

        This method is responsible for downloading, extracting, and preparing the dataset.
        """
        pass

    def setup(self, stage=None):
        """
        Set up the data for training, validation, and testing.

        Args:
        - stage (str, optional): Stage of training (e.g., 'fit', 'test').
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        classes = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

        cls = [os.path.join(self.data_dir, "TRAIN", cl) for cl in classes]
        images_filepaths = []
        [images_filepaths.extend(sorted([os.path.join(cls[i], f) for f in os.listdir(cls[i])])) for i in
         range(len(classes))]
        correct_train_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]

        cls = [os.path.join(self.data_dir, "TEST", cl) for cl in classes]
        images_filepaths = []
        [images_filepaths.extend(sorted([os.path.join(cls[i], f) for f in os.listdir(cls[i])])) for i in
         range(len(classes))]
        correct_test_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]

        self.test_dataset = BloodCellsDataset(images_filepaths=correct_test_images_filepaths,
                                              transform=self.test_transform)

        train_dataset = BloodCellsDataset(images_filepaths=correct_train_images_filepaths,
                                          transform=self.train_transform)
        # Augmented datasets
        augmented_datasets = []
        for _ in range(self.augmentation_ratio - 1):  # You can adjust the number of augmentations as needed
            augmented_datasets.append(
                BloodCellsDataset(images_filepaths=correct_train_images_filepaths,
                                  transform=self.train_transform)
            )

        # Concatenate datasets
        train_dataset = torch.utils.data.ConcatDataset([train_dataset] + augmented_datasets)
        self.val_dataset, self.train_dataset = torch.utils.data.random_split(train_dataset, [1000, len(train_dataset) - 1000])

        # self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [9000, 957])

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
        - torch.utils.data.DataLoader: DataLoader for the training dataset.
        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
        - torch.utils.data.DataLoader: DataLoader for the validation dataset.
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns:
        - torch.utils.data.DataLoader: DataLoader for the test dataset.
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)