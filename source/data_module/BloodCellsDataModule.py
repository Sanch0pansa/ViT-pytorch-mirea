import os

import lightning as L
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from source.data_module.BloodCellsDataset import BloodCellsDataset

# Note - you must have torchvision installed for this example

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class BloodCellsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = A.Compose(  # набор трансформаций для тренировочного набора
            [
                A.SmallestMaxSize(max_size=256),
                A.RandomCrop(height=224, width=224),
                # добавить несколько трансформаций по своему выбору
                A.OneOf([  # делает только один вид преобразования из списка
                    A.Blur(),
                    A.GaussNoise(),
                    A.RandomBrightnessContrast(),
                    A.RGBShift(),
                ], p=1),

                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # нормализация с теми же значениями, как предобрабатывались изображения набора данных ImageNet в предобученной нейронной сети
                ToTensorV2(),  # перевод к формату тензора для pytorch
            ]
        )
        self.test_transform = A.Compose(  # с тестовым набором минимум трансформаций
            [
                A.SmallestMaxSize(max_size=256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        classes = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

        cls = [os.path.join(PATH_DATASETS, "TRAIN", cl) for cl in classes]
        images_filepaths = []
        [images_filepaths.extend(sorted([os.path.join(cls[i], f) for f in os.listdir(cls[i])])) for i in
         range(len(classes))]
        correct_train_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]

        cls = [os.path.join(PATH_DATASETS, "TEST", cl) for cl in classes]
        images_filepaths = []
        [images_filepaths.extend(sorted([os.path.join(cls[i], f) for f in os.listdir(cls[i])])) for i in
         range(len(classes))]
        correct_test_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]

        if stage == "fit" or stage is None:
            self.train_dataset = BloodCellsDataset(images_filepaths=correct_train_images_filepaths,
                                                   transform=self.train_transform)
        if stage == "test" or stage is None:
            self.test_dataset = BloodCellsDataset(images_filepaths=correct_test_images_filepaths,
                                                  transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                           pin_memory=True, )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                           pin_memory=True, )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                           pin_memory=True, )


