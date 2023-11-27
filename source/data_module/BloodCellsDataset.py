from torch.utils.data import Dataset, DataLoader
import cv2
import os


class BloodCellsDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        """
        Initializes the BloodCellsDataset.

        Args:
        - images_filepaths (list): List of file paths for the images.
        - transform (optional): Image transformation pipeline.
        """
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
        - int: Number of images in the dataset.
        """
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        """
        Retrieves the image and its label at the specified index.

        Args:
        - idx (int): Index of the image.

        Returns:
        - tuple: A tuple containing the image and its label.
        """
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "EOSINOPHIL":
            label = 0
        elif os.path.normpath(image_filepath).split(os.sep)[-2] == "LYMPHOCYTE":
            label = 1
        elif os.path.normpath(image_filepath).split(os.sep)[-2] == "MONOCYTE":
            label = 2
        elif os.path.normpath(image_filepath).split(os.sep)[-2] == "NEUTROPHIL":
            label = 3
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label