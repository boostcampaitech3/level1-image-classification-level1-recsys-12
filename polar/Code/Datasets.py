import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, img_paths, classes, transform=None):
        self.img_paths = img_paths
        self.classes = classes
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])

        if self.transform:
            image = self.transform(image)

        return image, self.classes[idx]

    def __len__(self):
        return len(self.img_paths)