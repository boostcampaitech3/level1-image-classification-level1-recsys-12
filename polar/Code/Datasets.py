import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataSet(Dataset):
    def __init__(self, img_paths, classes=None, transform=None, train=True):
        self.img_paths = img_paths
        self.classes = classes
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])

        if self.transform:
            image = self.transform(image)
        if self.train:
            return image, self.classes[idx]
        else:
            return image

    def __len__(self):
        return len(self.img_paths)


class CustomDataSet2(Dataset):
    def __init__(self, img_paths, classes=None, pre_transform=None, transform=None, train=True):
        self.images = []
        self.classes = classes
        self.transform = transform
        self.train = train

        for path in img_paths:
            image = Image.open(path)
            image = pre_transform(image)
            self.images.append(image)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)
        if self.train:
            return image, self.classes[idx]
        else:
            return image

    def __len__(self):
        return len(self.images)


class EnsembleDataSet(Dataset):
    def __init__(self, df, pre_transforms=None, transforms=None, train=True):
        self.df = df
        self.gender = df['gender'].values
        self.age = df['age'].values
        self.path = df['path'].values
        self.types = df['types'].values
        self.classes = df['class'].values
        self.transforms = transforms
        self.train = train

    def __getitem__(self, idx):
        image = Image.open(self.path[idx])

        if self.transforms:
            image = self.transforms(image)

        if self.train:
            return image, self.gender[idx], self.age[idx], self.types[idx], self.classes[idx]
        else:
            return image

    def __len__(self):
        return len(self.path)
