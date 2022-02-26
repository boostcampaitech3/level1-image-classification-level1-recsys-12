import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, img_paths, resize):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize),
            CenterCrop((100, 100)),
            ToTensor(),
            ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)