import torch
import torchvision.transforms as transforms
from torchvision.transforms import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
#             Resize(resize),
            CenterCrop((350 , 250)),
#             ColorJitter(0.1, 0.1, 0.1, 0.1),
#             RandomHorizontalFlip(),
#             RandomRotation(10),
#             RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            ToTensor(),
            Normalize(mean=mean, std=std),
            ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
    
class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
    
# residual block
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels))

        if self.stride != 1 or self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(self.out_channels))

    def forward(self, x):
        out = self.conv_block(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            x = self.downsample(x)

        out = F.relu(x + out)
        return out
    

    # ResNet
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=18):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.base = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.gap = nn.AvgPool2d(4) # 4: 필터 사이즈
        self.fc1 = nn.Linear(10240, 512)
#         self.fc1 = nn.Linear(30720, 512)
#         self.fc1 = nn.Linear(6144, 512)
#         self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            block = ResidualBlock(self.in_channels, out_channels, stride)
            layers.append(block)
            self.in_channels = out_channels
    
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.base(x)
#         print('base :', out.size())
        out = self.layer1(out)
#         print('layer1 :', out.size())
        out = self.layer2(out)
#         print('layer2 :', out.size())
        out = self.layer3(out)
#         print('layer3 :', out.size())
        out = self.layer4(out)
#         print('layer4:', out.size())
        out = self.gap(out)
#         print('avgpool :', out.size())
        out = out.view(out.size(0), -1)
#         print('view :', out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        return out