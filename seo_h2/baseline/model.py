import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class resnet18(nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(in_features=512, out_features= num_classes, bias=True)
            torch.nn.init.kaiming_uniform_(self.model.fc.weight, nonlinearity='relu')
            # torch.nn.init.xavier_uniform_(self.model.fc.weight)
            stdv = 1/np.sqrt(512)
            self.model.fc.bias.data.uniform_(-stdv, stdv)

        def forward(self, x):

            return self.model(x)

class resnet34(nn.Module):
        def __init__(self, num_classes):
            super().__init__()

            self.model = torchvision.models.resnet34(pretrained=True)
            self.model.fc = torch.nn.Linear(in_features=512, out_features= num_classes, bias=True)
            torch.nn.init.kaiming_uniform_(self.model.fc.weight, nonlinearity='relu')
            stdv = 1/np.sqrt(512)
            self.model.fc.bias.data.uniform_(-stdv, stdv)

        def forward(self, x):
            return self.model(x)
