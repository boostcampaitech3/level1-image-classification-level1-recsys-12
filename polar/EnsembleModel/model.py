import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


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
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.modelA = torchvision.models.resnet18(pretrained=True)
        self.modelB = torchvision.models.regnet_y_8gf(pretrained=True)
        self.modelC = torchvision.models.efficientnet_b5(pretrained=True)

        self.modelA.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.modelA.fc.weight)
        stdv = 1 / np.sqrt(self.modelA.fc.weight.size(1))
        self.modelA.fc.bias.data.uniform_(-stdv, stdv)

        self.modelB.fc = nn.Linear(in_features=2016, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.modelB.fc.weight)
        stdv = 1 / np.sqrt(self.modelB.fc.weight.size(1))
        self.modelB.fc.bias.data.uniform_(-stdv, stdv)

        self.modelC.classifier[1] = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.modelC.classifier[1].weight)
        stdv = 1 / np.sqrt(2048)
        self.modelC.classifier[1].bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x1 = self.modelA(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.modelC(x)
        x3 = x3.view(x3.size(0), -1)
        x = (x1+x2+x3)/3

        return x


_model_version = {
    "resnet": {
        18: torchvision.models.resnet18(pretrained=True),
        34: torchvision.models.resnet34(pretrained=True),
        50: torchvision.models.resnet50(pretrained=True),
        152: torchvision.models.resnet152(pretrained=True)
    },
    "efficientnet": {
        0: torchvision.models.efficientnet_b0(pretrained=True),
        1: torchvision.models.efficientnet_b1(pretrained=True),
        2: torchvision.models.efficientnet_b2(pretrained=True),
        3: torchvision.models.efficientnet_b3(pretrained=True),
        4: torchvision.models.efficientnet_b4(pretrained=True),
        5: torchvision.models.efficientnet_b5(pretrained=True),
    }
}


class ResNet(nn.Module):
    def __init__(self, num_classes, version=18):
        super(ResNet, self).__init__()
        self.net = _model_version['resnet'][version]
        self.net.fc = nn.Linear(in_features=self.net.fc.in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes, version=3):
        super(EfficientNet, self).__init__()
        self.net = _model_version['efficientnet'][version]
        self.net.classifier[1] = nn.Linear(in_features=self.net.classifier[1].in_features,
                                           out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)
