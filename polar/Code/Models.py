import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(CustomModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Ref : https://deep-learning-study.tistory.com/561
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class DepthWise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

        self.seblock = SEBlock(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.seblock(x) * x
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, alpha=1, num_classes=10, init_weights=True):
        super(MobileNet, self).__init__()

        self.init_weights = init_weights

        self.conv1 = BasicConv2d(3, int(32*alpha), 3, stride=2, padding=1)
        self.conv2 = DepthWise(int(32*alpha), int(64*alpha), stride=1)
        # down sample
        self.conv3 = nn.Sequential(
            DepthWise(int(64*alpha), int(128*alpha), stride=2),
            DepthWise(int(128*alpha), int(128*alpha), stride=1),
        )
        # down sample
        self.conv4 = nn.Sequential(
            DepthWise(int(128 * alpha), int(256 * alpha), stride=2),
            DepthWise(int(256 * alpha), int(256 * alpha), stride=1)
        )
        # down sample
        self.conv5 = nn.Sequential(
            DepthWise(int(256 * alpha), int(512 * alpha), stride=2),
            DepthWise(int(512 * alpha), int(512 * alpha), stride=1),
            DepthWise(int(512 * alpha), int(512 * alpha), stride=1),
            DepthWise(int(512 * alpha), int(512 * alpha), stride=1),
            DepthWise(int(512 * alpha), int(512 * alpha), stride=1),
            DepthWise(int(512 * alpha), int(512 * alpha), stride=1),
        )
        # down sample
        self.conv6 = nn.Sequential(
            DepthWise(int(512 * alpha), int(1024 * alpha), stride=2)
        )
        # down sample
        self.conv7 = nn.Sequential(
            DepthWise(int(1024 * alpha), int(1024 * alpha), stride=2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(1024 * alpha), num_classes)

        # weights initialization
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

        # weights initialization function

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

