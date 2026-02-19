import torch
from torch import nn


def _build_activation(activation_cls):
    # Matches torchvision defaults while accommodating other activations.
    try:
        return activation_cls(inplace=True)
    except TypeError:
        return activation_cls()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = _build_activation(activation)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.activation = _build_activation(activation)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, class_num=5, image_channel=3, activation=nn.ReLU):
        super().__init__()
        self.inplanes = 64
        self.activation_cls = activation

        self.conv1 = nn.Conv2d(image_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = _build_activation(activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, class_num)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.activation_cls)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=self.activation_cls))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(class_num=5, image_channel=3, activation=nn.ReLU):
    return ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num, image_channel=image_channel, activation=activation)


def resnet34(class_num=5, image_channel=3, activation=nn.ReLU):
    return ResNet(BasicBlock, [3, 4, 6, 3], class_num=class_num, image_channel=image_channel, activation=activation)


def resnet50(class_num=5, image_channel=3, activation=nn.ReLU):
    return ResNet(Bottleneck, [3, 4, 6, 3], class_num=class_num, image_channel=image_channel, activation=activation)


def resnet101(class_num=5, image_channel=3, activation=nn.ReLU):
    return ResNet(Bottleneck, [3, 4, 23, 3], class_num=class_num, image_channel=image_channel, activation=activation)


def resnet152(class_num=5, image_channel=3, activation=nn.ReLU):
    return ResNet(Bottleneck, [3, 8, 36, 3], class_num=class_num, image_channel=image_channel, activation=activation)