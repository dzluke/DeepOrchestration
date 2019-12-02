import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CNN(nn.Module):
    def __init__(self, out_num):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # # bottleneck start
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=16,
        #               kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16,
        #               kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )

        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=64,
        #               kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(64)
        #     # nn.ReLU()
        # )
        # # bottleneck end

        self.lstm = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)

        self.layer7 = nn.Sequential(
            nn.Linear(in_features=64*16*16, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.layer8 = nn.Linear(4096, out_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # res = out
        out = out.view(-1, 64*16, 16)
        out, _ = self.lstm(out)

        # out = out.contiguous().view(-1, 32, 32, 32)
        # out += res
        # out = self.relu(out)

        out = out.contiguous().view(out.size()[0], -1)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        )

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, out_num):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = self._make_layer(block, 64, layers[0])
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[3], stride=2)

        self.classifier = nn.Linear(2048, out_num)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        # residual part
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)

        x = F.dropout(x, 0.5)
        x = self.classifier(x)

        return x


class OrchMatchNet(nn.Module):
    def __init__(self, out_num, model_select):
        super(OrchMatchNet, self).__init__()
        if model_select == 'cnn':
            self.net = CNN(out_num)
        elif model_select == 'resnet':
            self.net = ResNet(Bottleneck, [3, 4, 6, 3], out_num)

    def forward(self, x):
        out = self.net(x)

        return out
