import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math
from resnet import resnet, ResNet, count_parameters, init_weights, device, test_tensor


class CNN(nn.Module):
    def __init__(self, out_num):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.lstm = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32*21*8, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(in_features=2048, out_features=out_num)

        self.dropout = nn.Dropout(0.5)
        self.activation = F.softmax

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(-1, 43, 16)
        out, _ = self.lstm(out)

        out = out.contiguous().view(-1, 32, 43, 16)
        out = self.layer4(out)

        out = out.contiguous().view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc3(out)
        out = self.activation(out, dim=1)

        return out


class OrchMatchNet(nn.Module):
    def __init__(self, out_num, model_select):
        super(OrchMatchNet, self).__init__()
        if model_select == 'cnn':
            self.net = CNN(out_num)
        elif model_select == 'resnet':
            self.net = ResNet(num_classes=out_num)

    def forward(self, x):
        out = self.net(x)

        return out


cnn = CNN(out_num=505)
init_weights(cnn)
cnn.to(device)

print("Network output shape:")
print(cnn(test_tensor).shape)

count_parameters(cnn)
