import torch
import torch.nn as nn
import numpy as np


# Multi-lables classification:
# class number: N kinds of sound


class OrchMatchNet(nn.Module):
    def __init__(self, out_num):
        super(OrchMatchNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024)
            nn.ReLU()
            nn.Dropout(0.5)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256)
            nn.ReLU()
            nn.Dropout(0.5)
        )

        self.layer7 = nn.Linear(256, out_num)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size()[0], -1)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out
