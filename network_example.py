import os, sys, pickle, time, librosa, torch, numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.categorical import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {counts:,} trainable parameters')

def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out
        
class ResNet(nn.Module):
    def __init__(self, num_classes=3000, dropout_rate=0.5, output_nonlinearity='sigmoid'):
        super(ResNet, self).__init__()
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )
        
        if output_nonlinearity is 'sigmoid':
            self.output_nonlinearity=torch.sigmoid
        elif output_nonlinearity is 'softmax':
            self.output_nonlinearity=torch.softmax
        else:
            raise Exception("")

        self.bn1 = nn.BatchNorm2d(32)

        self.block1 = self._create_block(32, 32, stride=1)
        self.block2 = self._create_block(32, 16, stride=2)
        self.block3 = self._create_block(16, 16, stride=2)
        self.block4 = self._create_block(16, 16, stride=2)
        self.bn2 = nn.BatchNorm1d(320)
        self.bn3 = nn.BatchNorm1d(100)
        self.linear1 = nn.Linear(320, 100)
        self.linear2 = nn.Linear(100, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.output_nonlinearity(out)
        return out

    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)
        
# Make a test input for the network and compute the features
import librosa
sr = 22050 # sample rate
test_audio_signal = librosa.core.tone(1000, sr=sr, duration=4)

test_spec = librosa.feature.melspectrogram(test_audio_signal)
print("Melspec Feature Shape:")
print(test_spec.shape)


# Expected input size to the network is [batch_size, 128, 173]
batch_size = 20
batch_specs = np.tile(test_spec, (batch_size,1,1))
print("Batch Feature shape:")
print(batch_specs.shape)

# Network expects a channel dimension for the inputs, so add a dimension
test_tensor = torch.from_numpy(batch_specs).float().to(device)
test_tensor = test_tensor.unsqueeze(1)
print("Network input shape:")
print(test_tensor.shape)

resnet = ResNet(num_classes=3000, dropout_rate=0.5)
init_weights(resnet)
resnet.set_device(device)

print("Network output shape:")
print(resnet(test_tensor).shape)

count_parameters(resnet)