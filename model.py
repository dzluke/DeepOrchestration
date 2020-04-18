import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet, count_parameters, init_weights, device


class CNN(nn.Module):
    def __init__(self, out_num, features_dim):
        super(CNN, self).__init__()

        shape_input = features_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        shape_input = (shape_input[0]//2, shape_input[1]//2)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        shape_input = (shape_input[0]//2, shape_input[1]//2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        shape_input = (shape_input[0]//2, shape_input[1]//2)
        self.lstm_shape = (shape_input[0], shape_input[1])

        self.lstm = nn.LSTM(input_size=shape_input[1], hidden_size=shape_input[1], batch_first=True)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        shape_input = (shape_input[0]//2, shape_input[1]//2)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32*shape_input[0]*shape_input[1], out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(in_features=2048, out_features=out_num)

        self.dropout = nn.Dropout(0.5)
        self.activation = F.sigmoid

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)

        out = out.view(-1, self.lstm_shape[0], self.lstm_shape[1])
        #print(out.shape)
        out, _ = self.lstm(out)

        #print(out.shape)
        out = out.view(-1, 32, self.lstm_shape[0], self.lstm_shape[1])
#        print(out.shape)
        out = self.layer4(out)
        #print(out.shape)

        #print(out.shape)
        out = out.contiguous().view(out.size()[0], -1)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc3(out)
        #print(out.shape)
        out = self.activation(out)
        #print(out.shape)

        return out


class OrchMatchNet(nn.Module):
    def __init__(self, out_num, model_select, features_dim):
        super(OrchMatchNet, self).__init__()
        if model_select == 'cnn':
            self.net = CNN(out_num, features_dim)
        elif model_select == 'resnet':
            self.net = ResNet(num_classes=out_num)

    def forward(self, x):
        out = self.net(x)

        return out

if __name__ == "__main__":
    cnn = CNN(out_num=505)
    init_weights(cnn)
    cnn.to(device)

    print("Network output shape:")
    # `test_tensor` was trying to be imported from resnet.py
    # this will have to be fixed if you want to run the main method of this file
    print(cnn(test_tensor).shape)

    count_parameters(cnn)
