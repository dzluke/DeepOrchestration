import torch
import numpy as np
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(-1, self.lstm_shape[0], self.lstm_shape[1])
        out, _ = self.lstm(out)

        out = out.view(-1, 32, self.lstm_shape[0], self.lstm_shape[1])
        out = self.layer4(out)
        
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc3(out)
        out = self.activation(out)

        return out
    
    def getLatentSpace(self, x):
        res = []
        
        out = self.layer1(x)
        res.append(out.detach().cpu().numpy())
        out = self.layer2(out)
        res.append(out.detach().cpu().numpy())
        out = self.layer3(out)
        res.append(out.detach().cpu().numpy())

        out = out.view(-1, self.lstm_shape[0], self.lstm_shape[1])
        out, _ = self.lstm(out)

        out = out.view(-1, 32, self.lstm_shape[0], self.lstm_shape[1])
        res.append(out.detach().cpu().numpy())
        out = self.layer4(out)
        res.append(out.detach().cpu().numpy())
        
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc3(out)
        out = self.activation(out)
        res.append(out.detach().cpu().numpy())
        return res


class OrchMatchNet(nn.Module):
    def __init__(self, out_num, model_select, features_dim):
        super(OrchMatchNet, self).__init__()
        if model_select == 'cnn':
            self.net = CNN(out_num, features_dim)
        elif model_select == 'resnet':
            self.net = ResNet(num_classes=out_num, features_dim=features_dim)

    def forward(self, x):
        out = self.net(x)

        return out
    
    def getLatentSpace(self, feature):
        return self.net.getLatentSpace(feature)


class SemanticLoss(nn.Module):
    def __init__(self):
        super(SemanticLoss, self).__init__()

    def forward(self, probs, constraints):
        """

        :param probs: output of network. has shape (batch size, num classes)
        :param constraints: binary 0/1 vector. has shape (batch size, num classes)
        :return: loss (scalar)
        """
        mask = constraints == 1
        inverse_mask = torch.logical_not(constraints)
        literals = probs[mask]  # probs of samples we want to include
        num_literals = literals.shape[0]
        negated_literals = probs[inverse_mask]  # probs of samples we don't want
        negated_literals = torch.sub(torch.ones(negated_literals.shape), negated_literals)  # (1 - p)
        literals = torch.sum(torch.log(literals))  # log(xy) = log(x) + log(y)
        negated_literals = torch.sum(torch.log(negated_literals))
        negated_literals = torch.mul(negated_literals, num_literals)  # log(x^y) = y*log(x)
        loss = torch.add(literals, negated_literals)  # p + (1 - p)
        loss = torch.mul(loss, torch.full(loss.shape, -1))
        loss = torch.div(loss, torch.full(loss.shape, probs.shape[0]))  # average by diving by batch size
        return loss  # scalar


class CustomLoss(nn.Module):
    """
    A loss function that is a sum of Binary Cross Entropy Loss and Semantic Loss
    CustomLoss = BCELoss + w * SemanticLoss
    """
    def __init__(self, semantic_weight):
        super(CustomLoss, self).__init__()
        self.semantic_weight = semantic_weight
        self.BCELoss = nn.BCELoss()
        self.SemanticLoss = SemanticLoss()

        self.bce_loss_list = []
        self.semantic_loss_list = []

    def forward(self, probs, labels, constraints):
        bce_loss = self.BCELoss(probs, labels)
        semantic_loss = self.SemanticLoss(probs, constraints)
        semantic_loss = self.semantic_weight * semantic_loss
        # print("------------------------")
        # print("BCE Loss: {:.3f}, Semantic Loss: {:.3f}".format(bce_loss, semantic_loss))
        self.bce_loss_list.append(bce_loss)
        self.semantic_loss_list.append(semantic_loss)
        return bce_loss + semantic_loss
