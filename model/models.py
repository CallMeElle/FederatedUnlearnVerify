"""
Source: https://github.com/weiaicunzai/pytorch-cifar100
"""
from torch import nn
import torch
from model.resnet import ResNet, BasicBlock, BottleNeck
import torch.nn.functional as F


def ResNet18(num_classes, input_channels):
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channel= input_channels)


def ResNet34(num_classes, input_channels):
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes= num_classes, input_channel= input_channels)


def ResNet50(num_classes, input_channels):
    """return a ResNet 50 object"""
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes= num_classes, input_channel= input_channels)


def ResNet101(num_classes, input_channels):
    """return a ResNet 101 object"""
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes= num_classes, input_channel= input_channels)


def ResNet152(num_classes, input_channels):
    """return a ResNet 152 object"""
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes= num_classes, input_channel= input_channels)


class LinearModelTabular(nn.Module):
    def __init__(self, input_features=8, hidden_layer1=20, hidden_layer2=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden_layer1)
        self.f_connected2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x