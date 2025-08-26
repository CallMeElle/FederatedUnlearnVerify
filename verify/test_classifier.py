import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
import argparse
from tqdm import tqdm
import numpy as np


#models pre and post unlearning
model_pre = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, input_channel= 3)
model_pre.load_state_dict(torch.load('None/Cifar10/backdoor/baseline.pth'))
model_pre.eval()
