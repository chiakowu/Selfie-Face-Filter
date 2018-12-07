import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
import src.dataloader as dataloader


class CoolNet(nn.module):
    def __init__(self):
        super(CoolNet, self).__init__()
        # TODO: Define model here
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=3)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=3)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 16, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.max_pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
