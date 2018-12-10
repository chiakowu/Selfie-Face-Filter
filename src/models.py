import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module):
    # best now is 3 class 4 layers
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 7, padding=5)
        self.conv2 = nn.Conv2d(6, 12, 5, padding=3)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def __init__(self):
    #     super(ConvNet, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, kernel_size=7, padding=5)
    #     self.conv2 = nn.Conv2d(6, 6, kernel_size=5, padding=3)
    #     self.conv3 = nn.Conv2d(6, 12, kernel_size=5, padding=3)
    #     self.conv3_bn = nn.BatchNorm2d(12)
    #     self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
    #     self.conv5 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
    #     self.conv5_bn = nn.BatchNorm2d(16)
    #     self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    #     self.conv7 = nn.Conv2d(16, 24, 3)
    #     self.conv7_bn = nn.BatchNorm2d(24)
    #     self.max_pool = nn.MaxPool2d(2)
    #     self.fc1 = nn.Linear(32, 3)
    #     # self.fc2 = nn.Linear(96, 7)
    #     #pycharm four, terminal 3
    # def forward(self, x):
    #     x = self.max_pool(F.relu(self.conv1(x)))
    #     x = self.max_pool(F.relu(self.conv2(x)))
    #     x = self.max_pool(F.relu(self.conv3_bn(self.conv3(x))))
    #     x = self.max_pool(F.relu(self.conv4(x)))
    #     x = self.max_pool(F.relu(self.conv5_bn(self.conv5(x))))
    #     x = self.max_pool(F.relu(self.conv6(x)))
    #     x = self.max_pool(F.relu(self.conv7_bn(self.conv7(x))))
    #     x = x.view(-1, 32)
    #     x = self.fc1(x)
    #     # x = self.fc2(x)
    #     return x
    # PyCharm emotion net
    # terminal 5 layers

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self, learning_rate, momentum, weight_decay):
        return optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    def adjust_learning_rate(self, optimizer, epoch, lr):
        lr = lr * pow(0.9, epoch / 50)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
