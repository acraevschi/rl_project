import torch
import torch.nn as nn
from .layers import ResBlock


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.bn1 = nn.GroupNorm(4, 32)
        self.maxpool1 = nn.MaxPool2d(2)

        self.res1 = ResBlock(32, 64)
        self.maxpool2 = nn.MaxPool2d(2)

        self.res2 = ResBlock(64, 128)
        self.maxpool3 = nn.MaxPool2d(2)

        # After these layers, the feature map is (128, 22, 40) from a 640x360 input.
        flat_size = 128 * 22 * 40
        self.fc1 = nn.Linear(flat_size, 1024)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.res1(x)
        x = self.maxpool2(x)

        x = self.res2(x)
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x