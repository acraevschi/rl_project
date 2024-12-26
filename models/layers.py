import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.bn1 = nn.GroupNorm(4, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=True)
        self.bn2 = nn.GroupNorm(4, out_channels)

        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=True) if in_channels != out_channels else nn.Sequential()

    def forward(self, x):
        residual = self.skip(x)
        out = torch.relu(self.bn1(self.pointwise(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)
