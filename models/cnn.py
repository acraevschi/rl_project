import torch
import torch.nn as nn
from .layers import ResBlock


class SimpleCNN(nn.Module):
    def __init__(
        self, num_actions, input_shape
    ):  # Default shape based on your current rescaled dimensions
        super().__init__()

        # Store the layer configurations
        self.conv_configs = [
            # (in_channels, out_channels, kernel_size, stride, padding)
            (input_shape[0], 16, 7, 2, 3),  # conv1
        ]

        self.pool_configs = [
            # (kernel_size, stride)
            (2, 2),  # maxpool1
            (2, 2),  # maxpool2
            (2, 2),  # maxpool3
        ]

        self.res_configs = [
            # (in_channels, out_channels)
            (16, 32),  # res1
            (32, 48),  # res2
        ]

        # Initialize layers
        self.conv1 = nn.Conv2d(*self.conv_configs[0])
        self.bn1 = nn.GroupNorm(4, 16)
        self.maxpool1 = nn.MaxPool2d(*self.pool_configs[0])

        self.res1 = ResBlock(*self.res_configs[0])
        self.maxpool2 = nn.MaxPool2d(*self.pool_configs[1])

        self.res2 = ResBlock(*self.res_configs[1])
        self.maxpool3 = nn.MaxPool2d(*self.pool_configs[2])

        # Calculate the size of the flattened features
        flatten_size = self._get_flatten_size(input_shape)

        self.fc1 = nn.Linear(flatten_size, 192)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(192, num_actions)

    def _get_flatten_size(self, input_shape):
        """Calculate the size of the flattened features dynamically."""
        # Create a dummy tensor to track the size through the network
        x = torch.zeros(1, *input_shape)

        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.res1(x)
        x = self.maxpool2(x)

        x = self.res2(x)
        x = self.maxpool3(x)

        # Get the flattened size
        return x.numel() // x.size(0)  # Divide by batch size

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.res1(x)
        x = self.maxpool2(x)

        x = self.res2(x)
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
