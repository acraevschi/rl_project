# This is here for testing purposes only

import multiprocessing
from multiprocessing import Pipe
import cv2
import mss
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Set up the screen capture area (adjust as needed)
original_resolution = (1920, 1080)
width, height = original_resolution
monitor = {"top": 0, "left": 0, "width": width, "height": height}

# Initialize mss for screen capture
sct = mss.mss()

rescaled_width = int(width / 6)
rescaled_height = int(height / 6)


def compute_dims(dims, kernel_size, stride):
    width = dims[0]
    height = dims[1]
    new_width = (width - kernel_size) // stride + 1
    new_height = (height - kernel_size) // stride + 1
    return int(new_width), int(new_height)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Depthwise convolution (groups=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )  # Pointwise conv to change channels
        self.bn1 = nn.GroupNorm(4, out_channels)

        # Another depthwise + pointwise combination
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
        )
        self.bn2 = nn.GroupNorm(4, out_channels)

        # Skip connection (only applies if the number of channels changes)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)

        # Apply depthwise and pointwise convolutions
        out = torch.relu(self.bn1(self.pointwise(self.conv1(x))))
        out = self.bn2(self.conv2(out))

        # Add residual connection
        out += residual
        return torch.relu(out)


class SimpleCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # Initial convolution with a larger kernel for downsampling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.GroupNorm(4, 16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # First residual block: 16->32 channels
        self.res1 = ResBlock(16, 32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Second residual block: 32->48 channels
        self.res2 = ResBlock(32, 48)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Adjusted fully connected layer dimensions
        self.fc1 = nn.Linear(10560, 192)  # Adjust based on input size
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(192, num_actions)

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


# Instantiate and evaluate the optimized CNN model
model = SimpleCNN(num_actions=10).eval()


# Forward pass through the model
def process_image(img):
    with torch.no_grad():
        output = model(img)

    # For visualization, convert output tensor back to NumPy (if needed)
    output_np = output.squeeze(0).cpu().numpy()
    return output_np


def grab_screen(p_input):
    while True:
        # Capture screen
        img = np.array(sct.grab(monitor))
        img = cv2.resize(
            img, (rescaled_width, rescaled_height)
        )  # Resize image to model input size
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Convert to RGB
        # Convert image from NumPy array to Torch tensor
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1).unsqueeze(0)  # Change shape to (1, 3, H, W)
        # Send the image to the input pipe
        p_input.send(img)


def model_processing(p_output):
    # Create a single window for display
    # cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)

    while True:
        # Receive image from pipe
        img = p_output.recv()

        # Measure the start time
        start_time = time.time()

        # Process image with the CNN model
        processed_img = process_image(img)

        # Measure the end time
        end_time = time.time()

        # Calculate the time difference
        time_taken = end_time - start_time

        # Calculate frames per second (FPS)
        fps = 1 / time_taken

        print(f"Time taken to process one frame: {time_taken:.4f} seconds")
        print(f"Frames per second (FPS): {fps:.2f}")

        # Display the processed image (if needed)
        # cv2.imshow("Processed Image", processed_img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == "__main__":
    # Set up pipes for multiprocessing
    p_output, p_input = Pipe()

    # Create processes for screen capture and model processing
    p1 = multiprocessing.Process(target=grab_screen, args=(p_input,))
    p2 = multiprocessing.Process(target=model_processing, args=(p_output,))

    # Start processes
    p1.start()
    p2.start()

    # Ensure processes end together
    p1.join()
    p2.join()
