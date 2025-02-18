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
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# Initialize mss for screen capture
sct = mss.mss()

rescaled_width = 640
rescaled_height = 360

# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_dims(dims, kernel_size, stride):
    width = dims[0]
    height = dims[1]
    new_width = (width - kernel_size) // stride + 1
    new_height = (height - kernel_size) // stride + 1
    return int(new_width), int(new_height)

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

model = SimpleCNN().to("cuda" if torch.cuda.is_available() else "cpu").eval()

# Forward pass through the model
def process_image(img):
    # img.to("cuda" if torch.cuda.is_available() else "cpu")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model(img)

    # For visualization, convert output tensor back to NumPy (if needed)
    output_np = output.squeeze(0).to("cpu").numpy()
    return output_np

# Instantiate and evaluate the optimized CNN model

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
        img = img.to("cuda" if torch.cuda.is_available() else "cpu")
        # Measure the start time
        start_time = time.time()

        # Process image with the CNN model
        processed_img = process_image(img)

        # Measure the end time
        end_time = time.time()

        # Calculate the time difference
        time_taken = end_time - start_time

        # Calculate frames per second (FPS)
        fps = 1 / time_taken if time_taken > 0 else 1/0.001

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
