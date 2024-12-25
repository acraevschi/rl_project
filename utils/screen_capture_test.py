# This is here to find out the coordinates of the happiness bar in the screenshot

import cv2
import mss
import numpy as np
import torch
from config import MONITOR_CONFIG, RESCALED_WIDTH, RESCALED_HEIGHT, ORIGINAL_RESOLUTION
import time


# Coordinates for cropping the health bar (to be adjusted based on your screen setup)
# Example coordinates: (x1, y1, x2, y2)

original_width, original_height = ORIGINAL_RESOLUTION

HAPPY_BAR = (
    57,
    -73,
    357,
    -53,
)  # Adjust these based on the position of the health bar in the screenshot


def grab_screen():
    sct = mss.mss()  # Create mss instance inside the process
    # while True:
    # Capture the full screen or specified monitor region
    img = np.array(sct.grab(MONITOR_CONFIG))
    # height, width, _ = img.shape
    happy_bar = img[
        HAPPY_BAR[1] : HAPPY_BAR[3],
        HAPPY_BAR[0] : HAPPY_BAR[2],
    ]

    img = cv2.resize(img, (RESCALED_WIDTH, RESCALED_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Make a copy of the image and crop out the health bar

    # Display the cropped health bar for visual inspection
    # cv2.imshow("Cropped Health Bar", health_bar)

    # Convert the full image to a tensor for the model
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # Send the image and health percentage through the pipe
    # p_input.send((img_tensor, health_percentage))

    # # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

    # Release resources
    cv2.destroyAllWindows()

    return happy_bar


#### Play around here

import matplotlib.pyplot as plt

time.sleep(1.5)
# coordinates: (x1, y1, x2, y2); нужо правее и ниже (потом отдельно для населения)
HAPPY_BAR = (
    1200,
    -73,
    1550,
    -53,
)  # Adjust these based on the position of the health bar in the screenshot

sct = mss.mss()  # Create mss instance inside the process
# while True:
# Capture the full screen or specified monitor region
img = np.array(sct.grab(MONITOR_CONFIG))
# height, width, _ = img.shape
happy_bar = img[
    HAPPY_BAR[1] : HAPPY_BAR[3],
    HAPPY_BAR[0] : HAPPY_BAR[2],
]

plt.imshow(cv2.cvtColor(happy_bar, cv2.COLOR_BGR2RGB))
plt.title("Reward info")
plt.axis("off")
plt.show()
