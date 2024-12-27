import cv2
import mss
import numpy as np
import torch
from config import (
    MONITOR_CONFIG,
    RESCALED_WIDTH,
    RESCALED_HEIGHT,
    HAPPY_BAR,
    ECON_BAR,
)
from reward.in_game_rewards import * # TBA
from image_processing import image_crop

### Requires many changes, as it was not fully implemented yet


def grab_screen(p_input):
    sct = mss.mss()  # Create mss instance inside the process
    first_measure = True
    while True:
        # Capture the full screen or specified monitor region
        img = np.array(sct.grab(MONITOR_CONFIG))

        # Make a copy of the image and crop out the health bar
        happy = image_crop(img, HAPPY_BAR)
        economy = image_crop(img, ECON_BAR)


        img = cv2.resize(img, (RESCALED_WIDTH, RESCALED_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Convert the full image to a tensor for the model
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Send the image and health percentage through the pipe
        p_input.send((img_tensor, happy, economy))
