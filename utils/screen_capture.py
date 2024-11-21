import cv2
import mss
import numpy as np
import torch
from config import (
    MONITOR_CONFIG,
    RESCALED_WIDTH,
    RESCALED_HEIGHT,
    HEALTH_BAR_CROP,
    E_BUTTON_CROP,
)
from in_game_rewards import extract_health, check_e_button
from image_processing import image_crop

### Finish button implementation


def grab_screen(p_input):
    sct = mss.mss()  # Create mss instance inside the process
    first_measure = True
    while True:
        # Capture the full screen or specified monitor region
        img = np.array(sct.grab(MONITOR_CONFIG))

        # Make a copy of the image and crop out the health bar
        health_bar = image_crop(img, HEALTH_BAR_CROP)
        button = image_crop(img, E_BUTTON_CROP)

        button_displayed = check_e_button(
            button
        )  # None if no button is displayed, otherwise the button name

        if first_measure:
            total_health = extract_health(health_bar, first_measure)
            first_measure = False
        # Extract the health level as a percentage
        health = extract_health(health_bar, first_measure, total_health)

        img = cv2.resize(img, (RESCALED_WIDTH, RESCALED_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Convert the full image to a tensor for the model
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Send the image and health percentage through the pipe
        p_input.send((img_tensor, health))
