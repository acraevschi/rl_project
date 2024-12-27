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

MONITOR_CONFIG = {"top": 0, "left": 0, "width": 2560, "height": 1440}

time.sleep(3)
# coordinates: (x1, y1, x2, y2); нужо правее и ниже (потом отдельно для населения)
HAPPY_BAR = (
    1240,
    -40,
    1525,
    -5,
)  # Adjust these based on the position of the health bar in the screenshot

sct = mss.mss()  # Create mss instance inside the process
# while True:
# Capture the full screen or specified monitor region
img = np.array(sct.grab(MONITOR_CONFIG))
img = cv2.resize(
        img, (1920, 1080)
    )
# height, width, _ = img.shape
happy_bar = img[
    HAPPY_BAR[1] : HAPPY_BAR[3],
    HAPPY_BAR[0] : HAPPY_BAR[2],
]

plt.imshow(cv2.cvtColor(happy_bar, cv2.COLOR_BGR2RGB))
plt.title("")
plt.axis("off")
plt.show()


####### ECONOMY BAR

time.sleep(1.5)
# coordinates: (x1, y1, x2, y2); нужо правее и ниже (потом отдельно для населения)
ECON_BAR = (
    960,
    -40,
    1215,
    -5,
)  # Adjust these based on the position of the health bar in the screenshot

sct = mss.mss()  # Create mss instance inside the process
# while True:
# Capture the full screen or specified monitor region
img = np.array(sct.grab(MONITOR_CONFIG))
# height, width, _ = img.shape
econ_bar = img[
    ECON_BAR[1] : ECON_BAR[3],
    ECON_BAR[0] : ECON_BAR[2],
]

plt.imshow(cv2.cvtColor(econ_bar, cv2.COLOR_BGR2RGB))
plt.title("Reward info")
plt.axis("off")
plt.show()

#######################

possible_states = ["very_unhappy", "unhappy", "satisfied", "happy", "very_happy"]
for state in possible_states:
    happy = cv2.imread("reward/happiness_levels/happy.png", cv2.IMREAD_UNCHANGED)
    current_state = cv2.imread("reward/scrns_test/citizen_happy.png", cv2.IMREAD_UNCHANGED)

    match = cv2.matchTemplate(current_state, happy, cv2.TM_CCOEFF_NORMED)

    # Get the best match position
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

    # Define a threshold for a good match
    threshold = 0.7

    if max_val >= threshold:
        print("Match found with confidence:", max_val)
        top_left = max_loc
        bottom_right = (top_left[0] + happy.shape[1], top_left[1] + happy.shape[0])
        cv2.rectangle(current_state, top_left, bottom_right, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(current_state, cv2.COLOR_BGR2RGB))
        plt.title("Match found")
        plt.axis("off")
        plt.show()
    else:
        print("No match found. Confidence:", max_val)


