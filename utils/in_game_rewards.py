import cv2
import numpy as np


def extract_health(cropped_health_bar, first_measure=False, total_pixels=None):
    """
    Extract the red channel intensity to estimate health.
    """
    # Convert the cropped health bar image to HSV color space
    hsv_image = cv2.cvtColor(cropped_health_bar, cv2.COLOR_BGR2HSV)

    # Define the range of red in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Calculate the proportion of red pixels (as a proxy for health)
    red_pixels = cv2.countNonZero(red_mask)

    if first_measure:
        return red_pixels

    else:
        health_ratio = red_pixels / total_pixels  # Health as a percentage (0.0 to 1.0)
        return health_ratio


def check_e_button(cropped_img):
    """
    Compare the capture to the images from screenshots folder and return which button is displayed.
    If no button is displayed, return None.
    """
