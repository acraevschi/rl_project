import cv2
import mss
import numpy as np
import torch
from config import MONITOR_CONFIG, RESCALED_WIDTH, RESCALED_HEIGHT


def grab_screen(p_input):
    sct = mss.mss()  # Create mss instance inside the process
    while True:
        img = np.array(sct.grab(MONITOR_CONFIG))
        img = cv2.resize(img, (RESCALED_WIDTH, RESCALED_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1).unsqueeze(0)
        p_input.send(img)
