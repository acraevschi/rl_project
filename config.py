ORIGINAL_RESOLUTION = (1920, 1080)
WIDTH, HEIGHT = ORIGINAL_RESOLUTION

MONITOR_CONFIG = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}

RESCALED_WIDTH = 320
RESCALED_HEIGHT = 180

INPUT_SHAPE = (3, RESCALED_HEIGHT, RESCALED_WIDTH)

NUM_ACTIONS = 10 

# Coordinates for cropping the rewards bars: happiness and economy (based on 1080p resolution)
### Simply reshape the screenshot to 1080p each time before cropping
HAPPY_BAR = (
    1240,
    -40,
    1525,
    -5,
)

ECON_BAR = (
    960,
    -40,
    1215,
    -5,
)
