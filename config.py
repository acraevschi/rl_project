ORIGINAL_RESOLUTION = (1920, 1080)
WIDTH, HEIGHT = ORIGINAL_RESOLUTION

MONITOR_CONFIG = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}

RESCALED_WIDTH = int(WIDTH / 6)
RESCALED_HEIGHT = int(HEIGHT / 6)

INPUT_SHAPE = (3, RESCALED_HEIGHT, RESCALED_WIDTH)

NUM_ACTIONS = 10

# Coordinates for cropping the health bar, e button presses, and so on, coordinates: (x1, y1, x2, y2)
HEALTH_BAR_CROP = (
    57,
    -73,
    357,
    -53,
)

center = WIDTH // 2

E_BUTTON_CROP = (
    center - 125,
    -90,
    center + 125,
    -45,
)
