from keys.keys import Keys
from time import sleep

action_to_key = {
    0: None,  # Do nothing
    1: "A",  # Move left
    2: "D",  # Move right
    3: "W",  # Move up
    4: "S",  # Move down
    5: "SPACE",  # Dash
}

# Instantiate the Keys object
keys = Keys()


def perform_action(action):
    if action == 0:
        pass
    else:
        # Map action to key
        key = action_to_key.get(action)
        # Simulate pressing the key down and then releasing it
        # You can adjust sleep duration based on how fast you want actions to be
        keys.parseKeyString(f"{key}_DOWN")  # Press the key down
        sleep(0.05)  # Hold for 50 ms (adjust as needed)
        keys.parseKeyString(f"{key}_UP")  # Release the key
