import multiprocessing
from multiprocessing import Pipe
import torch
from models.cnn import SimpleCNN
from utils.screen_capture import grab_screen
from utils.image_processing import process_stream
from config import NUM_ACTIONS, INPUT_SHAPE


def main():
    # Make sure to use spawn method for Windows
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize model
    model = SimpleCNN(num_actions=NUM_ACTIONS, input_shape=INPUT_SHAPE).eval()

    # Set up pipes for multiprocessing
    p_output, p_input = Pipe()

    # Create processes
    p1 = multiprocessing.Process(target=grab_screen, args=(p_input,))
    p2 = multiprocessing.Process(target=process_stream, args=(p_output, model))

    try:
        # Start processes
        p1.start()
        p2.start()

        # Wait for processes to complete
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("\nStopping processes...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
    finally:
        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()


if __name__ == "__main__":
    main()
