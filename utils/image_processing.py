import time
import torch


def process_image(model, img):
    with torch.no_grad():
        output = model(img)
        return output.squeeze(0).cpu().numpy()


def process_stream(p_output, model, print_fps=True):
    while True:
        img = p_output.recv()
        start_time = time.time() if print_fps else None

        processed_img = process_image(model, img)

        if print_fps:
            end_time = time.time()
            time_taken = end_time - start_time
            fps = 1 / time_taken

            print(f"Time taken to process one frame: {time_taken:.4f} seconds")
            print(f"Frames per second (FPS): {fps:.2f}")
