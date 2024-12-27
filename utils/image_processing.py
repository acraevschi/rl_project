import time
import torch


def image_crop(img, crop_dims):
    return img[crop_dims[1] : crop_dims[3], crop_dims[0] : crop_dims[2]]


def process_image(model, img):
    with torch.no_grad():
        output = model(img)
    return output.squeeze(0).cpu().numpy()


def process_stream(p_output, model, print_fps=False):
    if print_fps:
        f = open("fps.txt", "w")
    while True:
        # Receive the tuple (img_tensor, health_percentage)
        img = p_output.recv()
        start_time = time.time() if print_fps else None

        # Process the image using the model
        processed_img = process_image(model, img)

        if print_fps:
            end_time = time.time()
            time_taken = end_time - start_time
            fps = 1 / time_taken if time_taken > 0 else 1 / 0.001

            # Print FPS to the file
            print(f"Frames per second (FPS): {fps:.2f}", file=f)
    if print_fps:
        f.close()
