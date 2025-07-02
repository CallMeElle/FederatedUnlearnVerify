import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


def create_directory_if_not_exists(file_path: str) -> None:
    # Check the directory exist,
    # If not then create the directory
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory and its parent directories if necessary
        os.makedirs(directory)
        print(f"Created new directory: {file_path}")


def generate_heatmap(
        heatmap: torch.Tensor,
        image: torch.Tensor,
        alpha: float) -> np.array:

    # Detach to cpu in numpy array for opencv processing
    heatmap = heatmap.cpu().numpy()
    original_image = image.cpu().numpy().transpose(1, 2, 0)

    heatmap_image = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_image = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET)
    image_norm = cv2.cvtColor(original_image * 255, cv2.COLOR_RGB2BGR)
    overlayed_image = cv2.addWeighted(image_norm.astype(np.uint8), 1 - alpha, heatmap_image, alpha, 0)
    overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)

    return overlayed_image


def save_heatmap(
        gradcam_dir: str,
        heatmap_img: np.array,
        batch_idx: int,
        img_idx: int,
        batch_size: int) -> None:

    save_path = f"{gradcam_dir}{img_idx + (batch_size * batch_idx)}.png"
    plt.imshow(heatmap_img)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)