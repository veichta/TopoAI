import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.data_utils import denormalize_img


def overlay_image_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay image and mask.
    Args:
        img (np.ndarray): Image.
        mask (np.ndarray): Mask.
        alpha (float, optional): Alpha value. Defaults to 0.5.
    Returns:
        np.ndarray: Overlayed image.
    """
    img = img.copy()
    mask = mask.copy()

    img = (1 - alpha) * img + (mask[:, :, None] * 255 * alpha)
    return img


def plot_batch_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    img_mean: np.array,
    img_std: np.array,
    num_images: int = 5,
    filename: str = None,
) -> None:
    num_images = min(num_images, images.shape[0])
    fig, ax = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    for i in range(num_images):
        img = denormalize_img(images[i].detach().cpu(), img_mean, img_std)
        prediction = predictions[i].detach().cpu().numpy()
        overlay = overlay_image_mask(img, prediction)
        # print(overlay)
        overlay = overlay.clip(0, 255).astype(np.uint8)
        ax[i, 0].imshow(overlay)
        ax[i, 1].imshow(prediction)
        ax[i, 2].imshow(weights[i].detach().cpu().numpy() + masks[i].detach().cpu().numpy())

    ax[0, 0].set_title("Image + Mask")
    ax[0, 1].set_title("Mask")
    ax[0, 2].set_title("Weight")

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()
