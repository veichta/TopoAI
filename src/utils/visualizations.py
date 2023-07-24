import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb

from src.losses import GapLoss_weights

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

    img = (1 - alpha) * img + alpha * mask[:, :, None]
    return img * 255


def plot_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor = None,
    filename: str = None,
    log_wandb: bool = False,
    plot_Gaploss: bool = False,
) -> None:
    num_images = images.shape[0]
    fig, ax = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    if plot_Gaploss:
        C, W, skeletons = GapLoss_weights(predictions, to_plot=True)
        
    for i in range(num_images):
        img = images[i].detach().cpu().numpy()
        prediction = predictions[i].detach().cpu().numpy()
        overlay = overlay_image_mask(img, prediction)
        overlay = overlay.astype(np.uint8)
        ax[i, 0].imshow(overlay)

        # make prediction white and draw the weights in red ontop
        pred_img = np.zeros_like(img)
        if weights is not None:
            pred_img = np.stack([prediction] * 3, axis=-1) * 255
            contour = weights[i].detach().cpu().numpy()
            contour = (contour - contour.min()) / (contour.max() - contour.min())
            pred_img[contour > 0.3] = [255, 0, 0]
        else:
            pred_img = (prediction > 0.5) * 255

        pred_img = pred_img.astype(np.uint8)
        ax[i, 1].imshow(pred_img)
        if weights is not None and not plot_Gaploss:
            ax[i, 2].imshow(weights[i].detach().cpu().numpy() + masks[i].detach().cpu().numpy())
        elif plot_Gaploss:
            ax[i, 2].imshow(W[i])

    ax[0, 0].set_title("Image + Mask")
    ax[0, 1].set_title("Mask")
    ax[0, 2].set_title("Weight")

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    if log_wandb:
        wandb.log({"predictions": wandb.Image(plt)})

    plt.close()
