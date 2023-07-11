import os

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load normalized image from path.

    Args:
        path (str): Path to image.

    Returns:
        np.ndarray: Image as numpy array.
    """
    return np.array(Image.open(path), dtype=np.float32)[:, :, :3] / 255.0


def load_mask(path: str) -> np.ndarray:
    """Load mask from path.

    Args:
        path (str): Path to mask.

    Returns:
        np.ndarray: Mask as numpy array.
    """
    mask = np.array(Image.open(path))

    # convert to binary mask
    mask[mask > 0] = 1

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    return np.array(mask, dtype=np.uint8)


def load_weight(path: str) -> np.ndarray:
    """Load weight from path.

    Args:
        path (str): Path to weight.

    Returns:
        np.ndarray: Weight as numpy array.
    """
    return np.array(Image.open(path), dtype=np.uint8)


def save_image(img: np.ndarray, path: str) -> None:
    """Save image to path.

    Args:
        img (np.ndarray): Image to save.
        path (str): Path to save image to.
    """
    Image.fromarray(img).save(path)


def save_mask(mask: np.ndarray, path: str) -> None:
    """Save mask to path.

    Args:
        mask (np.ndarray): Mask to save.
        path (str): Path to save mask to.
    """
    mask = Image.fromarray(mask)
    mask = mask.convert("L")
    mask.save(path)


def list_dir(path: str) -> list:
    """List all files in a directory.

    Args:
        path (str): Path of the directory.

    Returns:
        list: List of paths to files in the directory.
    """
    fnames = os.listdir(path)
    return [os.path.join(path, fname) for fname in fnames]
