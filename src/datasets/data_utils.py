import argparse
import logging
from typing import List, Tuple

import albumentations as A
import numpy as np
import torch
from tqdm import tqdm

from src.utils.io import load_image


def get_augmentation(resolution: int = 400) -> A.Compose:
    """Get augmentation.

    Args:
        resolution (int): Image resolution.

    Returns:
        A.Compose: The augmentations.
    """
    transform = A.Compose(
        [
            A.PadIfNeeded(min_height=resolution, min_width=resolution, p=1.0),
            A.Resize(height=resolution, width=resolution),
        ]
    )

    return A.Compose(
        [
            transform,
            A.OneOf(
                [
                    A.RandomRotate90(p=1),
                    A.RandomResizedCrop(
                        resolution, resolution, p=1, scale=(0.5, 1.2), ratio=(0.85, 1.15)
                    ),
                    A.GridDistortion(p=1, distort_limit=0.5),
                    A.ElasticTransform(p=1, alpha=120, sigma=750 * 0.05, alpha_affine=120 * 0.03),
                    A.RandomShadow(
                        p=1, num_shadows_lower=1, num_shadows_upper=5, shadow_dimension=3
                    ),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=50,
                        max_width=50,
                        min_holes=5,
                        min_height=10,
                        min_width=10,
                        fill_value=0,
                        p=1,
                    ),
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.Transpose(p=1),
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                    A.RandomGamma(p=1, gamma_limit=(50, 500)),
                    A.ColorJitter(p=1),
                ],
                0.8,
            ),
        ]
    )


def get_dataloader(
    dataset: torch.utils.data.Dataset, args: argparse.Namespace, shuffle: bool = True
):
    """Return the dataloader for the dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to use.
        args (argparse.Namespace): Arguments.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the dataset.
    """
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers
    )


def calculate_channel_mean_std(image_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the mean and standard deviation of the channels of the images.

    Args:
        image_paths (List[str]): List of paths to images.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and standard deviation of the channels.
    """
    mean = np.zeros(3)
    std = np.zeros(3)

    logging.info("Calculating mean and std of the dataset...")
    # TODO: make this more performant
    for image_path in tqdm(image_paths):
        image = load_image(image_path)
        mean += image.mean(axis=(0, 1))
        std += image.std(axis=(0, 1))

    mean /= len(image_paths)
    std /= len(image_paths)

    logging.info(f"Mean: {mean}")
    logging.info(f"Std: {std}")

    return mean, std
