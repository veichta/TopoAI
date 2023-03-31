import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.utils.io import load_image


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
