import argparse
import logging
import os

import numpy as np
import torch
import torchvision

from src.datasets.data_utils import calculate_channel_mean_std, get_dataloader
from src.utils.enums import DatasetEnum
from src.utils.io import list_dir, load_image, load_mask, load_weight


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class."""

    def __init__(
        self,
        img_paths: str,
        mask_paths: str,
        weight_paths: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        args: argparse.Namespace,
        split: str = "train",
    ):
        """Initialize the dataset.

        Args:
            img_paths (str): Path to images.
            mask_paths (str): Path to masks.
            weight_paths (str): Path to weights.
            mean (torch.Tensor): Mean of the dataset.
            std (torch.Tensor): Standard deviation of the dataset.
            args (argparse.Namespace): Arguments.
            split (str, optional): Split of the dataset. Defaults to "train".
        """
        super(BaseDataset, self).__init__()
        self.args = args
        self.split = split

        self.images = img_paths
        self.masks = mask_paths
        self.weights = weight_paths

        self.img_mean = mean
        self.img_std = std
        self.img_norm = torchvision.transforms.Normalize(mean=mean, std=std)

        self.transforms = []
        # if self.split == "train":
        #     self.transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
        #     self.transforms.append(torchvision.transforms.RandomVerticalFlip(0.5))

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        image = load_image(self.images[index])
        mask = load_mask(self.masks[index])
        weight = load_weight(self.weights[index])

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        weight = torch.from_numpy(weight).float()

        image = image.permute(2, 0, 1)
        image = self.img_norm(image)

        for transform in self.transforms:
            image = transform(image)
            mask = transform(mask)

        logging.debug(f"Image shape: {image.shape}")
        logging.debug(f"Mask shape: {mask.shape}")
        logging.debug(f"Weight shape: {weight.shape}")

        return image, mask, weight


def get_splits(dataset: str, args: argparse.Namespace):
    """Return the splits of the dataset.

    Args:
        dataset (str): Dataset to use.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Train and validation datasets.
    """

    images = list_dir(os.path.join(args.data_path, "images"))
    masks = list_dir(os.path.join(args.data_path, "masks"))
    weights = list_dir(os.path.join(args.data_path, "weights"))

    if dataset != DatasetEnum.ALL.value:
        images = [image for image in images if dataset in image]
        masks = [mask for mask in masks if dataset in mask]
        weights = [weight for weight in weights if dataset in weight]

    images = sorted(images)
    masks = sorted(masks)
    weights = sorted(weights)

    order = np.random.permutation(len(images))

    images = np.array(images)[order]
    masks = np.array(masks)[order]
    weights = np.array(weights)[order]

    train_images = images[: int(0.8 * len(images))]
    train_masks = masks[: int(0.8 * len(masks))]
    train_weights = weights[: int(0.8 * len(weights))]

    val_images = images[int(0.8 * len(images)) :]
    val_masks = masks[int(0.8 * len(masks)) :]
    val_weights = weights[int(0.8 * len(weights)) :]

    train_mean, train_std = calculate_channel_mean_std(train_images)

    train_dataset = BaseDataset(
        img_paths=train_images,
        mask_paths=train_masks,
        weight_paths=train_weights,
        mean=train_mean,
        std=train_std,
        args=args,
        split="train",
    )
    val_dataset = BaseDataset(
        img_paths=val_images,
        mask_paths=val_masks,
        weight_paths=val_weights,
        mean=train_mean,
        std=train_std,
        args=args,
        split="val",
    )

    return get_dataloader(train_dataset, args), get_dataloader(val_dataset, args, shuffle=False)
