import argparse
import json
import logging
import os
from typing import List

import albumentations as A
import torch
import torchvision
from torch import nn

import src.utils.angles as affinity_utils
from src.datasets.data_utils import get_augmentation, get_dataloader
from src.utils.enums import DatasetEnum, ModelsEnum
from src.utils.io import list_dir, load_image, load_mask, load_weight
from src.utils.visualizations import plot_predictions


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class."""

    def __init__(
        self,
        img_paths: str,
        mask_paths: str,
        weight_paths: str,
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

        self.metadata = json.load(open(self.args.metadata, "r"))

        self.angle_theta = 10.0

        if self.split == "train":
            self.transforms = get_augmentation(resolution=400)
        else:
            self.transforms = A.Compose(
                [
                    A.PadIfNeeded(min_height=400, min_width=400, p=1.0),
                    A.Resize(height=400, width=400),
                ]
            )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    def getOrientationGT(self, keypoints, height, width):
        """Create Orientation Ground Truth

        Args:
            keypoints (list): List of keypoints.
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            vecmap_angle (torch.tensor): Orientation Ground Truth in the shape of (h, w).
        """
        vecmap, vecmap_angles = affinity_utils.getVectorMapsAngles(
            (height, width), keypoints, theta=self.angle_theta, bin_size=10
        )
        vecmap_angles = torch.from_numpy(vecmap_angles)

        return vecmap_angles

    def get_spin_out(self, image, mask, weight):
        labels = []
        weights = []
        vecmap_angles = []

        smoothness = [1, 2, 4]
        scale = [4, 2, 1]

        h, w, _ = image.shape
        for i, sc in enumerate(scale):
            new_h, new_w = int(h / sc), int(w / sc)

            # Resize mask and weight
            transform = A.Resize(height=new_h, width=new_w)
            transformed = transform(image=image, masks=[mask, weight])
            mask_scaled = transformed["masks"][0]
            weight_scaled = transformed["masks"][1]

            # Create Orientation Ground Truth
            keypoints = affinity_utils.getKeypoints(
                mask_scaled, is_gaussian=False, smooth_dist=smoothness[i]
            )
            vecmap_angle = self.getOrientationGT(
                keypoints,
                height=new_h,
                width=new_w,
            )

            labels.append(torch.from_numpy(mask_scaled).float())
            weights.append(torch.from_numpy(weight_scaled).float())
            vecmap_angles.append(vecmap_angle)

        return labels, weights, vecmap_angles

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        image = load_image(self.images[index])
        mask = load_mask(self.masks[index])
        weight = load_weight(self.weights[index])

        augmented = self.transforms(image=image, masks=[mask, weight])
        image = augmented["image"]
        mask = augmented["masks"][0]
        weight = augmented["masks"][1]

        assert image.max() <= 1.0, f"Image max: {image.max()}"
        assert image.min() >= 0.0, f"Image min: {image.min()}"
        assert mask.max() <= 1.0, f"Mask max: {mask.max()}"
        assert mask.min() >= 0.0, f"Mask min: {mask.min()}"

        if self.args.model == ModelsEnum.SPIN.value:
            mask, weight, vecmaps = self.get_spin_out(image, mask, weight)

            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)
            image = self.normalize_image(image, self.images[index])
            return image, mask, weight, vecmaps

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        weight = torch.from_numpy(weight).float()

        image = image.permute(2, 0, 1)
        image = self.normalize_image(image, self.images[index])

        logging.debug(f"Image shape: {image.shape}")
        logging.debug(f"Mask shape: {mask.shape}")
        logging.debug(f"Weight shape: {weight.shape}")

        return image, mask, weight

    def normalize_image(self, image: torch.tensor, img_path: str) -> torch.tensor:
        """Normalize the image.

        Args:
            image (torch.tensor): Image to normalize.
            img_path (str): Path to the image.

        Returns:
            torch.tensor: Normalized image.
        """
        dataset = img_path.split("_")[-1].split(".")[0]
        mean = self.metadata[dataset]["img_mean"]
        std = self.metadata[dataset]["img_std"]
        return torchvision.transforms.Normalize(mean=mean, std=std)(image)

    def denormalize_image(self, image: torch.tensor, img_path: str) -> torch.tensor:
        """Denormalize the image.

        Args:
            image (torch.tensor): Image to denormalize.
            img_path (str): Path to the image.

        Returns:
            torch.tensor: Denormalized image.
        """
        dataset = img_path.split("_")[-1].split(".")[0]
        mean = self.metadata[dataset]["img_mean"]
        std = self.metadata[dataset]["img_std"]

        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)

        return ((image * std) + mean).permute(1, 2, 0)

    def plot_predictions(self, model: nn.Module, n_samples: int = 5, filename: str = None, args: argparse.Namespace = None) -> None:
        model.eval()

        batch = [self[i] for i in range(n_samples)]
        if len(batch[0]) > 3:  # SPIN
            batch = [(b[0], b[1][-1], b[2][-1]) for b in batch]

        images, masks, weights = zip(*batch)

        images = torch.stack(images)
        masks = torch.stack(masks)
        weights = torch.stack(weights)

        with torch.no_grad():
            predictions = model(images.to(self.args.device))

        images = torch.stack(
            [self.denormalize_image(image, self.images[i]) for i, image in enumerate(images)]
        )

        if type(predictions) == tuple:  # SPIN
            predictions = predictions[0][-1]

        plot_predictions(
            images=images,
            masks=masks,
            predictions=predictions,
            weights=weights,
            filename=filename,
            log_wandb=self.args.wandb,
            args=args,
        )


def get_splits(datasets: List[str], args: argparse.Namespace):
    """Return the splits of the dataset.

    Args:
        datasets (List[str]): Datasets to use.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Train and validation datasets.
    """

    # images = list_dir(os.path.join(args.data_path, "images"))
    # masks = list_dir(os.path.join(args.data_path, "masks"))
    # weights = list_dir(os.path.join(args.data_path, "weights"))

    train_images = list_dir(os.path.join(args.data_path, "training", "images"))
    train_masks = list_dir(os.path.join(args.data_path, "training", "masks"))
    train_weights = list_dir(os.path.join(args.data_path, "training", "weights"))

    val_images = list_dir(os.path.join(args.data_path, "validation", "images"))
    val_masks = list_dir(os.path.join(args.data_path, "validation", "masks"))
    val_weights = list_dir(os.path.join(args.data_path, "validation", "weights"))

    if DatasetEnum.ALL.value not in datasets:
        train_images = [
            image
            for image in train_images
            if any(dataset in image.split("/")[-1] for dataset in datasets)
        ]
        train_masks = [
            mask
            for mask in train_masks
            if any(dataset in mask.split("/")[-1] for dataset in datasets)
        ]
        train_weights = [
            weight
            for weight in train_weights
            if any(dataset in weight.split("/")[-1] for dataset in datasets)
        ]

        val_images = [
            image
            for image in val_images
            if any(dataset in image.split("/")[-1] for dataset in datasets)
        ]
        val_masks = [
            mask
            for mask in val_masks
            if any(dataset in mask.split("/")[-1] for dataset in datasets)
        ]
        val_weights = [
            weight
            for weight in val_weights
            if any(dataset in weight.split("/")[-1] for dataset in datasets)
        ]

    # val images only from cil dataset
    val_images = [image for image in val_images if "cil" in image.split("/")[-1]]
    val_masks = [mask for mask in val_masks if "cil" in mask.split("/")[-1]]
    val_weights = [weight for weight in val_weights if "cil" in weight.split("/")[-1]]

    logging.info(f"Train images: {len(train_images)}")
    logging.info(f"Valid images: {len(val_images)}")

    train_dataset = BaseDataset(
        img_paths=sorted(train_images),
        mask_paths=sorted(train_masks),
        weight_paths=sorted(train_weights),
        args=args,
        split="train",
    )
    val_dataset = BaseDataset(
        img_paths=sorted(val_images),
        mask_paths=sorted(val_masks),
        weight_paths=sorted(val_weights),
        args=args,
        split="val",
    )

    return get_dataloader(train_dataset, args), get_dataloader(val_dataset, args, shuffle=False)
