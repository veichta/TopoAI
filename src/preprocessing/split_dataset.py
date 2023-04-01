# This script creates a train and validation set from the original dataset

import argparse
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm

from src.utils.enums import DatasetEnum


def main():
    # Create train and validation directories
    os.makedirs(os.path.join(args.dataset, "training", "images"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "training", "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "training", "weights"), exist_ok=True)

    os.makedirs(os.path.join(args.dataset, "validation", "images"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "validation", "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "validation", "weights"), exist_ok=True)

    # get all file names
    image_names = os.listdir(os.path.join(args.dataset, "images"))
    mask_names = os.listdir(os.path.join(args.dataset, "masks"))
    weight_names = os.listdir(os.path.join(args.dataset, "weights"))

    image_names = sorted(image_names)
    mask_names = sorted(mask_names)
    weight_names = sorted(weight_names)

    for dataset in DatasetEnum:
        if dataset == DatasetEnum.ALL:
            continue

        # get all file names for the current dataset
        image_names_dataset = [name for name in image_names if dataset.value in name]
        mask_names_dataset = [name for name in mask_names if dataset.value in name]
        weight_names_dataset = [name for name in weight_names if dataset.value in name]

        idx_train = np.random.choice(
            len(image_names_dataset),
            int(len(image_names_dataset) * args.train_split),
            replace=False,
        )

        # copy images to train and validation directories
        print(f"Copying {dataset.value} images to train and validation directories")
        for idx, image_name in tqdm(enumerate(image_names_dataset), total=len(image_names_dataset)):
            if idx in idx_train:
                shutil.copy(
                    os.path.join(args.dataset, "images", image_name),
                    os.path.join(args.dataset, "training", "images", image_name),
                )
            else:
                shutil.copy(
                    os.path.join(args.dataset, "images", image_name),
                    os.path.join(args.dataset, "validation", "images", image_name),
                )

        # copy masks to train and validation directories
        print(f"Copying {dataset.value} masks to train and validation directories")
        for idx, mask_name in tqdm(enumerate(mask_names_dataset), total=len(mask_names_dataset)):
            if idx in idx_train:
                shutil.copy(
                    os.path.join(args.dataset, "masks", mask_name),
                    os.path.join(args.dataset, "training", "masks", mask_name),
                )
            else:
                shutil.copy(
                    os.path.join(args.dataset, "masks", mask_name),
                    os.path.join(args.dataset, "validation", "masks", mask_name),
                )

        # copy weights to train and validation directories
        print(f"Copying {dataset.value} weights to train and validation directories")
        for idx, weight_name in tqdm(
            enumerate(weight_names_dataset), total=len(weight_names_dataset)
        ):
            if idx in idx_train:
                shutil.copy(
                    os.path.join(args.dataset, "weights", weight_name),
                    os.path.join(args.dataset, "training", "weights", weight_name),
                )
            else:
                shutil.copy(
                    os.path.join(args.dataset, "weights", weight_name),
                    os.path.join(args.dataset, "validation", "weights", weight_name),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
    parser.add_argument("--train_split", type=float, default=0.8, help="percentage of train split")
    args = parser.parse_args()

    # fix random seed for reproducibility
    np.random.seed(42)

    main()
