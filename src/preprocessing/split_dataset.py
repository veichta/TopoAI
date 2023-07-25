# This script creates a train and validation set from the original dataset

import argparse
import os
import shutil

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

    os.makedirs(os.path.join(args.dataset, "testing", "images"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "testing", "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset, "testing", "weights"), exist_ok=True)

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
        image_names_dataset = np.array([name for name in image_names if dataset.value in name])
        mask_names_dataset = np.array([name for name in mask_names if dataset.value in name])
        weight_names_dataset = np.array([name for name in weight_names if dataset.value in name])

        # Randomly shuffle the indices of the dataset
        shuffled_indices = np.random.permutation(len(image_names_dataset))

        # Calculate the sizes of the train, validation, and test sets
        num_train = int(len(image_names_dataset) * args.train_split)
        num_val = (len(image_names_dataset) - num_train) // 2
        num_test = len(image_names_dataset) - num_train - num_val

        # Divide the shuffled indices into train, validation, and test sets
        idx_train = shuffled_indices[:num_train]
        idx_val = shuffled_indices[num_train : num_train + num_val]
        idx_test = shuffled_indices[num_train + num_val :]

        # Get file names
        train_img_fn = image_names_dataset[idx_train]
        val_img_fn = image_names_dataset[idx_val]
        test_img_fn = image_names_dataset[idx_test]

        train_mask_fn = mask_names_dataset[idx_train]
        val_mask_fn = mask_names_dataset[idx_val]
        test_mask_fn = mask_names_dataset[idx_test]

        train_weight_fn = weight_names_dataset[idx_train]
        val_weight_fn = weight_names_dataset[idx_val]
        test_weight_fn = weight_names_dataset[idx_test]

        # copy images
        print(f"Copying {dataset.value} images to train and validation directories")
        for img_fn in tqdm(image_names_dataset, ncols=80):
            if img_fn in train_img_fn:
                shutil.copy(
                    os.path.join(args.dataset, "images", img_fn),
                    os.path.join(args.dataset, "training", "images", img_fn),
                )
            elif img_fn in val_img_fn:
                shutil.copy(
                    os.path.join(args.dataset, "images", img_fn),
                    os.path.join(args.dataset, "validation", "images", img_fn),
                )
            elif img_fn in test_img_fn:
                shutil.copy(
                    os.path.join(args.dataset, "images", img_fn),
                    os.path.join(args.dataset, "testing", "images", img_fn),
                )
            else:
                raise ValueError(f"Image {img_fn} not found in any split")

        # copy masks
        print(f"Copying {dataset.value} masks to train and validation directories")
        for mask_fn in tqdm(mask_names_dataset, ncols=80):
            if mask_fn in train_mask_fn:
                shutil.copy(
                    os.path.join(args.dataset, "masks", mask_fn),
                    os.path.join(args.dataset, "training", "masks", mask_fn),
                )
            elif mask_fn in val_mask_fn:
                shutil.copy(
                    os.path.join(args.dataset, "masks", mask_fn),
                    os.path.join(args.dataset, "validation", "masks", mask_fn),
                )
            elif mask_fn in test_mask_fn:
                shutil.copy(
                    os.path.join(args.dataset, "masks", mask_fn),
                    os.path.join(args.dataset, "testing", "masks", mask_fn),
                )
            else:
                print(f"Mask {mask_fn} not found in any split")

        # copy weights
        print(f"Copying {dataset.value} weights to train and validation directories")
        for weight_fn in tqdm(weight_names_dataset, ncols=80):
            if weight_fn in train_weight_fn:
                shutil.copy(
                    os.path.join(args.dataset, "weights", weight_fn),
                    os.path.join(args.dataset, "training", "weights", weight_fn),
                )
            elif weight_fn in val_weight_fn:
                shutil.copy(
                    os.path.join(args.dataset, "weights", weight_fn),
                    os.path.join(args.dataset, "validation", "weights", weight_fn),
                )
            elif weight_fn in test_weight_fn:
                shutil.copy(
                    os.path.join(args.dataset, "weights", weight_fn),
                    os.path.join(args.dataset, "testing", "weights", weight_fn),
                )
            else:
                print(f"Weight {weight_fn} not found in any split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
    parser.add_argument("--train_split", type=float, default=0.8, help="percentage of train split")
    args = parser.parse_args()

    # fix random seed for reproducibility
    np.random.seed(42)

    main()
