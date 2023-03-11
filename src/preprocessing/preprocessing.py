import argparse
import itertools
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.estimate_road_width import estimate_road_widths
from src.utils.io import list_dir, load_image, load_mask, save_image


def compute_weights(mask: np.ndarray, kernel_size: tuple = (21, 21)) -> np.ndarray:
    """Compute weights for each pixel in the mask based on the distance to the nearest edge.

    Args:
        mask (np.ndarray): Mask to compute weights for.
        kernel_size (tuple, optional): Kernel size of the Gaussian Blur. Defaults to (21, 21).

    Returns:
        np.ndarray: Weights for each pixel in the mask.
    """
    edges = cv2.Canny(mask, 0, 1)

    return cv2.GaussianBlur(edges, kernel_size, 0) + 1


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Smooth mask by filling small holes.

    Args:
        mask (np.ndarray): Mask to smooth.

    Returns:
        np.ndarray: Smoothed mask.
    """
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize image with a given scale.

    Args:
        image (np.ndarray): Image to resize.
        scale (float): Scale to resize the image with.

    Returns:
        np.ndarray: Resized image.
    """
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def is_white_image(img: np.ndarray, tol: float = 0.01) -> bool:
    """Check if image is white.

    Args:
        img (np.ndarray): Image to check.

    Returns:
        bool: True if image is white, False otherwise.
    """
    # make sure image is grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.sum(img == 255) / (img.shape[0] * img.shape[1]) > tol


def contains_road(mask: np.ndarray, tol: float = 0.005) -> bool:
    """Check if mask contains road.

    Args:
        mask (np.ndarray): Mask to check.

    Returns:
        bool: True if mask contains road, False otherwise.
    """
    # make sure mask is grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return np.sum(mask == 1) / (mask.shape[0] * mask.shape[1]) > tol


def extract_images(
    img_path: str, mask_path: str, size: tuple = (400, 400), scale: float = 1
) -> tuple:
    """Extract non-overlapping patches of size from an image and its mask.

    Args:
        img_path (str): Path to image.
        mask_path (str): Path to mask.
        size (tuple, optional): Size of the resulting patches. Defaults to (400, 400).
        scale (float, optional): Scale factor. Defaults to 1.

    Returns:
        tuple: Tuple containing:
            img_patches (list): List of image patches.
            mask_patches (list): List of mask patches.
            weights (list): List of weights for each patch.
    """
    image = load_image(img_path)
    mask = load_mask(mask_path)

    image = resize_image(image, scale)
    mask = resize_image(mask, scale)

    dataset = img_path.split("_")[1].split(".")[0]
    if dataset != "cil":
        mask = smooth_mask(mask)

    # extract non-overlapping patches with specified size
    img_patches = []
    mask_patches = []
    weights = []
    for i, j in itertools.product(
        range(0, image.shape[0], size[0]), range(0, image.shape[1], size[1])
    ):
        image_patch = image[i : i + size[0], j : j + size[1]]
        mask_patch = mask[i : i + size[0], j : j + size[1]]

        if image_patch.shape[:2] == size and mask_patch.shape[:2] == size:
            img_patches.append(image_patch)
            mask_patches.append(mask_patch)
            weights.append(compute_weights(mask_patch))

    return img_patches, mask_patches, weights


def main(args):
    images = list_dir(os.path.join(args.path, "images"))
    masks = list_dir(os.path.join(args.path, "masks"))

    # # estimated scale factors are not useful due to difference in masking techniques / quality
    # (
    #     cil_road_width,
    #     epfl_road_width,
    #     roadtracer_road_width,
    #     deepglobe_road_width,
    #     mit_road_width,
    # ) = estimate_road_widths(masks)

    # epfl_scale = cil_road_width / epfl_road_width
    # roadtracer_scale = cil_road_width / roadtracer_road_width
    # deepglobe_scale = cil_road_width / deepglobe_road_width
    # mit_scale = cil_road_width / mit_road_width

    # logging.info(f"Estimated scale for EPFL:       {epfl_scale}")
    # logging.info(f"Estimated scale for RoadTracer: {roadtracer_scale}")
    # logging.info(f"Estimated scale for DeepGlobe:  {deepglobe_scale}")
    # logging.info(f"Estimated scale for MIT:        {mit_scale}")

    # split images into datasets
    cil_images = sorted([image for image in images if "cil" in image])
    epfl_images = sorted([image for image in images if "epfl" in image])
    roadtracer_images = sorted([image for image in images if "roadtracer" in image])
    deepglobe_images = sorted([image for image in images if "dg" in image])
    mit_images = sorted([image for image in images if "mit" in image])

    # split masks into datasets
    cil_masks = sorted([mask for mask in masks if "cil" in mask])
    epfl_masks = sorted([mask for mask in masks if "epfl" in mask])
    roadtracer_masks = sorted([mask for mask in masks if "roadtracer" in mask])
    deepglobe_masks = sorted([mask for mask in masks if "dg" in mask])
    mit_masks = sorted([mask for mask in masks if "mit" in mask])

    # setup output directories
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if not os.path.exists(os.path.join(args.output, "images")):
        os.makedirs(os.path.join(args.output, "images"))

    if not os.path.exists(os.path.join(args.output, "masks")):
        os.makedirs(os.path.join(args.output, "masks"))

    if not os.path.exists(os.path.join(args.output, "weights")):
        os.makedirs(os.path.join(args.output, "weights"))

    extract_patches(cil_images, cil_masks, "cil", 1, args)
    extract_patches(epfl_images, epfl_masks, "epfl", 1, args)
    extract_patches(roadtracer_images, roadtracer_masks, "roadtracer", 1, args)
    extract_patches(deepglobe_images, deepglobe_masks, "deepglobe", 1.5, args)
    extract_patches(mit_images, mit_masks, "mit", 2, args)


def extract_patches(
    img_paths: list, mask_paths: list, dataset: str, scale: float, args: argparse.Namespace
):
    """Extract patches from images and masks and store them in the output directory.

    Args:
        img_paths (list): Path to images.
        mask_paths (list): Path to masks.
        dataset (str): Dataset name.
        scale (float): Scale to resize the images with.
        args (argparse.Namespace): Arguments.
    """
    img_id = 0
    n_ignored_images = 0
    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
        img_patches, mask_patches, weight_patches = extract_images(img_path, mask_path, scale=scale)
        for img_patch, mask_patch, weight_patch in zip(img_patches, mask_patches, weight_patches):
            if not is_white_image(img_patch) and contains_road(mask_patch):
                save_image(
                    img_patch, os.path.join(args.output, "images", f"{img_id:09d}_{dataset}.jpg")
                )
                save_image(
                    mask_patch * 255,
                    os.path.join(args.output, "masks", f"{img_id:09d}_{dataset}.png"),
                )
                save_image(
                    weight_patch,
                    os.path.join(args.output, "weights", f"{img_id:09d}_{dataset}.png"),
                )
                img_id += 1
            else:
                n_ignored_images += 1

    logging.info(f"Total number of images in {dataset}: {img_id}")
    logging.info(
        f"Number of ignored images in {dataset}: {n_ignored_images} "
        + f"({n_ignored_images / img_id * 100:.2f}%))"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="data.nosync/raw/all",
        help="Path to raw data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data.nosync/processed",
        help="Path to processed data.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
    )

    main(args)
