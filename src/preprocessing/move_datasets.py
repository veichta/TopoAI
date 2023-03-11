import logging
import os
from enum import Enum

from PIL import Image
from tqdm import tqdm

TARGET_DIR = "data.nosync/raw/all"


class SplitsMIT(Enum):
    train = "train"
    val = "val"
    test = "test"


def copy_file(source: str, target: str):
    """Copy file from source to target.

    Args:
        source (str): Source path.
        target (str): Target path.
    """
    image = Image.open(source)
    if target.endswith(".jpg") and source.endswith(".png"):
        image = image.convert("RGB")

    if target.endswith(".png") and source.endswith(".jpg"):
        image = image.convert("L")

    if target.endswith(".tiff"):
        image = image.convert("L")

    image.save(target)


def prepare_cil(dataset_size: int) -> int:
    """Move CIL dataset to target directory.

    Args:
        dataset_size (int): Number of images currently in dataset.

    Returns:
        int: Number of images.
    """
    path = "data.nosync/raw/cil"
    logging.info("Preparing CIL dataset...")

    images = os.listdir(os.path.join(path, "training", "images"))
    masks = os.listdir(os.path.join(path, "training", "groundtruth"))

    for idx, (image, mask) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        fname = f"{idx + dataset_size:07d}_cil"
        source = os.path.join(path, "training", "images", image)
        target = os.path.join(TARGET_DIR, "images", f"{fname}.jpg")
        copy_file(source, target)

        source = os.path.join(path, "training", "groundtruth", mask)
        target = os.path.join(TARGET_DIR, "masks", f"{fname}.png")
        copy_file(source, target)

    return len(images)


def prepare_deepglobe(dataset_size: int) -> int:
    """Move DeepGlobe dataset to target directory.

    Args:
        dataset_size (int): Number of images currently in dataset.
        split (Split): Split of dataset.

    Returns:
        int: Number of images.
    """
    path = "data.nosync/raw/deepglobe"
    logging.info("Preparing DeepGlobe dataset...")
    files = os.listdir(os.path.join(path, "train"))

    images = [file for file in files if file.endswith(".jpg")]
    masks = [file for file in files if file.endswith(".png")]

    # sort images and masks
    images = sorted(images)
    masks = sorted(masks)

    for idx, (image, mask) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        assert image.split("_")[0] == mask.split("_")[0], "Image and mask do not match."

        fname = f"{idx + dataset_size:07d}_dg"
        source = os.path.join(path, "train", image)
        target = os.path.join(TARGET_DIR, "images", f"{fname}.jpg")
        copy_file(source, target)

        source = os.path.join(path, "train", mask)
        target = os.path.join(TARGET_DIR, "masks", f"{fname}.png")
        copy_file(source, target)

    return len(images)


def prepare_epfl(dataset_size: int) -> int:
    """Move EPFL dataset to target directory.

    Args:
        dataset_size (int): Number of images currently in dataset.

    Returns:
        int: Number of images.
    """
    path = "data.nosync/raw/epfl/training"
    logging.info("Preparing EPFL dataset...")

    images = os.listdir(os.path.join(path, "images"))
    masks = os.listdir(os.path.join(path, "groundtruth"))

    for idx, (image, mask) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        fname = f"{idx + dataset_size:07d}_epfl"
        source = os.path.join(path, "images", image)
        target = os.path.join(TARGET_DIR, "images", f"{fname}.jpg")
        copy_file(source, target)

        source = os.path.join(path, "groundtruth", mask)
        target = os.path.join(TARGET_DIR, "masks", f"{fname}.png")
        copy_file(source, target)

    return len(images)


def prepare_roadtracer(dataset_size: int) -> int:
    """Move Roadtracer dataset to target directory.

    Args:
        dataset_size (int): Number of images currently in dataset.

    Returns:
        int: Number of images.
    """
    path = "data.nosync/raw/roadtracer/training"
    logging.info("Preparing Roadtracer dataset...")

    images = os.listdir(os.path.join(path, "images"))
    masks = os.listdir(os.path.join(path, "groundtruth"))

    for idx, (image, mask) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        fname = f"{idx + dataset_size:07d}_roadtracer"
        source = os.path.join(path, "images", image)
        target = os.path.join(TARGET_DIR, "images", f"{fname}.jpg")
        copy_file(source, target)

        source = os.path.join(path, "groundtruth", mask)
        target = os.path.join(TARGET_DIR, "masks", f"{fname}.png")
        copy_file(source, target)

    return len(images)


def prepare_mit(dataset_size: int, split: SplitsMIT) -> int:
    """Move MIT dataset to target directory.

    Args:
        dataset_size (int): Number of images currently in dataset.
        split (SplitsMIT): Split of dataset.

    Returns:
        int: Number of images.
    """
    path = "data.nosync/raw/MIT/tiff"
    logging.info("Preparing MIT dataset...")

    images = os.listdir(os.path.join(path, split.value))
    masks = os.listdir(os.path.join(path, f"{split.value}_labels"))

    images = sorted(images)
    masks = sorted(masks)

    for idx, (image, mask) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        assert image.split(".")[0] == mask.split(".")[0], "Image and mask do not match."

        fname = f"{idx + dataset_size:07d}_mit"
        source = os.path.join(path, split.value, image)
        target = os.path.join(TARGET_DIR, "images", f"{fname}.jpg")
        copy_file(source, target)

        source = os.path.join(path, f"{split.value}_labels", mask)
        target = os.path.join(TARGET_DIR, "masks", f"{fname}.png")
        copy_file(source, target)

    return len(images)


def main():
    # create target dir
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        os.makedirs(os.path.join(TARGET_DIR, "images"))
        os.makedirs(os.path.join(TARGET_DIR, "masks"))

    dataset_size = 0

    # prepare cil
    cil_size = prepare_cil(dataset_size)
    dataset_size += cil_size
    logging.info(f"Prepared CIL dataset with {cil_size} images.")

    # prepare epfl
    epfl_size = prepare_epfl(dataset_size)
    dataset_size += epfl_size
    logging.info(f"Prepared EPFL dataset with {epfl_size} images.")

    # prepare roadtracer
    roadtracer_size = prepare_roadtracer(dataset_size)
    dataset_size += roadtracer_size
    logging.info(f"Prepared Roadtracer dataset with {roadtracer_size} images.")

    # prepare deepglobe
    deepglobe_size = prepare_deepglobe(dataset_size)
    dataset_size += deepglobe_size
    logging.info(f"Prepared DeepGlobe dataset with {deepglobe_size} images.")

    # prepare MIT
    mit_size = prepare_mit(dataset_size, SplitsMIT.train)
    mit_size += prepare_mit(dataset_size + mit_size, SplitsMIT.val)
    mit_size += prepare_mit(dataset_size + mit_size, SplitsMIT.test)
    dataset_size += mit_size
    logging.info(f"Prepared MIT dataset with {mit_size} images.")

    logging.info(f"Prepared dataset with {dataset_size} images.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
    )

    main()
