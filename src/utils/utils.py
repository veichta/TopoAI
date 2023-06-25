import argparse
import datetime
import json
import logging
import os
import time

import numpy as np
import torch

import wandb
from src.utils.enums import DatasetEnum, ModelsEnum


def get_args() -> argparse.Namespace:
    """Get arguments from command line.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser()

    # GENERAL
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model",
    )

    # DATA
    parser.add_argument(
        "--data_path",
        type=str,
        default="data.nosync/processed",
        help="Path to data",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=[e.value for e in DatasetEnum],
        default=[DatasetEnum.CIL.value],
        help="Dataset to use",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.json",
        help="Path to metadata.json.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for dataloader",
    )

    # MODEL
    parser.add_argument(
        "--model",
        type=str,
        choices=[e.value for e in ModelsEnum],
        default=ModelsEnum.UNET.value,
        help="Model to use",
    )

    # TRAINING
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )

    parser.add_argument(
        "--edge_weight",
        type=float,
        default=0.01,
        help="Weight for edges in loss",
    )

    # LOGGING
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use wandb for logging",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Log directory",
    )

    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Log to file",
    )

    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        help="Number of batches per training epoch",
    )

    return parser.parse_args()


def setup(args: argparse.Namespace):
    """Set up the environment for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # setup wandb
    if args.wandb:
        wandb.init(
            project="DiffusionRoads",
            config=vars(get_args()),
        )

    # log dir
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = f"{args.log_dir}/{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
        filename=os.path.join(args.log_dir, "log.txt") if args.log_to_file else None,
    )

    logging.info("Arguments:")
    for k, v in vars(args).items():
        logging.info(f"\t{k}: {v}")

    # save args
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def cleanup(args: argparse.Namespace):
    """Clean up the environment."""
    if args.wandb:
        wandb.finish()
