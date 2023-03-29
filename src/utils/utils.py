import argparse
import logging

import numpy as np
import torch

import wandb
from src.utils.enums import DatasetEnum


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

    # DATA
    parser.add_argument(
        "--data_path",
        type=str,
        default="data.nosync/processed",
        help="Path to data",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[e.value for e in DatasetEnum],
        default="cil",
        help="Dataset to use",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for dataloader",
    )

    # TRAINING
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        type=bool,
        default=False,
        help="Use wandb for logging",
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

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
    )

    logging.info("Arguments:")
    for k, v in vars(args).items():
        logging.info(f"\t{k}: {v}")


def cleanup(args: argparse.Namespace):
    """Clean up the environment."""
    if args.wandb:
        wandb.finish()
