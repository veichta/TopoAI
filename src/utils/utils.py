import argparse
import logging

import numpy as np
import torch

import wandb


def get_args() -> argparse.Namespace:
    """Get arguments from command line.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument(
        "--data_path",
        type=str,
        default="data.nosync/preprocessed",
        help="Path to data",
    )

    # TRAINING

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
