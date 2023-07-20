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
        default=8,
        help="Number of workers for dataloader",
    )

    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory of dataloader",
    )

    # MODEL
    parser.add_argument(
        "--model",
        type=str,
        choices=[e.value for e in ModelsEnum],
        default=ModelsEnum.UNET.value,
        help="Model to use",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Depth of UNet",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=5,
        help="Width of UNet i.e. number of channels in first layer (2^width)",
    )
    parser.add_argument(
        "--num_stacks",
        type=int,
        default=2,
        help="Number of stacks in SPIN",
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
        "--batches_per_epoch",
        type=int,
        help="Number of batches per training epoch",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for lr scheduler",
    )

    # LOSS
    parser.add_argument(
        "--miou_weight",
        type=float,
        default=1.0,
        help="Weight for miou loss",
    )
    parser.add_argument(
        "--bce_weight",
        type=float,
        default=1.0,
        help="Weight for cross entropy loss",
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=1.0,
        help="Weight for mse loss",
    )
    parser.add_argument(
        "--focal_weight",
        type=float,
        default=1.0,
        help="Weight for focal loss",
    )
    parser.add_argument(
        "--vec_weight",
        type=float,
        default=0.1,
        help="Weight for vector loss",
    )
    
    # LOSS WEIGHTS
    parser.add_argument(
        "--edge_weight",
        type=float,
        default=0,
        help="Weight for edges in loss (should be in [0,1)!)",
    )
    
    parser.add_argument(
        "--gaploss_weight",
        type=float,
        default=0,
        help="Weight for GAPLOSS in loss (should be in [0,1)!)",
    )

    # SOFT CL DICE 
    parser.add_argument(
        "--soft_skeleton_iter",
        type=float,
        default=5,
        help="number of iterations for soft skeletonization",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="convex combination between soft dice and soft cl dice, value in range [0, 0.5]",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.,
        help="for numerical stability in soft cl dice loss calculation",
    )

    parser.add_argument(
        "--cl_dice_weight",
        type=float,
        default=1,
        help="weight of cl_dice loss",
    )
    parser.add_argument(
        "--topo_weight",
        type=float,
        default=0.1,
        help="Weight for topological loss",
    )

    # LOGGING
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use wandb for logging",
    )

    parser.add_argument(
        "--wandb_dir",
        type=str,
        default=".",
        help="Wandb directory",
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
            entity="diffusion-roads",
            config=vars(get_args()),
            dir=args.wandb_dir,
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


    # check loss weights
    assert (args.edge_weight <= 1)
    assert (args.gaploss_weight <= 1)
    assert (args.edge_weight + args.gaploss_weight <= 1)

def cleanup(args: argparse.Namespace):
    """Clean up the environment."""
    if args.wandb:
        wandb.finish()
