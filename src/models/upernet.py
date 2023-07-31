import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import nn
from tqdm import tqdm
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from src.losses import calculate_weights
from src.metrics import Metrics


class UperNet(nn.Module):
    def __init__(self, backbone: str = "upernet-t", freeze_backbone: bool = True) -> None:
        """UperNet model for semantic segmentation using ConvNext backbone.

        Args:
            backbone (str): Backbone to use. One of {"tiny", "base", "large"}.
            freeze_backbone (bool): Whether to freeze the backbone.
        """
        super().__init__()
        assert backbone in {"upernet-t", "upernet-b", "upernet-l"}

        size = {"upernet-t": "tiny", "upernet-b": "base", "upernet-l": "large"}[backbone]
        logging.info(f"Loading UperNet-{size} with ConvNext backbone")

        self.upernet = UperNetForSemanticSegmentation.from_pretrained(
            f"openmmlab/upernet-convnext-{size}"
        )

        self.upernet.decode_head.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
        )

        self.processor = AutoImageProcessor.from_pretrained(f"openmmlab/upernet-convnext-{size}")

        # freeze backbone
        if freeze_backbone:
            for param in self.upernet.backbone.parameters():
                param.requires_grad = False

        self.n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: torch.tensor) -> torch.tensor:
        # upscaled to 512x512
        rescale = batch.shape[-2:]
        batch = self.processor(images=batch, return_tensors="pt")
        batch["pixel_values"] = batch["pixel_values"].to(next(self.parameters()).device)

        # forward pass
        out = self.upernet(**batch)
        logits = out.logits

        # downscale to original size
        logits = nn.functional.interpolate(
            logits, size=rescale, mode="bilinear", align_corners=False
        )

        # squeeze to remove channel dimension
        logits = logits.squeeze(1)

        # sigmoid to get probabilities
        return logits.sigmoid()


def load_model(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    """Load model from checkpoint.

    Args:
        model (nn.Module): Model to load.
        args (argparse.Namespace): _description_

    Raises:
        ValueError: If no model path is specified.

    Returns:
        nn.Module: Loaded model.
    """
    if args.resume or args.eval:
        if not args.model_path:
            raise ValueError("Please specify a model path to resume from.")

        logging.info(f"Loading model checkpoint from {args.model_path}")

        model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
        model.to(args.device)

    return model


def train_one_epoch(
    model: UperNet,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    metrics: Metrics,
    epoch: int,
    args: argparse.Namespace,
):
    """Train the model for one epoch.

    Args:
        model (UperNet): Upernet model.
        train_dl (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        metrics (Metrics): Metrics object.
        epoch (int): Current epoch.
        args (argparse.Namespace): Arguments.
    """
    logging.info(f"Epoch {epoch + 1}/{args.epochs}")
    pbar = tqdm(train_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    if args.batches_per_epoch is not None:
        pbar.total = args.batches_per_epoch
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
    model.train()
    metrics.start_epoch()
    batch_count = 0
    for img, mask, weight in train_dl:
        img = img.to(args.device)
        mask = mask.to(args.device)

        out = model(img)

        weight = calculate_weights(out, weight, args)

        loss = criterion(out, mask, weight)

        metrics.update(out, mask, weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=np.mean(metrics.epoch_loss),
            iou=np.mean(metrics.epoch_iou),
            acc=np.mean(metrics.epoch_acc),
        )
        pbar.update()

        batch_count += 1

        if args.batches_per_epoch is not None and batch_count >= args.batches_per_epoch:
            break

    pbar.close()
    metrics.end_epoch(epoch=epoch, mode="train", log_wandb=args.wandb)


def eval(
    model: UperNet,
    val_dl: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: Metrics,
    epoch: int,
    args: argparse.Namespace,
):
    """Evaluate the model on the validation set.

    Args:
        model (UperNet): Upernet model.
        val_dl (torch.utils.data.DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        metrics (Metrics): Metrics object.
        epoch (int): Current epoch.
        args (argparse.Namespace): Arguments.
    """
    logging.info(f"Eval epoch {epoch + 1}/{args.epochs}")
    pbar = tqdm(val_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
    model.eval()
    metrics.start_epoch()
    with torch.no_grad():
        for img, mask, weight in val_dl:
            img = img.to(args.device)
            mask = mask.to(args.device)

            out = model(img)

            weight = calculate_weights(out, weight, args)

            metrics.update(out, mask, weight)

            pbar.set_postfix(
                loss=np.mean(metrics.epoch_loss),
                iou=np.mean(metrics.epoch_iou),
                acc=np.mean(metrics.epoch_acc),
            )
            pbar.update()

    pbar.close()
    metrics.end_epoch(epoch=epoch, mode="eval", log_wandb=args.wandb)
