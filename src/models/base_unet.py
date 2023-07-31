import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.losses import calculate_weights
from src.metrics import Metrics


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class BaseUNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        """UNet model.

        Args:
            chs (tuple, optional): Number of channels in each layer. Defaults to (3, 64, 128, 256,
            512, 1024).
        """
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.dec_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1),  # output is a single channel
        )  # 1x1 convolution for producing the output

        self.n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x).squeeze(1).sigmoid()  # reduce to 1 channel


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


def eval(
    model: BaseUNet,
    val_dl: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: Metrics,
    epoch: int,
    args: argparse.Namespace,
):
    """Evaluate the model on the validation set.

    Args:
        model (nn.Module): BaseUNet model.
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


def train_one_epoch(
    model: BaseUNet,
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    metrics: Metrics,
    epoch: int,
    args: argparse.Namespace,
):
    """Train the model for one epoch.

    Args:
        model (BaseUNet): BaseUNet model.
        train_dl (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        metrics (Metrics): Metrics object.
        epoch (int): Current epoch.
        args (argparse.Namespace): Arguments.
    """
    logging.info(f"Epoch {epoch + 1}/{args.epochs}")
    pbar = tqdm(train_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
    model.train()
    metrics.start_epoch()
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

    pbar.close()
    metrics.end_epoch(epoch=epoch, mode="train", log_wandb=args.wandb)
