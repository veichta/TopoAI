import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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


class UNetPlus(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        
        # compute input channels for skip convolutional layers
        # can be understood when looking at the figure in the paper
        num_cols = len(enc_chs) - 1
        self.skip_in_chs = np.zeros((num_cols, num_cols))
        self.skip_out_chs = np.zeros((num_cols, num_cols))
        self.skip_in_chs[:,0] = enc_chs[:-1]
        self.skip_out_chs[:,0] = enc_chs[1:]
        # first go through columns and then rows
        for j in range(1, num_cols):
            for i in range(0, num_cols - j):
                self.skip_in_chs[i,j] = np.sum(self.skip_out_chs[i,:j]) + self.skip_out_chs[i+1,j-1] / 2
                self.skip_out_chs[i,j] = chs[1] * 2**i

        # skip convolutions
        self.skip_convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(int(self.skip_in_chs[i,j]), chs[1] * 2**i, 1)
                        for j in range(1, num_cols - 1 - i)
                    ]
                )
                for i in range(0, num_cols - 2)
            ]
        )

        # skip up sampling
        self.skip_upconvs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(int(self.skip_out_chs[i,j]), \
                                           int(self.skip_out_chs[i,j] / 2), 2, 2)
                        for j in range(num_cols - 1 - i)
                    ]
                )
                for i in range(1, num_cols-1)
            ]
        )
        self.dec_in_chs = [self.skip_in_chs[num_cols-i, i-1] for i in range(2, num_cols+1)]
        # deconvolution
        self.dec_blocks = nn.ModuleList(
            [Block(int(in_ch), out_ch) for in_ch, out_ch in zip(self.dec_in_chs, dec_chs[1:])]
        )  
        # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1),  # output is a single channel
        )  # 1x1 convolution for producing the output

        # decoder block up sampling
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  

        self.n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # encode
        enc_features = []
        for i, block in enumerate(self.enc_blocks[:-1]):
            x = block(x)  # pass through the block
            enc_features.append([x])  # save features for skip connections
            for j in range(1, i+1): # iterate columns of the pyramid
                input = enc_features[i - j][0] # get first feature map
                for k in range(1, j): # concat feature maps of current row
                    input = torch.cat([input, enc_features[i - j][k]], dim=1)
                up = self.skip_upconvs[i - j][j - 1](enc_features[i - j + 1][j - 1]) # upsample feature map of row below
                input = torch.cat([input, up], dim=1)
                assert input.shape[1] == self.skip_in_chs[i-j, j]
                enc_features[i - j].append(self.skip_convs[i - j][j-1](input))

            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for i, (block, upconv) in enumerate(zip(self.dec_blocks, self.upconvs)):
            x = upconv(x)  # increase resolution
            for k in range(i+1):
                x = torch.cat([x, enc_features[len(self.dec_blocks)-1-i][k]], dim=1) # concatenate skip features
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
    model: UNetPlus,
    val_dl: torch.utils.data.DataLoader,
    metrics: Metrics,
    epoch: int,
    args: argparse.Namespace,
):
    """Evaluate the model on the validation set.

    Args:
        model (nn.Module): BaseUNet model.
        val_dl (torch.utils.data.DataLoader): Validation data loader.
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
            weight = weight.to(args.device)

            out = model(img)
            metrics.update(out, mask, weight)

            pbar.set_postfix(
                loss=np.mean(metrics.epoch_loss),
                iou=np.mean(metrics.epoch_iou),
                acc=np.mean(metrics.epoch_acc),
            )
            pbar.update()

    pbar.close()
    metrics.end_epoch(epoch=epoch, mode="eval")


def train_one_epoch(
    model: UNetPlus,
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
    if args.batches_per_epoch is not None:
        pbar.total = args.batches_per_epoch
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
    model.train()
    metrics.start_epoch()
    batch_count = 0
    for img, mask, weight in train_dl:
        img = img.to(args.device)
        mask = mask.to(args.device)
        weight = weight.to(args.device)

        out = model(img)
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
    metrics.end_epoch(epoch=epoch, mode="train")

class UPlusLoss(nn.Module):

    # loss used in the paper
    # TODO: implement dice loss

    def __init__(self):
        super(UPlusLoss, self).__init__()

        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, outputs, masks):
        
        return self.bce_loss(outputs, masks)
