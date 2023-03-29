import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.datasets.base_dataset import get_splits
from src.losses import Criterion
from src.metrics import Metrics
from src.models.base_unet import BaseUNet
from src.utils.utils import cleanup, get_args, setup
from src.utils.visualizations import plot_batch_predictions


def main():
    """Main function."""
    args = get_args()
    setup(args)

    # TODO: Add your code here

    model = BaseUNet()
    model.to(args.device)
    logging.info(f"Number of trainable parameters: {model.n_trainable_params / 1e6:.2f} M")

    train_dl, val_dl = get_splits(args.dataset, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = Criterion(args)
    metrics = Metrics(criterion)

    for epoch in range(args.epochs):
        train_one_epoch(args, model, train_dl, optimizer, criterion, epoch, metrics)
        eval(args, model, val_dl, criterion, epoch, metrics)
        # exit()

    metrics.plot_metrics("plots/metrics.png")

    cleanup(args)


def eval(args, model, val_dl, criterion, epoch, metrics: Metrics):
    logging.info(f"Eval epoch {epoch + 1}/{args.epochs}")
    pbar = tqdm(val_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
    model.eval()
    metrics.start_epoch()
    with torch.no_grad():
        show = True
        for img, mask, weight in val_dl:
            img = img.to(args.device)
            mask = mask.to(args.device)
            weight = weight.to(args.device)

            out = model(img)
            metrics.update(out, mask, weight * args.edge_weight)

            if show:
                plot_batch_predictions(
                    images=img,
                    masks=mask,
                    predictions=out.sigmoid(),
                    weights=weight * args.edge_weight,
                    img_mean=val_dl.dataset.img_mean,
                    img_std=val_dl.dataset.img_std,
                    filename="plots/eval.png",
                )
                # pbar.close()
                # return
                show = False

            pbar.set_postfix(
                loss=np.mean(metrics.epoch_loss),
                iou=np.mean(metrics.epoch_iou),
                acc=np.mean(metrics.epoch_acc),
            )
            pbar.update()

    pbar.close()
    metrics.end_epoch(epoch=epoch, mode="eval")


def train_one_epoch(args, model, train_dl, optimizer, criterion, epoch, metrics: Metrics):
    logging.info(f"Epoch {epoch + 1}/{args.epochs}")
    pbar = tqdm(train_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
    model.train()
    metrics.start_epoch()
    for img, mask, weight in train_dl:
        img = img.to(args.device)
        mask = mask.to(args.device)
        weight = weight.to(args.device)

        out = model(img)
        loss = criterion(out, mask, weight * args.edge_weight)

        metrics.update(out, mask, weight * args.edge_weight)

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
    metrics.end_epoch(epoch=epoch, mode="train")


if __name__ == "__main__":
    main()
