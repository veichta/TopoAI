import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.base_dataset import get_splits
from src.losses import Criterion
from src.metrics import Metrics
from src.models.base_unet import BaseUNet
from src.utils.utils import cleanup, get_args, setup


def main():
    """Main function."""
    args = get_args()
    setup(args)

    model = BaseUNet()
    model.to(args.device)
    logging.info(f"Number of trainable parameters: {model.n_trainable_params / 1e6:.2f} M")

    train_dl, val_dl = get_splits(args.datasets, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    criterion = Criterion(args)
    metrics = Metrics(criterion)

    if args.eval:
        if not args.model_path:
            raise ValueError("Please specify a model path to evaluate.")

        model.load_state_dict(torch.load(args.model_path))
        eval(args, model, val_dl, 0, metrics)

        val_dl.dataset.plot_predictions(model, filename=os.path.join(args.log_dir, "eval.png"))
        metrics.save_metrics(os.path.join(args.log_dir, "metrics.json"))
        return

    for epoch in range(args.epochs):
        train_one_epoch(args, model, train_dl, optimizer, criterion, epoch, metrics)
        eval(args, model, val_dl, epoch, metrics)
        scheduler.step(metrics.val_loss[-1])

        # log metrics
        val_dl.dataset.plot_predictions(model, filename=os.path.join(args.log_dir, "eval.png"))
        metrics.plot_metrics(os.path.join(args.log_dir, "metrics.png"))
        metrics.save_metrics(os.path.join(args.log_dir, "metrics.json"))

        # save best model
        if metrics.val_loss[-1] == np.min(metrics.val_loss):
            torch.save(model.state_dict(), os.path.join(args.log_dir, "best_model.pt"))

    best_epoch = np.argmax(metrics.val_acc)
    logging.info(f"Best epoch: {best_epoch + 1}")
    metrics.print_metrics(best_epoch, "eval")

    # load best model
    model.load_state_dict(torch.load(os.path.join(args.log_dir, "best_model.pt")))
    train_dl.dataset.plot_predictions(model, filename=os.path.join(args.log_dir, "best_train.png"))
    val_dl.dataset.plot_predictions(model, filename=os.path.join(args.log_dir, "best_eval.png"))

    cleanup(args)


def eval(args, model, val_dl, epoch, metrics: Metrics):
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
    metrics.end_epoch(epoch=epoch, mode="train")


if __name__ == "__main__":
    main()
