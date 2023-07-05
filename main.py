import logging
import os

import numpy as np
import torch

from src.datasets.base_dataset import get_splits
from src.losses import Criterion
from src.metrics import Metrics
from src.utils.enums import ModelsEnum
from src.utils.utils import cleanup, get_args, setup


def main():
    """Main function."""
    args = get_args()
    setup(args)

    if args.model == ModelsEnum.UNET.value:
        from src.models.base_unet import BaseUNet, eval, load_model, train_one_epoch

        model = BaseUNet()
        model = load_model(model=model, args=args)
        model.to(args.device)

    elif args.model == ModelsEnum.UNETPP.value:
        from src.models.unet_pp import UNetPlus, eval, load_model, train_one_epoch

        model = UNetPlus()
        model = load_model(model=model, args=args)
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
        eval(
            model=model,
            val_dl=val_dl,
            metrics=metrics,
            epoch=0,
            args=args,
        )

        val_dl.dataset.plot_predictions(model, filename=os.path.join(args.log_dir, "eval.png"))
        metrics.save_metrics(os.path.join(args.log_dir, "metrics.json"))
        return

    for epoch in range(args.epochs):
        train_one_epoch(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            epoch=epoch,
            args=args,
        )
        eval(
            model=model,
            val_dl=val_dl,
            metrics=metrics,
            epoch=epoch,
            args=args,
        )
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


if __name__ == "__main__":
    main()
