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

    # get model and criterion
    chs = [3] + [2 ** (i + args.width) for i in range(args.depth)]
    if args.model == ModelsEnum.UNET.value:
        from src.models.base_unet import BaseUNet, eval, load_model, train_one_epoch

        model = BaseUNet(chs)
        model = load_model(model=model, args=args)
        model.to(args.device)
        criterion = Criterion(args)

    elif args.model == ModelsEnum.UNETPP.value:
        from src.models.unet_pp import UNetPlus, eval, load_model, train_one_epoch

        model = UNetPlus(chs)
        model = load_model(model=model, args=args)
        model.to(args.device)
        criterion = Criterion(args)

    elif args.model == ModelsEnum.SPIN.value:
        from src.models.hr_spin import HRSPIN, SPINCriterion, eval, load_model, train_one_epoch

        loss_fn = Criterion(args).to(args.device)
        criterion = SPINCriterion(loss_fn=loss_fn, args=args).to(args.device)

        model = HRSPIN(num_stacks=args.num_stacks)
        model = load_model(model=model, args=args)
        model.to(args.device)

    logging.info(f"Number of trainable parameters: {model.n_trainable_params / 1e6:.2f} M")

    train_dl, val_dl, test_dl = get_splits(args.datasets, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=args.patience, verbose=True
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, args.epochs // 5, args.lr / 10
    # )

    metrics = Metrics(Criterion(args).to(args.device))

    if args.eval:
        eval(
            model=model,
            val_dl=val_dl,
            metrics=metrics,
            epoch=0,
            args=args,
        )

        val_dl.dataset.plot_predictions(
            model,
            filename=os.path.join(args.log_dir, "eval.png"),
            plot_Gaploss=args.gaploss_weight > 0,
        )
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
            criterion=criterion,
            metrics=metrics,
            epoch=epoch,
            args=args,
        )
        scheduler.step(metrics.val_loss[-1])
        # scheduler.step()

        # log metrics
        if epoch % 5 == 0:
            val_dl.dataset.plot_predictions(
                model, filename=os.path.join(args.log_dir, "eval.png"), args=args
            )

        metrics.plot_metrics(os.path.join(args.log_dir, "metrics.png"))
        metrics.save_metrics(os.path.join(args.log_dir, "metrics.json"))

        # save best model
        if metrics.val_loss[-1] == np.min(metrics.val_acc):
            torch.save(model.state_dict(), os.path.join(args.log_dir, "best_model.pt"))

    best_epoch = np.argmax(metrics.val_acc)
    logging.info(f"Best epoch: {best_epoch + 1}")
    metrics.print_metrics(best_epoch, "eval")

    if args.wandb:
        metrics.log_to_wandb(best_epoch, "eval")

    # store last model
    torch.save(model.state_dict(), os.path.join(args.log_dir, "best_model.pt"))

    # Eval last model on test set
    logging.info("TEST RESULTS (LAST MODEL)")
    eval(
        model=model,
        val_dl=test_dl,
        criterion=criterion,
        metrics=metrics,
        epoch=args.epochs,
        args=args,
    )
    if args.wandb:
        metrics.log_to_wandb(args.epochs, "test")

    # load best model
    model.load_state_dict(torch.load(os.path.join(args.log_dir, "best_model.pt")))
    train_dl.dataset.plot_predictions(
        model, filename=os.path.join(args.log_dir, "best_train.png"), args=args
    )
    val_dl.dataset.plot_predictions(
        model, filename=os.path.join(args.log_dir, "best_eval.png"), args=args
    )
    test_dl.dataset.plot_predictions(
        model, filename=os.path.join(args.log_dir, "best_test.png"), args=args
    )

    # Eval best model on test set
    logging.info("TEST RESULTS (BEST MODEL)")
    eval(
        model=model,
        val_dl=test_dl,
        criterion=criterion,
        metrics=metrics,
        epoch=args.epochs + 1,
        args=args,
    )
    if args.wandb:
        metrics.log_to_wandb(args.epochs + 1, "test")

    cleanup(args)


if __name__ == "__main__":
    main()
