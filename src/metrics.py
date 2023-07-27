import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.morphology import skeletonize, skeletonize_3d

import wandb
from src.losses import Criterion

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road


def patch_accuracy_fn(inputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """Computes patch accuracy between inputs and targets.

    Args:
        inputs (torch.tensor): Model predictions.
        targets (torch.tensor): Ground truth.

    Returns:
        torch.tensor: The patch accuracy computed as the mean of the rounded inputs and targets.
    """
    h_patches = targets.shape[-2] // PATCH_SIZE
    w_patches = targets.shape[-1] // PATCH_SIZE
    patches_hat = (
        inputs.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    )
    patches = (
        targets.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    )
    return (patches == patches_hat).float().mean()


def patch_f1(inputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """Computes patch accuracy between inputs and targets.

    Args:
        inputs (torch.tensor): Model predictions.
        targets (torch.tensor): Ground truth.

    Returns:
        torch.tensor: The patch accuracy computed as the mean of the rounded inputs and targets.
    """
    h_patches = targets.shape[-2] // PATCH_SIZE
    w_patches = targets.shape[-1] // PATCH_SIZE
    patches_hat = (
        inputs.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    )
    patches = (
        targets.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    )

    # compute true positives, false positives, false negatives for each sample
    tp = (patches & patches_hat).float().sum((1, 2, 3))
    fp = (~patches & patches_hat).float().sum((1, 2, 3))
    fn = (patches & ~patches_hat).float().sum((1, 2, 3))

    # compute micro f1 score
    f1_scores = tp / (tp + 0.5 * (fp + fn) + 1e-8)
    return f1_scores.mean()


def accuracy_fn(inputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """Computes accuracy between inputs and targets.

    Args:
        inputs (torch.tensor): Model predictions.
        targets (torch.tensor): Ground truth.

    Returns:
        torch.tensor: The accuracy computed as the mean of the rounded inputs and targets.
    """
    return (inputs.round() == targets.round()).float().mean()


def iou_fn(inputs: torch.tensor, target: torch.tensor) -> torch.tensor:
    """Computes IoU between inputs and targets.

    Args:
        inputs (torch.tensor): Model predictions.
        targets (torch.tensor): Ground truth.

    Returns:
        torch.tensor: The IoU computed as the intersection over the union of the inputs and targets.
    """
    inputs = (inputs > 0.5).float()
    intersection = (inputs * target).sum()
    union = inputs.sum() + target.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v * s) / (np.sum(s) + 1e-8)


def clDice_fn(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    v_l = v_l.clone().detach().cpu().numpy()
    v_p = v_p.clone().detach().cpu().numpy()
    v_p[v_p > 0.5] = 1
    v_p = v_p.astype(np.uint8)
    # print(np.sum(v_p))
    cl_dice = []
    for i in range(v_p.shape[0]):
        tprec = cl_score(v_p[i], skeletonize(v_l[i]))
        tsens = cl_score(v_l[i], skeletonize(v_p[i]))
        cl_dice.append(2 * tprec * tsens / (tprec + tsens + 1e-8))
    return np.array(cl_dice).mean()


class Metrics:
    def __init__(self, loss_fn: Criterion):
        self.loss_fn = loss_fn
        self.iou_fn = iou_fn
        self.acc_fn = accuracy_fn
        self.f1_fn = patch_f1
        self.cl_dice_fn = clDice_fn

        self.train_loss = []
        self.train_bce = []
        self.train_miou = []
        self.train_mse = []
        self.train_topo =[]

        self.train_iou = []
        self.train_acc = []
        self.train_f1 = []
        self.train_cl_dice = []

        self.val_loss = []
        self.val_bce = []
        self.val_miou = []
        self.val_mse = []

        self.val_iou = []
        self.val_acc = []
        self.val_f1 = []
        self.val_cl_dice = []

    def start_epoch(self):
        """Starts a new epoch by resetting the metrics."""
        self.epoch_loss = []
        self.epoch_bce = []
        self.epoch_miou = []
        self.epoch_mse = []

        self.epoch_iou = []
        self.epoch_acc = []
        self.epoch_f1 = []
        self.epoch_cl_dice = []

    def update(
        self,
        pred: torch.tensor,
        target: torch.tensor,
        weight: torch.tensor,
        loss: torch.tensor = None,
    ):
        """Updates the metrics with the given predictions and targets.

        Args:
            pred (torch.tensor): Model predictions.
            target (torch.tensor): Ground truth.
            weight (torch.tensor): Weight for each prediction.
            loss (torch.tensor): Loss for the given predictions and targets.
        """
        if loss is None:
            loss = self.loss_fn(pred, target, weight)

        bce = self.loss_fn.bce_fn(pred, target, weight)
        miou = self.loss_fn.mIoU_fn(pred, target, weight)
        mse = self.loss_fn.mse_fn(pred, target, weight)

        iou = self.iou_fn(pred, target)
        acc = self.acc_fn(pred, target)
        f1 = self.f1_fn(pred, target)
        cl_dice = self.cl_dice_fn(pred, target)

        self.epoch_loss.append(loss.item())
        self.epoch_bce.append(bce.item())
        self.epoch_miou.append(miou.item())
        self.epoch_mse.append(mse.item())

        self.epoch_iou.append(iou.item())
        self.epoch_acc.append(acc.item())
        self.epoch_f1.append(f1.item())
        self.epoch_cl_dice.append(cl_dice.item())

    def end_epoch(self, epoch: int, mode: str, log_wandb: bool = False):
        """Ends the current epoch by computing the mean of the metrics and printing them.

        Args:
            epoch (int): The current epoch.
            mode (str): The current mode, either "train" or "eval".
        """
        if mode == "train":
            self.train_loss.append(np.mean(self.epoch_loss))
            self.train_bce.append(np.mean(self.epoch_bce))
            self.train_miou.append(np.mean(self.epoch_miou))
            self.train_mse.append(np.mean(self.epoch_mse))

            self.train_iou.append(np.mean(self.epoch_iou))
            self.train_acc.append(np.mean(self.epoch_acc))
            self.train_f1.append(np.mean(self.epoch_f1))
            self.train_cl_dice.append(np.mean(self.epoch_cl_dice))

        elif mode == "eval":
            self.val_loss.append(np.mean(self.epoch_loss))
            self.val_bce.append(np.mean(self.epoch_bce))
            self.val_miou.append(np.mean(self.epoch_miou))
            self.val_mse.append(np.mean(self.epoch_mse))

            self.val_iou.append(np.mean(self.epoch_iou))
            self.val_acc.append(np.mean(self.epoch_acc))
            self.val_f1.append(np.mean(self.epoch_f1))
            self.val_cl_dice.append(np.mean(self.epoch_cl_dice))

        else:
            raise ValueError(f"Unknown mode {mode}")

        self.print_metrics(epoch, mode)

        if log_wandb:
            self.log_to_wandb(epoch, mode)

    def print_metrics(self, epoch: int, mode: str):
        """Prints the metrics for the given epoch and mode.

        Args:
            epoch (int): The current epoch.
            mode (str): The current mode, either "train" or "eval".
        """
        if epoch > len(self.train_loss) - 1 and mode == "train":
            raise ValueError(f"Epoch {epoch} is out of range")
        elif epoch > len(self.val_loss) - 1 and mode == "eval":
            raise ValueError(f"Epoch {epoch} is out of range")

        if mode not in ["train", "eval"]:
            raise ValueError(f"Unknown mode {mode}")

        logging.info("-" * 30)
        logging.info(f"Epoch {epoch+1} ({mode}):")
        if mode == "train":
            logging.info(f"\tloss: {self.train_loss[epoch]:.4f}")
            logging.info(f"\tbce:  {self.train_bce[epoch]:.4f}")
            logging.info(f"\tmiou: {self.train_miou[epoch]:.4f}")
            logging.info(f"\tmse:  {self.train_mse[epoch]:.4f}")

            logging.info(f"\tiou:  {self.train_iou[epoch]:.4f}")
            logging.info(f"\tacc:  {self.train_acc[epoch]:.4f}")
            logging.info(f"\tf1:   {self.train_f1[epoch]:.4f}")
            logging.info(f"\tcld:  {self.train_cl_dice[epoch]:.4f}")

        elif mode == "eval":
            logging.info(f"\tloss: {self.val_loss[epoch]:.4f}")
            logging.info(f"\tbce:  {self.val_bce[epoch]:.4f}")
            logging.info(f"\tmiou: {self.val_miou[epoch]:.4f}")
            logging.info(f"\tmse:  {self.val_mse[epoch]:.4f}")

            logging.info(f"\tiou:  {self.val_iou[epoch]:.4f}")
            logging.info(f"\tacc:  {self.val_acc[epoch]:.4f}")
            logging.info(f"\tf1:   {self.val_f1[epoch]:.4f}")
            logging.info(f"\tcld:  {self.val_cl_dice[epoch]:.4f}")

        logging.info("-" * 30)

    def log_to_wandb(self, epoch: int, mode: str):
        """Logs the metrics to wandb.

        Args:
            epoch (int): The current epoch.
            mode (str): The current mode, either "train" or "eval".
        """
        if epoch > len(self.train_loss) - 1 and mode == "train":
            raise ValueError(f"Epoch {epoch} is out of range")
        elif epoch > len(self.val_loss) - 1 and mode == "eval":
            raise ValueError(f"Epoch {epoch} is out of range")

        if mode not in ["train", "eval", "test"]:
            raise ValueError(f"Unknown mode {mode}")

        if mode == "train":
            wandb.log(
                {
                    "train_loss": self.train_loss[epoch],
                    "train_bce": self.train_bce[epoch],
                    "train_miou": self.train_miou[epoch],
                    "train_mse": self.train_mse[epoch],
                    "train_iou": self.train_iou[epoch],
                    "train_acc": self.train_acc[epoch],
                    "train_f1": self.train_f1[epoch],
                    "train_cl_dice": self.train_cl_dice[epoch],
                },
                step=epoch,
            )
        elif mode == "eval":
            wandb.log(
                {
                    "val_loss": self.val_loss[epoch],
                    "val_bce": self.val_bce[epoch],
                    "val_miou": self.val_miou[epoch],
                    "val_mse": self.val_mse[epoch],
                    "val_iou": self.val_iou[epoch],
                    "val_acc": self.val_acc[epoch],
                    "val_f1": self.val_f1[epoch],
                    "val_cl_dice": self.val_cl_dice[epoch],
                },
                step=epoch,
            )
        elif mode == "test":
            wandb.log(
                {
                    "test_loss": self.val_loss[epoch],
                    "test_bce": self.val_bce[epoch],
                    "test_miou": self.val_miou[epoch],
                    "test_mse": self.val_mse[epoch],
                    "test_iou": self.val_iou[epoch],
                    "test_acc": self.val_acc[epoch],
                    "test_f1": self.val_f1[epoch],
                    "test_cl_dice": self.val_cl_dice[epoch],
                }
            )

    def save_metrics(self, filename: str):
        """Saves the metrics to a file.

        Args:
            filename (str): The name of the file.
        """
        metrics = {
            "train_loss": self.train_loss,
            "train_bce": self.train_bce,
            "train_miou": self.train_miou,
            "train_mse": self.train_mse,
            "train_iou": self.train_iou,
            "train_acc": self.train_acc,
            "train_f1": self.train_f1,
            "train_cl_dice": self.train_cl_dice,
            "val_loss": self.val_loss,
            "val_bce": self.val_bce,
            "val_miou": self.val_miou,
            "val_mse": self.val_mse,
            "val_iou": self.val_iou,
            "val_acc": self.val_acc,
            "val_f1": self.val_f1,
            "val_cl_dice": self.val_cl_dice,
        }

        with open(filename, "w") as f:
            json.dump(metrics, f, indent=4)

    def plot_metrics(self, filename: str = None):
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        ax[0, 0].plot(self.train_loss, label="train")
        ax[0, 0].plot(self.val_loss, label="val")
        ax[0, 0].set_title("Loss")
        ax[0, 0].legend()
        ax[0, 0].set_xlabel("Epoch")
        ax[0, 0].set_ylabel("Loss")

        ax[0, 1].plot(self.train_miou, label="train")
        ax[0, 1].plot(self.val_miou, label="val")
        ax[0, 1].set_title("mIoU")
        ax[0, 1].legend()
        ax[0, 1].set_xlabel("Epoch")
        ax[0, 1].set_ylabel("mIoU")

        ax[0, 2].plot(self.train_bce, label="train")
        ax[0, 2].plot(self.val_bce, label="val")
        ax[0, 2].set_title("BCE")
        ax[0, 2].legend()
        ax[0, 2].set_xlabel("Epoch")
        ax[0, 2].set_ylabel("BCE")

        ax[1, 0].plot(self.train_acc, label="train")
        ax[1, 0].plot(self.val_acc, label="val")
        ax[1, 0].set_title("Accuracy")
        ax[1, 0].legend()
        ax[1, 0].set_xlabel("Epoch")
        ax[1, 0].set_ylabel("Accuracy")

        ax[1, 1].plot(self.train_f1, label="train")
        ax[1, 1].plot(self.val_f1, label="val")
        ax[1, 1].set_title("Patch F1")
        ax[1, 1].legend()
        ax[1, 1].set_xlabel("Epoch")
        ax[1, 1].set_ylabel("F1")

        ax[1, 2].plot(self.train_iou, label="train")
        ax[1, 2].plot(self.val_iou, label="val")
        ax[1, 2].set_title("IoU")
        ax[1, 2].legend()
        ax[1, 2].set_xlabel("Epoch")
        ax[1, 2].set_ylabel("IoU")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()
