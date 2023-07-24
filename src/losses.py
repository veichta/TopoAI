import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss

import numpy as np

def normalize_weights(weights: torch.tensor) -> torch.tensor:
    """Normalize weights to [0, 1] along the batch dimension.

    Args:
        weights (torch.tensor):
    """
    w_min = weights.min(dim=0)[0]
    w_max = weights.max(dim=0)[0]
    return (weights - w_min) / (w_max - w_min + 1e-8)


def to_one_hot_var(tensor, nClasses, requires_grad=False):
    """Convert a tensor to one-hot encoding.
    Args:
        tensor (Tensor): Tensor to convert.
        nClasses (int): Number of classes.
        requires_grad (bool): If True, the returned tensor will be a gradient

    Returns:
        Tensor: One-hot encoded tensor oder first dim.
    """
    n, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w).to(dtype=torch.int64), 1)
    return Variable(one_hot, requires_grad=requires_grad)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255, reduce=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce)

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        return self.nll_loss(log_p, targets)


class mIoULoss(nn.Module):
    def __init__(self):
        super(mIoULoss, self).__init__()

    def forward(self, inputs, target, is_target_variable=False):
        """
        Args:
            inputs (torch.Tensor): A tensor of shape (N, H, W)
            target (torch.Tensor): A tensor of shape (N, H, W)
            is_target_variable (bool): To specify if the target is a variable or not

        Returns:
            torch.Tensor: miou loss
        """

        N = inputs.size()[0]

        # target_one_hot => (N, C, H, W)
        inputs_oneHot = torch.stack([1 - inputs, inputs], dim=1)
        target_oneHot = to_one_hot_var(target, 2).float()

        inputs_oneHot = F.softmax(inputs_oneHot, dim=1)
        inter = inputs_oneHot * target_oneHot

        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, 2, -1).sum(2)

        # Denominator
        union = inputs_oneHot + target_oneHot - (inputs_oneHot * target_oneHot)

        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, 2, -1).sum(2)

        loss = (inter) / (union + 1e-8)

        return 1 - torch.mean(loss)


class CustomBCELoss(nn.Module):
    def __init__(self, args, size_average=True):
        super(CustomBCELoss, self).__init__()
        self.bce_fn = nn.BCELoss(reduction="none")
        self.alpha = args.edge_weight
        self.size_average = size_average

    def forward(
        self, inputs: torch.tensor, targets: torch.tensor, weights: torch.tensor
    ) -> torch.tensor:
        """Compute weighted binary cross entropy loss.

        Args:
            inputs (torch.tensor): Predicted probabilities with shape (N, H, W).
            targets (torch.tensor): Ground truth with shape (N, H, W).
            weights (torch.tensor): Edge weights with shape (N, H, W).

        Returns:
            torch.tensor: Weighted binary cross entropy loss.
        """
        loss = self.bce_fn(inputs, targets)
        # w = (1 - self.alpha) + self.alpha * weights
        return torch.mean(weights * loss) if self.size_average else torch.sum(weights * loss)


class CustomMSELoss(nn.Module):
    def __init__(self, args) -> None:
        super(CustomMSELoss, self).__init__()
        self.mse_fn = nn.MSELoss(reduction="none")
        self.alpha = args.edge_weight

    def forward(
        self, inputs: torch.tensor, targets: torch.tensor, weights: torch.tensor
    ) -> torch.tensor:
        """Compute weighted mean squared error loss.

        Args:
            inputs (torch.tensor): Predicted probabilities with shape (N, H, W).
            targets (torch.tensor): Ground truth with shape (N, H, W).
            weights (torch.tensor): Edge weights with shape (N, H, W).

        Returns:
            torch.tensor: Weighted mean squared error loss.
        """
        loss = self.mse_fn(inputs, targets)
        # weight = normalize_weights(weights)
        # w = (1 - self.alpha) + self.alpha * weight
        return torch.mean(weights * loss)


class FocalLoss(nn.Module):
    def __inti__(self, args, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.bce_fn = CustomBCELoss(args, size_average=False)
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        bce = self.bce_fn(inputs, targets, weights)
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean() if self.size_average else focal_loss.sum()


class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.bce_fn = CustomBCELoss(args)
        self.mse_fn = CustomMSELoss(args)
        self.mIoU_fn = mIoULoss()
        self.focal_fn = sigmoid_focal_loss

        self.args = args

    def forward(self, inputs, targets, weights):
        miou_loss = self.mIoU_fn(inputs, targets) * self.args.miou_weight
        bce_loss = self.bce_fn(inputs, targets, weights) * self.args.bce_weight
        mse_loss = self.mse_fn(inputs, targets, weights) * self.args.mse_weight
        focal_loss = self.focal_fn(inputs, targets, reduction="mean") * self.args.focal_weight

        return miou_loss + bce_loss + mse_loss + focal_loss
    
def GapLoss_weights(pred_mask, k = 2, corner_region = 4, to_plot=False):
    from skimage.morphology import skeletonize
    skeletons = torch.zeros_like(pred_mask)
    #convert to binary
    pred_mask_copy = pred_mask.clone().detach().cpu().numpy()
    pred_mask_copy[pred_mask_copy > 0.5] = 1
    pred_mask_copy = pred_mask_copy.astype(np.uint8)
    
    for i in range(pred_mask.shape[0]):
        skeleton = skeletonize(pred_mask_copy[i])
        skeletons[i] = torch.from_numpy(skeleton).to(dtype=torch.float, device=pred_mask.device)
    
    kernel_3x3 = torch.ones(size=(1, 1, 3, 3)).to(pred_mask.device)
    kernel_3x3[0, 0, 1, 1] = 0  
    C = torch.conv2d(skeletons.unsqueeze(1), weight=kernel_3x3, padding=1)
    C = (C == 1).float()
    
    corner_kernel = torch.ones(size=(1, 1, corner_region*2 + 1, corner_region*2 + 1)).to(pred_mask.device)
    #for c consider only pixels that belong to the skeleton
    C = C * skeletons.unsqueeze(1)
    W = torch.conv2d(C, corner_kernel, padding=corner_region).squeeze(1)
    W = W * k + torch.ones_like(W)
    
    if to_plot:
        return C.detach().cpu().numpy(), W.detach().cpu().numpy(), skeletons.detach().cpu().numpy()
    
    return W


def calculate_weights(pred, edge_weights, args):
    loss_weights = torch.zeros_like(pred).to(args.device)
            
    if args.edge_weight > 0:
        weight = normalize_weights(edge_weights.to(args.device))
        weight = (1 - args.edge_weight) + args.edge_weight * weight
        loss_weights += weight
        
    if args.gaploss_weight > 0:
        weight = GapLoss_weights(pred, args.gaploss_weight)
        weight = normalize_weights(weight.to(args.device))
        weight = (1 - args.gaploss_weight) + args.gaploss_weight * weight
        loss_weights += weight
        
    if args.gaploss_weight == 0 and args.edge_weight == 0:
        loss_weights = torch.ones_like(pred).to(args.device)
        
    if args.gaploss_weight > 0 and args.edge_weight > 0:
        loss_weights = loss_weights / 2
        
    return loss_weights
    
