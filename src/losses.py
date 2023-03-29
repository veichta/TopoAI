import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
    def __init__(self, args):
        super(CustomBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")
        self.alpha = args.edge_weight

    def forward(self, inputs, targets, weights):
        loss = self.bce_loss(inputs, targets)
        loss += loss * (self.alpha * weights + targets)
        return loss.mean()


class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.bce_loss = CustomBCELoss(args)
        self.mIoU_loss = mIoULoss()

    def forward(self, inputs, targets, weights):
        mio_loss = self.mIoU_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets, weights)
        return mio_loss + bce_loss
