from .base_unet import BaseUNet
from .base_diffusion import SinusoidalPositionEmbeddings

import torch
import torch.nn as nn

class UnetDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.unet = BaseUNet(chs=(32, 64, 128, 256, 512))

        self.time_embed = SinusoidalPositionEmbeddings(32)

        self.conv_pred = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        self.n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, mask, t = None):
        t = self.time_embed(t)
        x = self.image_encoder(x)
        mask = self.mask_encoder(mask)
        x_intermediate = x + mask
        x = self.unet(x, t)

        # x = self.conv_pred(x)

        return x
