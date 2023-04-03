import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class resnet_block(nn.Module):
    '''Resnet block with time embedding, group norm and SiLU activation.'''
    def __init__(self, in_channels, out_channels):
        super(resnet_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(1, self.in_channels)
        self.gn2 = nn.GroupNorm(1, self.out_channels)
        self.act = nn.SiLU()

        time_dim = 32

        self.time_mlp = nn.Linear(time_dim, self.in_channels)

    def forward(self, x, t):
        x_ = x
        x = self.conv1(x)
        # x = self.gn1(x)
        x = self.act(x)
        if t is not None:
            print(t.shape)
            t_embed = F.relu(self.time_mlp(t)[0])
            t_embed = t_embed.unsqueeze(1).unsqueeze(2)
            x += t_embed
        x = self.conv2(x)
        # x = self.gn2(x)
        x_ = self.conv_res(x_)
        x = x + x_
        x = self.act(x)
        return x