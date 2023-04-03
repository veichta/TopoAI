import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from .resnet_block import resnet_block

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class base_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(base_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.resnet1 = resnet_block(self.in_channels, self.out_channels)
        self.resnet2 = resnet_block(self.out_channels, self.out_channels)

    def forward(self, x, t = None):
        x = self.resnet1(x, t)
        x = self.resnet2(x, t)
        return x
    
class encoder_decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, transpose) -> None:
        
        super(encoder_decoder_block, self).__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        self.double_resnet = base_encoder(out_channels, out_channels)

    def forward(self, x, t = None):
        x = self.conv(x)
        x = self.double_resnet(x, t)
        return x

    
    
class encoder_decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.encoder_blocks = nn.ModuleList()

        current_dimension = self.in_channels
        for i in range(4):
            self.encoder_blocks.append(encoder_decoder_block(current_dimension, current_dimension * 2, transpose=False))
            current_dimension *= 2
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(4):
            self.decoder_blocks.append(encoder_decoder_block(current_dimension, current_dimension // 2, transpose=True))
            current_dimension = current_dimension // 2

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t = None):
        x = self.encoder_blocks[0](x, t)
        x_1 = x
        x = self.encoder_blocks[1](x, t)
        x_2 = x
        x = self.encoder_blocks[2](x, t)
        x_3 = x
        x = self.encoder_blocks[3](x, t)
        
        x = self.decoder_blocks[0](x, t)
        x = self.decoder_blocks[1](x+x_3, t)
        x = self.decoder_blocks[2](x+x_2, t)
        x = self.decoder_blocks[3](x+x_1, t)

        x = self.conv(x)
        
        return x
    
class BaseDiffusion(nn.Module):
    def __init__(self, encoder_decoder_channels):
        '''
        Expects RGB image and binary mask
        '''
        super(BaseDiffusion, self).__init__()

        
        self.image_encoder = base_encoder(3, encoder_decoder_channels)
        self.mask_encoder = base_encoder(1, encoder_decoder_channels)
        self.decoder = encoder_decoder(encoder_decoder_channels, 1)
        
        self.time_embed = SinusoidalPositionEmbeddings(32)

        self.n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, x, mask, t):
        imgage_enc = self.image_encoder(x, None)
        mask_enc = self.mask_encoder(mask, None)

        x_intermediate = imgage_enc + mask_enc

        t = self.time_embed(t)

        output = self.decoder(x_intermediate, t)

        return output

