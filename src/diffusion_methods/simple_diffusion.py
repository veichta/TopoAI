import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import logging

import numpy as np

from tqdm import tqdm

def linear_schedule(T):
    # return torch.linspace(0.0001, 0.02, T)
    return torch.linspace(0.01, 0.2, T)

def forward_diffusion(mask, alpha_hat, noise):
    mask_diffused = torch.zeros_like(mask)
    for b in range(mask.shape[0]):
        mask_diffused[b] = torch.sqrt(alpha_hat[b]) * mask[b] + torch.sqrt(1 - alpha_hat[b]) * noise[b]
    return mask_diffused
    

def simple_diffusion_step(args, model, train_dl, optimizer, criterion,
                     T, epoch):
    logging.info(f"Epoch {epoch + 1}/{args.epochs}")
    pbar = tqdm(train_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

    betas = linear_schedule(T)
    alphahats = torch.cumprod(1 - betas, dim=0)
    i = 0

    model.train()
    for img, mask in train_dl:

        img = img.to(args.device)
        mask = mask.to(args.device)

        mask = mask.unsqueeze(1)
        # weight = weight.to(args.device)

        img = img.permute(0, 3, 1, 2)
        img = img[:, :3, :, :]

        # mask = mask
        # img = img

        t = torch.randint(1, T, (mask.shape[0], )).to(args.device)
        noise = torch.randn_like(mask)
        mask_diffused = forward_diffusion(mask, alphahats[t-1], noise)
        noise_prediction = model(img, mask_diffused, t)

        optimizer.zero_grad()

        loss = criterion(noise_prediction, noise, reduction="mean")
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        pbar.update()

    pbar.close()

def eval_simple_diffusion(args, model, val_dl, epoch, metrics):
    logging.info(f"Eval epoch {epoch + 1}/{args.epochs}")
    pbar = tqdm(val_dl, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
    model.eval()
    metrics.start_epoch()

    betas = linear_schedule(args.T)
    alpha_cumprod = torch.cumprod(1 - betas, dim=0)
    im_num = 0
    num_img = 0
    with torch.no_grad():
        for img, mask in val_dl:
            num_img += 1
            if num_img > 3:
                break
            img = img.to(args.device)
            mask = mask.to(args.device)
            # weight = weight.to(args.device)

            img = img.permute(0, 3, 1, 2)
            img = img[:, :3, :, :]

            mask = mask.unsqueeze(1)

            x_t = torch.randn_like(mask)
            im_num += 1
            for t in range(args.T, 0, -1):
                noise = torch.randn_like(mask) if t > 1 else torch.zeros_like(mask)
                noise_prediction = model(img, x_t, torch.tensor([t], device=args.device))
                alpha_t = 1 - betas[t-1]
                alphahat_t = alpha_cumprod[t-1]
                sigma = torch.sqrt(betas[t-1])
                mean_prediction = (1 / torch.sqrt(alpha_t)) * (x_t - (betas[t-1] / torch.sqrt(1 - alphahat_t)) * noise_prediction)
                x_t = mean_prediction + sigma * noise
            # print(x_t)
            # print(x_t.shape, mask.shape)
            prediction = x_t.squeeze().squeeze().cpu().numpy()
            # prediction[prediction > 0.5] = 1
            # prediction[prediction <= 0.5] = 0
            plt.imsave(f"./test/final_prediction_{im_num}.png", prediction)
            # metrics.update(x_t.squeeze(0), mask.squeeze(0), torch.ones_like(mask.squeeze(0)))
            # pbar.set_postfix(
            #     # loss=np.mean(metrics.epoch_loss),
            #     # iou=np.mean(metrics.epoch_iou),
            #     acc=np.mean(metrics.epoch_acc),
            # )
            pbar.update()

    pbar.close()
    # metrics.end_epoch(epoch=epoch, mode="eval")