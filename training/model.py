import os
import sys
import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import time
sys.path.append('../')
from training.blocks import *
from training.diffusion import Diffusion
from utils.plts import save_images, plot_images
from utils.logs import print_time_elapsed


class UNet(nn.Module):
    def __init__(self, channels_in=3, channels_out=3, time_embed_dim=256, device='cuda', conditional=False):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.device = device

        # project fMRI labels to proper dimension
        if conditional:
            self.fmri_layer = nn.Linear(None, self.time_embed_dim)

        # incoding and downsampling
        self.incode = DoubleConv(channels_in, 64) # wrapper for 2 convolutional layers
        self.down1 = Down(64, 128) # first arg is input dim and second is output dim
        self.sa1 = SelfAttention(128, 32) # first arg is chan dim and second is resolution
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # upsampling and decoding
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.decode = nn.Conv2d(64, channels_out, kernel_size=1)

    def encode_timestep(self, t, channels):
        # encode timestep based on researched sinusoidal embedding
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.encode_timestep(t, self.time_embed_dim)

        if y is not None:
            t += self.fmri_layer(y)

        x1 = self.incode(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        result = self.decode(x)
        return result




def train_unconditional_diffusion(dataloader, epochs, lr, image_size, device='cuda', model_in=None):

    t_start = time.time()
    print("STARTING TRAINING...\n\n")

    if model_in:
        model = UNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        checkpoint = torch.load(model_in)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['last_epoch']
        model.train()

    else:
        model = UNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        last_epoch = 0
        
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    n_total_steps = len(dataloader)

    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.get_sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise(images, t)
            predicted_noise = model(x_t, t, None)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%100==0:
                print(f'\tepoch {last_epoch + epoch+1}/{last_epoch + epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4}')

        if (last_epoch + epoch)%50==0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, f'../results/all_imgs/training_progression/{last_epoch + epoch+1}.jpg')

    print('Saving model...')
    torch.save({
        'last_epoch': last_epoch + epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'../results/models/unconditional_diffusion_{last_epoch + epochs}_epochs.pt')

    print(f'COMPLETE')
    print_time_elapsed(t_start)






def train_conditional_diffusion(dataloader, epochs, lr, image_size, device='cuda', model_in=None):

    t_start = time.time()
    print("STARTING TRAINING...\n\n")

    if model_in:
        model = UNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        checkpoint = torch.load(model_in)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['last_epoch']
        model.train()

    else:
        model = UNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        last_epoch = 0
        
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    n_total_steps = len(dataloader)

    for epoch in range(epochs):
        for i, (images, fmri_arr) in enumerate(dataloader):
            images = images.to(device)
            fmri_arr = fmri_arr.to(device)
            t = diffusion.get_sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise(images, t)
            predicted_noise = model(x_t, t, fmri_arr)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%100==0:
                print(f'\tepoch {last_epoch + epoch+1}/{last_epoch + epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4}')

        if (last_epoch + epoch)%50==0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, f'../results/fmri_imgs/training_progression/{last_epoch + epoch+1}.jpg')

    print('Saving model...')
    torch.save({
        'last_epoch': last_epoch + epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'../results/models/conditional_diffusion_{last_epoch + epochs}_epochs.pt')

    print(f'COMPLETE')
    print_time_elapsed(t_start)

