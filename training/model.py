import os
import sys
import torch
import torch.nn as nn
from torch import optim
from blocks import *
from diffusion import Diffusion
import torchvision
import torchvision.transforms as transforms
sys.path.append('../')
from utils.plts import save_images

class UNet(nn.Module):
    def __init__(self, channels_in=3, channels_out=3, time_embed_dim=256, device='cuda'):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.device = device

        # incoding and downsampling
        self.incode = DoubleConv(channels_in, 64) # wrapper for 2 convultional layers
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

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.encode_timestep(t, self.time_embed_dim)

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


def train(dataloader, epochs, lr, image_size, device='cuda', save_name=None):
    print("Starting Training...\n\n")

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    n_total_steps = len(dataloader)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.get_sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (i+1)%10==0:
                print(f'epoch {epoch+1}/{epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4}')

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, f'../results/CIFAR/training_progression/{epoch+1}.jpg')

    if save_name:
        torch.save(model.state_dict(), f'../results/models/{save_name}')


if __name__ == '__main__':

    epochs = 500
    lr = 3e-4
    image_size = 64
    batch_size = 12
    device = 'cuda'
    data_path = 'C:/Users/User/Documents/Python Projects/torch_data'


    transform = transforms.Compose([
        transforms.Resize(80),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True)


    train(train_loader, epochs, lr, image_size, device=device, save_name='CIFAR_0')
