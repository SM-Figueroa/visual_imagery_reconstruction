import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

	def __init__(self, channels_in, channels_out, mid_channels=None, residual=False):
		super().__init__()
		self.residual = residual
		if not mid_channels:
			mid_channels = channels_out
		self.double_conv = nn.Sequential(
			nn.Conv2d(channels_in, mid_channels, kernel_size=3, padding=1, bias=False),
			nn.GroupNorm(1, mid_channels),
			nn.GELU(),
			nn.Conv2d(mid_channels, channels_out, kernel_size=3, padding=1, bias=False),
			nn.GroupNorm(1, channels_out))

	def forward(self, x):
		if self.residual:
			return F.gelu(x + self.double_conv(x))
		else:
			return self.double_conv(x)



class Down(nn.Module):
	def __init__(self, channels_in, channels_out, embedding_dim=256):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(channels_in, channels_in, residual=True),
			DoubleConv(channels_in, channels_out))

		# project time step to proper dimensions
		self.embedding_layer = nn.Sequential(
			nn.SiLU(),
			nn.Linear(
				embedding_dim,
				channels_out))

	def forward(self, x, t):
		x = self.maxpool_conv(x)
		embedding = self.embedding_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
		return x + embedding



class Up(nn.Module):
	def __init__(self, channels_in, channels_out, embedding_dim=256):
		super().__init__()

		self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
		self.conv = nn.Sequential(
			DoubleConv(channels_in, channels_in, residual=True),
			DoubleConv(channels_in, channels_out, channels_in // 2))

		self.embedding_layer = nn.Sequential(
			nn.SiLU(),
			nn.Linear(
				embedding_dim,
				channels_out))

	def forward(self, x, skip_x, t):
		x = self.up(x)
		x = torch.cat([skip_x, x], dim=1)
		x = self.conv(x)
		embedding = self.embedding_layer(t)[:,:,None,None].repeat(1, 1, x.shape[-2], x.shape[-1])
		return x + embedding


class SelfAttention(nn.Module):
	def __init__(self, channels, size):
		super(SelfAttention, self).__init__()
		self.channels = channels
		self.size = size
		self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
		self.ln = nn.LayerNorm([channels])
		self.ff_self = nn.Sequential(
			nn.LayerNorm([channels]),
			nn.Linear(channels, channels),
			nn.GELU(),
			nn.Linear(channels, channels))


	def forward(self, x):
		x = x.view(-1, self.channels, self.size **2).swapaxes(1,2)
		x_ln = self.ln(x)
		attention_value, _ = self.mha(x_ln, x_ln, x_ln)
		attention_value = attention_value + x # apparently += is a nono in this situation
		attention_value = self.ff_self(attention_value) + attention_value # apparently += is a nono in this situation
		return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
