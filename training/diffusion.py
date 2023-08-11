import os
import torch
import numpy as np
import matplotlib.pyplot as plt


class Diffusion:
	def __init__(self, steps=1000, beta_0=0.0001, beta_t=0.02, img_size=64, device='cuda'):
		self.steps = steps
		self.beta_0 = beta_0
		self.beta_t = beta_t
		self.img_size = img_size
		self.device = device

		self.beta = self.get_noise_schedule().to(device)
		self.alpha = 1 - self.beta
		self.alpha_h = torch.cumprod(self.alpha, dim=0)

	def get_noise_schedule(self):
		return torch.linspace(self.beta_0, self.beta_t, self.steps)

	def add_noise(self, x, t, show=False):
		# randn_like returns output of same dimensions but values are sample from N(0,1)
		e = torch.randn_like(x)
		x_noise = torch.sqrt(self.alpha_h[t]) * x + torch.sqrt(1-self.alpha_h[t]) * e
		if show:
			plt.imshow(np.transpose(x_noise.numpy(), (1,2,0)))
			plt.show()
		return torch.sqrt(self.alpha_h[t]) * x + torch.sqrt(1-self.alpha_h[t]) * e, e

	def show_noise_steps(self, x, end, step):
		fig, ax = plt.subplots(end//step//5, 5, figsize=(15,15))
		for t in range(0, end, step):
			x_noise, _ = self.add_noise(x, t)
			ax[t//step//5, t//step%5].imshow(np.transpose(x_noise.numpy(), (1,2,0)))
			ax[t//step//5, t//step%5].set_title(f'T = {t}')

		plt.tight_layout()
		plt.show()


	def get_sample_timestep(self, n):
		return torch.randint(low=1, high=self.steps, size=(n,))

	def sample(self, model, n):
		model.eval()
		with torchl.no_grad():
			# create x of complete noise sampled from N(0,1)
			x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
			for i in range(self.steps - 1, 0, -1):
				# create tensor of length n with current timestep
				t = (torch.ones(n) * i).long().to(self.device)
				pred_noise = model(x, t)
				if i > 1:
					added_noise = torch.randn_like(x)
				else:
					added_noise = torch.zeros_like(x)
				x = 1/torch.sqrt(self.alpha[t]) * (x - ((1 - self.alpha[t])/(torch.sqrt(1 - self.alpha_h[t])))*pred_noise) + torch.sqrt(self.beta[t])*added_noise

		model.train()
		# bring values between -1 and 1 then between 0 and 1, scale by 255 and convert to int
		x = (x.clamp(-1, 1) + 1) / 2
		x = (x*255).type(torch.uint8)
		return x
