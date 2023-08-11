import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim


class Diffusion:
	def __init__(self, noise_)