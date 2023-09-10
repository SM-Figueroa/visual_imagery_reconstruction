import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch

def save_images(images, path, **kwargs):
	"""
	A function to convert torch tensor images to numpy format and save.

	Parameters
	----------
	images: torch.Tensor
		multiple images with 3 channels in RGB format.
	path: str
		file path to save images out to.

	Returns
	------
	None

	"""
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def plot_images(images):
	"""
	A function to convert torch tensor images to numpy format and plot.

	Parameters
	----------
	images: torch.Tensor
		multiple images with 3 channels in RGB format

	Returns
	------
	None
	
	"""
    plt.figure(figsize=(32,32))
    plt.imshow(torch.cat([torch.cat([i for i in images.cpu()], dim=1)], dim=-2).permute(1,2,0).cpu())
    plt.show()
