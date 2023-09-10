import numpy as np
import torch
import torchvision.transforms as T


class fMRILabeledDataLoader:
	
	def __init__(label_path, image_path, fmri_path):

		self.image_path = image_path
		self.fmri = np.load(fmri_path)
		self.current_idx = 0
		self.transform = T.Compose([
		        T.ToPILImage(),
		        T.Resize(image_size),
		        T.ToTensor()])

		with open(label_path) as f:
			self.labels = f.readlines()

	def __len__(self):
		return len(self.labels)

	def __iter__(self):
		return self

	def __next__(self):
		if self.current_idx < len(self.fmri):
			image_name = self.labels[self.current_idx]
			fmri_arr = torch.tensor(self.fmri[self.current_idx])
			image = self.transform(torch.load(f'{self.image_path}/{image_name}'))

			self.current_idx += 1

			return (image, fmri_arr)

		raise StopIteration

