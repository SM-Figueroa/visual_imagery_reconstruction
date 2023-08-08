import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# hyper parameters
num_epochs = 500
batch_size = 4
learning_rate = 0.001

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset images are in range [0, 1]
# want to transform them to tensors then to range [-1, 1]
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# load training and test data from torch
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
	download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
	shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
	download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
	shuffle=True)

# store class names for printing/eval later
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
	'ship', 'truck')

# plot train data
examples = iter(train_loader)
samples, labels = next(examples)
for i in range(batch_size):
	img = samples[i]/2 + 0.5
	npimg = img.numpy()
	plt.subplot(2, 2, i+1)
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.title(classes[labels[i].item()])

# plt.show()

# create network with 2 convolutional/relu/maxpool layers and 3 linear layers (relu on 2)
class ConvNet(nn.Module):

	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# flatten images for linear layers
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model = ConvNet().to(device) # instantiate and send to device

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # adam optimizer, can change or play around

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):

		# push to device
		images  = images.to(device)
		labels = labels.to(device)

		# forward
		outputs = model(images)
		loss = criterion(outputs, labels)

		# backwards
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1)%2500 == 0:
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4}')


# test (no grad bc not training, don't want to update gradients)
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	n_class_correct = [0 for i in range(10)]
	n_class_samples = [0 for i in range(10)]

	# for each batch find proportion of correct preds
	for images, labels in test_loader:
		images  = images.to(device)
		labels = labels.to(device)
		outputs = model(images)

		# value, index
		_, predictions = torch.max(outputs, 1)
		n_samples += labels.shape[0]
		n_correct += (predictions == labels).sum().item()

		# for each sample in batch find correct preds for each class
		for i in range(batch_size):
			label = labels[i]
			pred = predictions[i]
			if (label == pred):
				n_class_correct[label] += 1
			n_class_samples[label] += 1

	# print results
	acc = 100.0 * n_correct/n_samples
	print(f'Accuracy of Network: {acc} %')

	for i in range(10):
		acc = 100.0 * n_class_correct[i] / n_class_samples[i]
		print(f'Accuracy of {classes[i]}: {acc} %')