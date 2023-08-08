import torch

### One epoch example

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad = True)

y_hat = w*x
s = y_hat - y

loss = s**2

loss.backward()

print(w.grad)


### Many epochs

epochs = 10
learning_rate = 0.1

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad = True)

for i in range(epochs):
	y_hat = w*x
	s = y_hat - y

	loss = s**2

	print(loss.item())

	loss.backward()
	with torch.no_grad():
		w -= learning_rate*w.grad
		w.grad.zero_()


### Torch implementation

# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training Loop
# 		- forward pass: compute prediction
# 		- backward pass: gradients
# 		- update weights

import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
	y_pred = model(X)

	l = loss(Y, y_pred)

	l.backward()

	optimizer.step()

	w.grad.zero_()

	if epoch%10 == 0:
		[w,b] = model.parameters()
		print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')


### Custom OOP

class LinearRegression(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(LinearRegression, self).__init__()
		# define layers
		self.lin = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.lin(x)


X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = LinearRegression(input_size, output_size)


learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
	y_pred = model(X)

	l = loss(Y, y_pred)

	l.backward()

	optimizer.step()

	w.grad.zero_()

	if epoch%10 == 0:
		[w,b] = model.parameters()
		print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')