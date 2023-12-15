import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'metric_df' is your DataFrame with columns 'latency', 'scale', and 'performance'
# You might need to preprocess the data based on your specific requirements.

# Read in data
metric_df = pd.read_csv('data_files/user_jason/metric_df.csv')

# Extracting features and target variable
X = metric_df[['latency', 'scale']]
y = metric_df['throughput']

# Convert to torch tensors
X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y.values, dtype=torch.float32)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_tensor, y_train_tensor, test_size=0.2, random_state=42)

# Define the GP model
class GaussianProcessModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcessModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GaussianProcessModel(X_train, y_train, likelihood)

# Training the model
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

# Set the model and likelihood to evaluation mode
model.eval()
likelihood.eval()

# Make predictions on the test set
with torch.no_grad():
    y_pred = likelihood(model(X_test))

# Extract mean and standard deviation
y_mean = y_pred.mean.numpy()
y_std = np.sqrt(y_pred.variance.numpy())

# Visualization
fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='r', marker='o', label='Actual Throughput')
ax.set_xlabel('Latency')
ax.set_ylabel('Scale')
ax.set_zlabel('Throughput')
ax.set_title('Actual Throughput vs. Latency and Scale')

ax = fig.add_subplot(132, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_mean, color='b', marker='o', label='Predicted Throughput (Mean)')
ax.set_xlabel('Latency')
ax.set_ylabel('Scale')
ax.set_zlabel('Throughput')
ax.set_title('Predicted Throughput (Mean) vs. Latency and Scale')

ax = fig.add_subplot(133, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_std, color='g', marker='o', label='Predicted Throughput (Std)')
ax.set_xlabel('Latency')
ax.set_ylabel('Scale')
ax.set_zlabel('Throughput')
ax.set_title('Predicted Throughput (Std) vs. Latency and Scale')

plt.tight_layout()
plt.show()
