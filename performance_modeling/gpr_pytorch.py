import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Assuming 'metric_df' is your DataFrame with columns 'latency', 'scale', and 'performance'
# You might need to preprocess the data based on your specific requirements.

# Read in data
metric_df = pd.read_csv('data_files/user_jason/metric_df.csv')

# Extracting features and target variable
X = metric_df[['latency', 'scale']]
y = metric_df['throughput']

# Convert to torch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

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
model = GaussianProcessModel(X_tensor, y_tensor, likelihood)

# Set model to training mode
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
    output = model(X_tensor)
    loss = -mll(output, y_tensor)
    loss.backward()
    optimizer.step()

# Set the model and likelihood to evaluation mode
model.eval()
likelihood.eval()

# Make predictions on the test set

# Predict on a denser range of test inputs
latency_range = np.arange(0.0, 0.76, 0.01)
scale_range = np.arange(0.075, 1.025, 0.025)

test_inputs = torch.tensor(np.array(np.meshgrid(latency_range, scale_range)).T.reshape(-1, 2), dtype=torch.float32)

model.eval()
likelihood.eval()

# Get predictions and calculate mean and std
with torch.no_grad():
    predictions = likelihood(model(test_inputs))
    y_mean = predictions.mean
    y_std = predictions.stddev

# Create a DataFrame for the predicted values
pred_df = pd.DataFrame({
    'latency': test_inputs[:, 0].numpy(),
    'scale': test_inputs[:, 1].numpy(),
    'mean': y_mean.numpy(),
    'std': y_std.numpy()
})


# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# Plot original data
# Plot heatmap for average movement time
heatmap_TP = metric_df.pivot(
        index='latency', columns='scale', values='throughput')
#print(heatmap_TP)
ax = sns.heatmap(heatmap_TP, ax=axes[0], cmap="YlGnBu", annot=True, fmt='.3g')
axes[0].set_title('Throughput vs. Latency and Scale')
# annotate_extrema(heatmap_MT.values, ax, extrema_type='min')
indices = [(0, 0), (1, 1)]
#annotate(ax, indices)

heatmap_pred_TP = pred_df.pivot(
    index='latency', columns='scale', values='mean')
ax = sns.heatmap(heatmap_pred_TP, ax=axes[1], cmap="YlGnBu", xticklabels=9, yticklabels=9)
axes[1].set_title('Mean Predicted Throughput vs. Latency and Scale')

heatmap_pred_std = pred_df.pivot(
    index='latency', columns='scale', values='std')
ax = sns.heatmap(heatmap_pred_std, ax=axes[2], cmap="YlGnBu", xticklabels=9, yticklabels=9)
axes[2].set_title('Throughput Prediction STD vs. Latency and Scale')

plt.tight_layout()
plt.savefig('data_files/user_jason/gpr_results.png')
plt.show()