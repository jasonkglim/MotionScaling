# Modeling performance with Gaussian Process Regression

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns

def annotate(ax, indices, color='red'):
    for (i, j) in indices:
        ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor=color, lw=3))


# Assuming 'metric_df' is your DataFrame with columns 'latency', 'scale', and 'performance'
# You might need to preprocess the data based on your specific requirements.

# Read in data
metric_df = pd.read_csv('data_files/user_jason/metric_df.csv')

# Extracting features and target variable
X = metric_df[['latency', 'scale']]
y = metric_df['throughput']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Gaussian Process kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2))

# Initialize the Gaussian Process Regressor with the chosen kernel
noise_std = 0.1
gp_model = GaussianProcessRegressor(kernel=kernel, alpha=noise_std, n_restarts_optimizer=10, random_state=42)

# Fit the model to the training data
gp_model.fit(X, y)
print(gp_model.kernel_)

# Predict on a denser range of test inputs
# Generate a grid of test inputs
latency_range = np.arange(0.0, 0.76, 0.01)
scale_range = np.arange(0.075, 1.025, 0.025)

# Create a meshgrid from the input ranges
latency_grid, scale_grid = np.meshgrid(latency_range, scale_range)
test_inputs = np.c_[latency_grid.ravel(), scale_grid.ravel()]
test_inputs = np.round(test_inputs, 3)

# Predict on the test inputs
y_pred, sigma = gp_model.predict(test_inputs, return_std=True)

# Reshape the predictions and uncertainties to match the grid shape
y_pred = y_pred.reshape(latency_grid.shape)
sigma = sigma.reshape(latency_grid.shape)

# Create a DataFrame for the predicted values
pred_df = pd.DataFrame({
    'latency': test_inputs[:, 0].flatten(),
    'scale': test_inputs[:, 1].flatten(),
    'mean': y_pred.flatten(),
    'std': sigma.flatten()
})

# # Evaluate the model performance (you can use other metrics as well)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error on the test set: {mse}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"Kernel: {gp_model.kernel_}, Noise std {noise_std}")
# Plot original data
# Plot heatmap for average movement time
heatmap_TP = metric_df.pivot(
        index='latency', columns='scale', values='throughput')
#print(heatmap_TP)
ax = sns.heatmap(heatmap_TP, ax=axes[0], cmap="YlGnBu", annot=True, fmt='.3g')
axes[0].set_title('Throughput vs. Latency and Scale')
# annotate_extrema(heatmap_MT.values, ax, extrema_type='min')

heatmap_pred_TP = pred_df.pivot(
    index='latency', columns='scale', values='mean')
ax = sns.heatmap(heatmap_pred_TP, ax=axes[1], cmap="YlGnBu", xticklabels=9, yticklabels=9)
axes[1].set_title('Mean Predicted Throughput vs. Latency and Scale')

heatmap_pred_std = pred_df.pivot(
    index='latency', columns='scale', values='std')
ax = sns.heatmap(heatmap_pred_std, ax=axes[2], cmap="YlGnBu", xticklabels=9, yticklabels=9)
axes[2].set_title('Throughput Prediction STD vs. Latency and Scale')

plt.tight_layout()
plt.savefig(f'data_files/gpr_results/dense_interpolation_alpha{noise_std}.png')
plt.show()
