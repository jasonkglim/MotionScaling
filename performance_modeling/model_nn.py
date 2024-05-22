import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read in data
df = pd.read_csv('data_files/set1/metric_df.csv')

# Normalize the data
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[['latency', 'scale', 'combo_metric']])
df[['latency', 'scale', 'combo_metric']] = scaled_values

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convert the data to PyTorch tensors
X = torch.tensor(df[['latency', 'scale']].values, dtype=torch.float32)
y = torch.tensor(df['combo_metric'].values, dtype=torch.float32).view(-1, 1)

# Initialize the neural network
net = Net()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training the neural network
for epoch in range(1000):
    optimizer.zero_grad()
    output = net(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Creating a meshgrid for prediction
latency_vals = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
scale_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
latency_mesh, scale_mesh = np.meshgrid(latency_vals, scale_vals)
X_mesh = np.column_stack((latency_mesh.ravel(), scale_mesh.ravel()))
X_mesh_tensor = torch.tensor(X_mesh, dtype=torch.float32)


# Predicting the values for the meshgrid
with torch.no_grad():
    predicted_values = net(X_mesh_tensor)
    predicted_surface = predicted_values.numpy().reshape(latency_mesh.shape)

# Create a DataFrame for meshgrid values and predictions
mesh_df = pd.DataFrame()
mesh_df['latency'] = latency_mesh.ravel()
mesh_df['scale'] = scale_mesh.ravel()
mesh_df['predicted_values'] = predicted_values_numpy.ravel()

# Using the DataFrame to plot the heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

heatmap_original = df.pivot(
    index='latency', columns='scale', values='combo_metric')
sns.heatmap(heatmap_original, ax=axes[0], cmap="YlGnBu", annot=True)
axes[0].set_title('Performance Metric vs. Latency and Scale')

heatmap_predicted = mesh_df.pivot(index='latency', columns='scale', values='predicted_values')
sns.heatmap(heatmap_predicted, cmap='YlGnBu', annot=True)
plt.title('Predicted Surface using Neural Network')
plt.tight_layout()
plt.show()
