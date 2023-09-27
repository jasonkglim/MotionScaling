import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('game_data.csv', header=None, names=['latency', 'scale', 'd1', 'd2', 'd3', 'd4', 'time'])

# Compute the error for each trial
df['error'] = df['d1'] + df['d2'] + df['d3'] + df['d4']

# Create the first 3D surface plot for error
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlabel('Latency')
ax1.set_ylabel('Scale')
ax1.set_zlabel('Error')
ax1.set_title('Error vs. Latency and Scale')
ax1.plot_trisurf(df['latency'], df['scale'], df['error'], cmap='viridis')

# Create the second 3D surface plot for time
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlabel('Latency')
ax2.set_ylabel('Scale')
ax2.set_zlabel('Time')
ax2.set_title('Time vs. Latency and Scale')
ax2.plot_trisurf(df['latency'], df['scale'], df['time'], cmap='viridis')

# Show the plots
plt.show()

# # Create a figure with two subplots
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Plot the error heatmap
# ax1 = axes[0]
# ax1.set_title('Error Heatmap')
# im1 = ax1.imshow(df.pivot('latency', 'scale', 'error'), cmap='viridis', extent=[df['latency'].min(), df['latency'].max(), df['scale'].min(), df['scale'].max()])
# ax1.set_xlabel('Latency')
# ax1.set_ylabel('Scale')
# fig.colorbar(im1, ax=ax1, label='Error')

# # Plot the time heatmap
# ax2 = axes[1]
# ax2.set_title('Time Heatmap')
# im2 = ax2.imshow(df.pivot('latency', 'scale', 'time'), cmap='viridis', extent=[df['latency'].min(), df['latency'].max(), df['scale'].min(), df['scale'].max()])
# ax2.set_xlabel('Latency')
# ax2.set_ylabel('Scale')
# fig.colorbar(im2, ax=ax2, label='Time')

# # Show the plots
# plt.tight_layout()
# plt.show()

# Create a grid of unique latency and scale values
latency_values = df['latency'].unique()
scale_values = df['scale'].unique()

# Create a grid of error and time values based on latency and scale
error_grid = df.pivot_table(index='scale', columns='latency', values='error', aggfunc='sum')
time_grid = df.pivot_table(index='scale', columns='latency', values='time', aggfunc='sum')

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the error heatmap
ax1 = axes[0]
ax1.set_title('Error Heatmap')
im1 = ax1.imshow(error_grid, cmap='viridis', extent=[min(latency_values), max(latency_values), min(scale_values), max(scale_values)], aspect='auto', origin='lower')
ax1.set_xlabel('Latency')
ax1.set_ylabel('Scale')
fig.colorbar(im1, ax=ax1, label='Error')

# Plot the time heatmap
ax2 = axes[1]
ax2.set_title('Time Heatmap')
im2 = ax2.imshow(time_grid, cmap='viridis', extent=[min(latency_values), max(latency_values), min(scale_values), max(scale_values)], aspect='auto', origin='lower')
ax2.set_xlabel('Latency')
ax2.set_ylabel('Scale')
fig.colorbar(im2, ax=ax2, label='Time')

# Show the plots
plt.tight_layout()
plt.show()
