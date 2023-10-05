import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps

# List of CSV files to process
csv_files = ["data_files/l0.75s0.8.csv"]
error = []
# Loop through each CSV file and process the data
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # Find the indices where "click" is True
    click_indices = df.index[df['click']]

    # Ensure there are exactly 4 True values in "click"
    if len(click_indices) != 4:
        print(f"Warning: {csv_file} does not have exactly 4 'True' values in 'click' column.")
        continue

    # Initialize a list to store data segments
    data_segments = []

    # Split the data into segments using the click indices
    for i in range(len(click_indices)):
        start_idx = click_indices[i-1] if i > 0 else 0
        end_idx = click_indices[i] + 1
        segment = df.iloc[start_idx:end_idx]
        data_segments.append(segment)

    # Create a 2x2 subplot for each data segment
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(csv_file)

    # Plot d1, d2, d3, and d4 along with their derivatives for each segment
    for i, segment in enumerate(data_segments):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        cur_d = 'd{0}'.format(i+1)
        ax.plot(segment['time'], segment[cur_d], label=cur_d)

        # Calculate and plot derivatives
        derivative_d = segment[cur_d].diff() / segment["time"].diff()
#        derivative_d = derivative_d.fillna(0)
        
        smoothed_derivative_d = derivative_d.rolling(window=5).mean().fillna(0)

        ax.plot(segment['time'], smoothed_derivative_d, label='{0} Derivative'.format(cur_d))
        area = simps(np.maximum(smoothed_derivative_d, 0), segment['time'])
        print("Area for {0}, Target {1}: {2}".format(csv_file, i+1, area))
        
        ax.set_title(f"Segment {i+1}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("{0}_distance_plots.png".format(csv_file))
    plt.show()

#    print("Error metric for {0} latency and {1} scaling factor: {2}".format(
    

## Old code for generating surface plots
# Read the CSV file into a pandas DataFrame
# df = pd.read_csv('game_data.csv', header=None, names=['latency', 'scale', 'd1', 'd2', 'd3', 'd4', 'time'])

# # Compute the error for each trial
# df['error'] = df['d1'] + df['d2'] + df['d3'] + df['d4']

# # Create the first 3D surface plot for error
# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.set_xlabel('Latency')
# ax1.set_ylabel('Scale')
# ax1.set_zlabel('Error')
# ax1.set_title('Error vs. Latency and Scale')
# ax1.plot_trisurf(df['latency'], df['scale'], df['error'], cmap='viridis')

# # Create the second 3D surface plot for time
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_xlabel('Latency')
# ax2.set_ylabel('Scale')
# ax2.set_zlabel('Time')
# ax2.set_title('Time vs. Latency and Scale')
# ax2.plot_trisurf(df['latency'], df['scale'], df['time'], cmap='viridis')

# # Show the plots
# plt.show()

# # # Create a figure with two subplots
# # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # # Plot the error heatmap
# # ax1 = axes[0]
# # ax1.set_title('Error Heatmap')
# # im1 = ax1.imshow(df.pivot('latency', 'scale', 'error'), cmap='viridis', extent=[df['latency'].min(), df['latency'].max(), df['scale'].min(), df['scale'].max()])
# # ax1.set_xlabel('Latency')
# # ax1.set_ylabel('Scale')
# # fig.colorbar(im1, ax=ax1, label='Error')

# # # Plot the time heatmap
# # ax2 = axes[1]
# # ax2.set_title('Time Heatmap')
# # im2 = ax2.imshow(df.pivot('latency', 'scale', 'time'), cmap='viridis', extent=[df['latency'].min(), df['latency'].max(), df['scale'].min(), df['scale'].max()])
# # ax2.set_xlabel('Latency')
# # ax2.set_ylabel('Scale')
# # fig.colorbar(im2, ax=ax2, label='Time')

# # # Show the plots
# # plt.tight_layout()
# # plt.show()

# # Create a grid of unique latency and scale values
# latency_values = df['latency'].unique()
# scale_values = df['scale'].unique()

# # Create a grid of error and time values based on latency and scale
# error_grid = df.pivot_table(index='scale', columns='latency', values='error', aggfunc='sum')
# time_grid = df.pivot_table(index='scale', columns='latency', values='time', aggfunc='sum')

# # Create a figure with two subplots
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Plot the error heatmap
# ax1 = axes[0]
# ax1.set_title('Error Heatmap')
# im1 = ax1.imshow(error_grid, cmap='viridis', extent=[min(latency_values), max(latency_values), min(scale_values), max(scale_values)], aspect='auto', origin='lower')
# ax1.set_xlabel('Latency')
# ax1.set_ylabel('Scale')
# fig.colorbar(im1, ax=ax1, label='Error')

# # Plot the time heatmap
# ax2 = axes[1]
# ax2.set_title('Time Heatmap')
# im2 = ax2.imshow(time_grid, cmap='viridis', extent=[min(latency_values), max(latency_values), min(scale_values), max(scale_values)], aspect='auto', origin='lower')
# ax2.set_xlabel('Latency')
# ax2.set_ylabel('Scale')
# fig.colorbar(im2, ax=ax2, label='Time')

# # Show the plots
# plt.tight_layout()
# plt.show()
