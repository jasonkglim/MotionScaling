import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps
import re
import os
from scipy.signal import butter, filtfilt, welch

### Functions for calculating metrics

# Calculates overshoot distance. takes in dataframe representing segment of trial
def compute_os_dist(distance, time):
    # Calculate positive area under derivative
    derivative_d = distance.diff() / time.diff()
    return simps(np.maximum(derivative_d.fillna(0), 0), time)

# Calculates energy spectral density of high-passed signal
def compute_esd(signal, fs):

    # Compute the PSD of the filtered signal
    frequencies, psd = welch(signal, fs=fs)

    # Compute ESD
    duration = len(signal) / fs
    esd = psd * duration    

    return frequencies, esd


### Helper functions
# high pass bidirectional filter
def high_butter(signal, fs, fc, order):

    nyquist_frequency = 0.5 * fs
    
    # Design the high-pass filter
    b, a = butter(order, fc / nyquist_frequency, btype='high')

    # Apply the high-pass filter to the signal
    return filtfilt(b, a, signal)




### Plotting functions
def plot_heatmaps(metric_df):
    # Create a 1x2 subplot for the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the heatmap for 'error' vs. 'latency' and 'scale'
    heatmap_error = metric_df.pivot(
        index='latency', columns='scale', values='error')
    sns.heatmap(heatmap_error, ax=axes[0], cmap="YlGnBu")
    axes[0].set_title('Error vs. Latency and Scale')

    # Plot the heatmap for 'completion_time' vs. 'latency' and 'scale'
    heatmap_completion_time = metric_df.pivot(
        index='latency', columns='scale', values='completion_time')
    sns.heatmap(heatmap_completion_time, ax=axes[1], cmap="YlOrRd")
    axes[1].set_title('Completion Time vs. Latency and Scale')

    # Adjust subplot layout
    plt.tight_layout()
    plt.savefig('figures/set1_psd/heatmaps_set1.png')
    # Show the plots
    #plt.show()


if __name__ == "__main__":

    # List of CSV files to process
    set_num = 1
    data_folder = f"data_files/set{set_num}"
    pattern = r'l(\d+\.\d+)s(\d+\.\d+)\.csv'
    count = 0

    # main data frame for performance metrics across trials
    metric_data = []

    # Coefficients for error metric
    c1 = 1
    c2 = 1

    # cutoff frequency

    signals = []

    # Loop through each CSV file in data folder
    for filename in os.listdir(data_folder):
        # if count == 3:
        #     break
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)        
            match = re.match(pattern, filename)
            if match:

                count += 1

                latency = float(match.group(1))
                scale = float(match.group(2))            
                df = pd.read_csv(file_path)

                # Total time to complete trial
                completion_time = df['time'].iloc[-1] - df['time'].iloc[0]
                
                # Find the indices where "click" is True
                click_indices = df.index[df['click']]
                df_noclick = df[~df["click"]]               

                data_segments = []
                target_distances = []
                overshoot_distances = []
                esd_metric_set = []

                # Calculate mean and standard deviation of sampling rate in motion data file
                dt = df_noclick["time"].diff()
                fs = 1.0 / dt
                fs_mean = np.mean(fs)
                fs_std = np.std(fs)
                if fs_std > 5:
                    print("Warning! Sampling Rate std is: ", fs_std, "!")

                # print("Trial: ", latency, scale)
                # print("Mean and std of motion data sample rate: ", motion_fs_mean, motion_fs_std)
                # print("Mean and std of motion data sample rate no click: ", noclick_fs_mean, noclick_fs_std)
                # print("Mean and std of track_mouse fn call sample rate: ", tt_fs_mean, tt_fs_std)
                # print()


                # Split the data into segments using the click indices
                for i, click_idx in enumerate(click_indices):

                    # Get target distance for current segment
                    cur_d = f'd{i+1}'
                    target_distances.append(df[cur_d][click_idx])

                    # Segment data
                    start_idx = click_indices[i-1]+1 if i > 0 else 0
                    end_idx = click_idx+1
                    segment = df.iloc[start_idx:end_idx]
                    data_segments.append(segment)
                    signal = segment[cur_d]

                    # Calculate Overshoot distance
                    overshoot_distances.append(compute_os_dist(signal, segment["time"]))

                    # Calculate ESD
                    # fc = 0.1 # Hz
                    # order = 5
                    # duration = len(signal) / fs_mean
                    # padding_duration = 0.1 * duration # seconds
                    # num_padding_samples = int(padding_duration * fs_mean) # per side
                    # padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')
                    # filtered_signal = high_butter(padded_signal, fs_mean, fc, order)
                    # filtered_signal = filtered_signal[num_padding_samples:-num_padding_samples]
                    # freq, esd = compute_esd(signal, fs_mean)
                    # # Integrate over specified interval for total energy
                    # start_freq = fc
                    # start_idx = np.argmax(freq >= start_freq)
                    # esd_metric = simps(esd[start_idx:], freq[start_idx:])
                    # esd_metric_set.append(esd_metric)

                    # fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    # fig.suptitle(f"Latency = {latency}, Scale = {scale}")
                    if (latency == 0.75 and scale == 0.2 and i == 3) or (latency == 0.75 and scale == 1.0 and i == 1):

                        signal_data = {"signal": signal, "time": segment["time"], "latency": latency, "scale": scale, "target": i+1}
                        signals.append(signal_data)

        
                # for i in range(4):
                #     print(f"Target error: {target_distances[i]}, Overshoot error: {overshoot_distances[i]}")

                # error_metric = c1 * sum(target_distances) + c2 * sum(esd_metric_set)
                # metric_data.append([latency, scale, error_metric, completion_time])
                
    # metric_df = pd.DataFrame(metric_data, columns=['latency', 'scale', 'error', 'completion_time'])
    # plot_heatmaps(metric_df)
    
    # print(metric_df)
                    # # Create a 2x2 subplot for each data segment
                    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    # fig.suptitle(csv_file)

                    # # Plot d1, d2, d3, and d4 along with their derivatives for each segment
                    # for i, segment in enumerate(data_segments):
                    #     row, col = i // 2, i % 2
                    #     ax = axes[row, col]
                    #     cur_d = 'd{0}'.format(i+1)
                    #     ax.plot(segment['time'], segment[cur_d], label=cur_d)

    # Calculate and plot derivatives

            
    #         smoothed_derivative_d = derivative_d.rolling(window=5).mean().fillna(0)

    #         ax.plot(segment['time'], smoothed_derivative_d, label='{0} Derivative'.format(cur_d))
    #         area = 
    #         print("Area for {0}, Target {1}: {2}".format(csv_file, i+1, area))
            
    #         ax.set_title(f"Segment {i+1}")
    #         ax.legend()

    #     plt.tight_layout()
    #     plt.savefig("{0}_distance_plots.png".format(csv_file))
    #     plt.show()

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
