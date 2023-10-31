import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps
import re
import os
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft, fftfreq, fftshift

### Functions for calculating metrics

# Calculates overshoot distance. takes in dataframe representing segment of trial
def compute_osd(distance, time):
    # Calculate positive area under derivative
    dist_derivative = np.gradient(distance, time)
    return simps(np.maximum(np.nan_to_num(dist_derivative, nan=0), 0), time)

# Calculates power spectral density of signal
def compute_psd(signal, fs):

    # Compute the PSD of the filtered signal
    frequencies, psd = welch(signal, fs=fs)

    return frequencies, psd

# Calculates energy spectral density of signal (PSD scaled by signal duration)
def compute_esd(signal, fs):

    # Compute the PSD of the filtered signal
    frequencies, psd = welch(signal, fs=fs)

    # Compute ESD
    duration = len(signal) / fs
    esd = psd * duration    

    return frequencies, esd

# Trying different method of computing esd by just squaring magnitude of fft
def compute_fft(signal, fs):

    # Compute fft of signal
    signal = np.array(signal)
    signal_fft = fftshift(fft(signal))
    freq = fftshift(fftfreq(len(signal), 1.0 / fs))

    # # ESD is magnitued squared
#    esd = (np.abs(signal_fft))**2

    return freq, signal_fft

# Compute a "speed metric" that represents task completion time. Using average speed of motion
def compute_speed_metric(signal, time):
    return np.abs(signal[-1] - signal[0]) / (time[-1] - time[0])


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
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Plot heatmap for target error
    heatmap_error = metric_df.pivot(
        index='latency', columns='scale', values='target_error')
    sns.heatmap(heatmap_error, ax=axes[0], cmap="YlGnBu", annot=True)
    axes[0].set_title('Target Error vs. Latency and Scale')

    # Plot the heatmap for 'osd metric'
    heatmap_osd = metric_df.pivot(
        index='latency', columns='scale', values='osd_metric')
    sns.heatmap(heatmap_osd, ax=axes[1], cmap="YlGnBu", annot=True)
    axes[1].set_title('OSD vs. Latency and Scale')

    # # Plot the heatmap for 'PSD metric' 
    # heatmap_psd = metric_df.pivot(
    #     index='latency', columns='scale', values='psd_metric')
    # sns.heatmap(heatmap_psd, ax=axes[2], cmap="YlGnBu")
    # axes[2].set_title('PSD (fc = 0.1) vs. Latency and Scale')

    # plot heatmap for speed metric
    heatmap_speed = metric_df.pivot(
        index='latency', columns='scale', values='speed_metric')
    sns.heatmap(heatmap_speed, ax=axes[2], cmap="YlGnBu", annot=True)
    axes[2].set_title('Avg Speed vs. Latency and Scale')

    # Plot heatmap for combined performance metric
    heatmap_combo = metric_df.pivot(
        index='latency', columns='scale', values='combo_metric')
    sns.heatmap(heatmap_combo, ax=axes[3], cmap="YlGnBu", annot=True)
    axes[3].set_title('Performance vs. Latency and Scale')

    # Adjust subplot layout
    plt.tight_layout()
    plt.savefig('figures/set1_psd/heatmaps_combo_metric.png')
    # Show the plots
    plt.show()


if __name__ == "__main__":

    # List of CSV files to process
    set_num = 2
    data_folder = f"data_files/set{set_num}"
    pattern = r'l(\d+\.\d+)s(\d+\.\d+)\.csv'
    count = 0

    # main data frame for performance metrics across trials
    metric_data = []

    # Coefficients for error metric
    c1 = 1
    c2 = 1
    c3 = 1

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
                target_df = pd.read_csv(f"target_data_l{latency}s{scale}.csv")

                # Total time to complete trial
                completion_time = df['time'].iloc[-1] - df['time'].iloc[0]
                
                # Find the indices where "click" is True
                click_indices = df.index[df['click']]
                df_noclick = df[~df["click"]]               

                data_segments = []
                target_distances = []
                osd_metric_set = []
                esd_metric_set = []
                psd_metric_set = []
                esd_fft_metric_set = []
                speed_metric_set = []

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

                # # Generate figure for metrics
                # fig, axes = plt.subplots(2, 4, figsize=(24, 12))

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
                    signal = np.array(segment[cur_d])
                    time = np.array(segment["time"])
                    time = time - time[0]
                    normalized_signal = signal / np.max(signal)

                    # Calculate Overshoot distance
                    osd_metric = compute_osd(signal, segment["time"])
                    osd_metric_set.append(osd_metric)

                    # Calculate speed metric
                    speed_metric = compute_speed_metric(signal, time)
                    speed_metric_set.append(speed_metric)

                    # Calculate ESD
                    fc = 0.5 # Hz
                    # order = 5
                    # duration = len(signal) / fs_mean
                    # padding_duration = 0.1 * duration # seconds
                    # num_padding_samples = int(padding_duration * fs_mean) # per side
                    # padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')
                    # filtered_signal = high_butter(padded_signal, fs_mean, fc, order)
                    # filtered_signal = filtered_signal[num_padding_samples:-num_padding_samples]

                    # Compute spectrums
                    freq, esd = compute_esd(normalized_signal, fs_mean)
                    freq, psd = compute_psd(normalized_signal, fs_mean)
                    freq_fft, signal_fft = compute_fft(normalized_signal, fs_mean)
                    fft_mag = np.abs(signal_fft)**2
                    
                    # Compute Metrics 
                    start_freq = 0.01
                    start_idx_fft = np.argmax(freq_fft > start_freq)
                    start_idx_psd = np.argmax(freq > start_freq)
                    esd_metric = simps(esd[start_idx_psd:], freq[start_idx_psd:])
                    psd_metric = simps(psd[1:], freq[1:]) * 100
                    esd_metric_set.append(esd_metric)
                    psd_metric_set.append(psd_metric)
                    esd_fft_metric = simps(fft_mag[start_idx_fft:], freq_fft[start_idx_fft:])
                    esd_fft_metric_set.append(esd_fft_metric)

                    # # Plot Metrics for each target signal
                    # axes[0, i].plot(time, signal)
                    # axes[0, i].axhline(0, color='black')
                    # axes[0, i].set_title(f"Target {i+1}, OSD = {osd_metric:.3f}, Avg Speed = {speed_metric:.3f}")

                    # axes[1, i].plot(freq, psd, marker='o')
                    # axes[1, i].set_xlim(-5, 40)
                    # #axes[1, i].axvline(0, linestyle='--')
                    # axes[1, i].set_title(f"PSD, Integral(0.1:] = {psd_metric:.3f}")

                    # axes[2, i].plot(freq, esd)
                    # axes[2, i].set_xlim(-5, 40)
                    # #axes[2, i].axvline(0, linestyle='--')
                    # axes[2, i].set_title(f"ESD, Integral(0.5:] = {esd_metric}")

                    # axes[3, i].plot(freq_fft[start_idx:], fft_mag[start_idx:])
                    # axes[3, i].set_xlim(-5, 40)
                    # #axes[3, i].axvline(0, linestyle='--')
                    # axes[3, i].set_title(f"FFT mag^2, Integral(0.5:] = {esd_fft_metric}")


                # fig.suptitle(f"Latency {latency}, Scale {scale}, OSD = {sum(osd_metric_set):.3f}, \
                # PSD = {sum(psd_metric_set):.3f}, Avg Speed = {np.mean(speed_metric_set):.3f}, Target Dist = {sum(target_distances):.3f}")
                # plt.tight_layout()
                # plt.savefig(f"figures/set1_psd/fc0.1/l{latency}s{scale}_psdesdfft_fc0.1.png")
                # plt.close()
                # plt.show()

                combo_metric = -c1*sum(target_distances) - c2*sum(osd_metric_set) + c3*np.mean(speed_metric_set)
                metric_data.append([latency, scale, sum(target_distances), sum(osd_metric_set), np.mean(speed_metric_set), combo_metric])
                
                
    metric_df = pd.DataFrame(metric_data, columns=['latency', 'scale', 'target_error', 'osd_metric', 'speed_metric', 'combo_metric'])
    metric_df.to_csv('data_files/set1/metric_df.csv')
    #plot_heatmaps(metric_df)
    
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

            
    #         smoothed_dist_derivative = dist_derivative.rolling(window=5).mean().fillna(0)

    #         ax.plot(segment['time'], smoothed_dist_derivative, label='{0} Derivative'.format(cur_d))
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
