from process import high_butter, compute_esd, compute_fft, compute_os_dist, compute_psd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
from scipy.signal import welch


# Which files to read in
param_set = [(0.45, 0.8), (0.45, 1.0)]

for i, params in enumerate(param_set):
    latency = params[0]
    scale = params[1]
    # target_num = params[2]      

    filename = f"data_files/set1/l{latency}s{scale}.csv"
    df = pd.read_csv(filename)

    # Find the indices where "click" is True
    click_indices = df.index[df['click']]
    df_noclick = df[~df["click"]]

    # Calculate mean and standard deviation of sampling rate in motion data file
    dt = df_noclick["time"].diff()
    fs = 1.0 / dt
    fs_mean = np.mean(fs)
    fs_std = np.std(fs)
    if fs_std > 5:
        print("Warning! Sampling Rate std is: ", fs_std, "!")

    # Set up figure
    fig, axes = plt.subplots(4, 4, figsize=(24, 12))

    target_distances = []
    osd = []
    norm_osd = []
    esd_metric_set = []
    psd_metric_set = []
    esd_fft_metric_set = []
    # Split the data into segments using the click indices
    for i, click_idx in enumerate(click_indices):

        # Get target distance for current segment
        cur_d = f'd{i+1}'
        target_distances.append(df[cur_d][click_idx])
        
        # Segment data
        start_idx = click_indices[i-1]+1 if i > 0 else 0
        end_idx = click_idx+1
        segment = df.iloc[start_idx:end_idx]
        signal = np.array(segment[cur_d])
        time = np.array(segment["time"])
        time = time - time[0]
        normalized_signal = signal / np.max(signal)
        
        # Calculate Overshoot distance
        osd.append(compute_os_dist(signal, segment["time"]))
        norm_osd.append(compute_os_dist(normalized_signal, segment["time"]))
        
        # Calculate ESD
        fc = 0.1 # Hz
        # order = 5
        # duration = len(signal) / fs_mean
        # padding_duration = 0.1 * duration # seconds
        # num_padding_samples = int(padding_duration * fs_mean) # per side
        # padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')
        # filtered_signal = high_butter(padded_signal, fs_mean, fc, order)
        # filtered_signal = filtered_signal[num_padding_samples:-num_padding_samples]
        freq, esd = compute_esd(normalized_signal, fs_mean)
        _, psd = compute_psd(normalized_signal, fs_mean)
        
        # Compute esd through fft
        freq_fft, signal_fft = compute_fft(normalized_signal, fs_mean)
        fft_mag = np.abs(signal_fft)**2
        
        # # Integrate over specified interval for total energy
        start_freq = 0
        start_idx = np.argmax(freq_fft >= start_freq)
        esd_metric = simps(esd, freq)
        psd_metric = simps(psd, freq)
        esd_metric_set.append(esd_metric)
        psd_metric_set.append(psd_metric)
        esd_fft_metric = simps(fft_mag, freq_fft)
        esd_fft_metric_set.append(esd_fft_metric)

        axes[0, i].plot(time, signal)
        axes[0, i].axhline(0, color='black')
        axes[0, i].set_title(f"Target {i+1}, OSD = {osd[-1]}")

        axes[1, i].plot(freq, psd)
        axes[1, i].set_xlim(-5, 40)
        #axes[1, i].axvline(0, linestyle='--')
        axes[1, i].set_title(f"PSD, Integral[0:] = {psd_metric}")

        axes[2, i].plot(freq, esd)
        axes[2, i].set_xlim(-5, 40)
        #axes[2, i].axvline(0, linestyle='--')
        axes[2, i].set_title(f"ESD, Integral[0:] = {esd_metric}")
        
        axes[3, i].plot(freq_fft[start_idx:], fft_mag[start_idx:])
        axes[3, i].set_xlim(-5, 40)
        #axes[3, i].axvline(0, linestyle='--')
        axes[3, i].set_title(f"FFT mag^2, Integral[0:] = {esd_fft_metric}")

    fig.suptitle(f"Latency {latency}, Scale {scale}, OSD (not norm) = {sum(osd):.3f}, PSD = {sum(psd_metric_set):.3f}, ESD = {sum(esd_metric_set):.3f}, ESD/FFT = {sum(esd_fft_metric_set):.3f}, Target Dist = {sum(target_distances):.3f}")
    plt.tight_layout()
    plt.savefig(f"figures/set1_psd/l{latency}s{scale}_psdesdfft.png")
    plt.show()
        
        
    # error_metric = c1 * sum(target_distances) + c2 * sum(esd_metric_set)
    # metric_data.append([latency, scale, error_metric, completion_time])
        
    # # Get desired segment
    # d = f"d{target_num}"
    # start_idx = click_indices[target_num-2]+1 if target_num > 1 else 0
    # end_idx = click_indices[target_num-1]+1
    # print(start_idx, end_idx)
    # segment = df.iloc[start_idx:end_idx]
    # time = segment["time"]
    # signal = np.array(segment[d])
    # normalized_signal = signal / signal[0]

    # # Calculate ESD
    # fc = 0.1 # Hz
    # order = 5
    # # duration = len(signal) / fs_mean
    # # padding_duration = 0.1 * duration # seconds
    # # num_padding_samples = int(padding_duration * fs_mean) # per side
    # # padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')
    # # filtered_signal = high_butter(padded_signal, fs_mean, fc, order)
    # # filtered_signal = filtered_signal[num_padding_samples:-num_padding_samples]

    # #freq_fft, esd_fft = compute_esd_from_fft(signal, fs_mean) 
    # # Integrate over specified interval for total energy
    # # start_freq = fc
    # # start_idx = np.argmax(freq >= start_freq)
    # # esd_metric = simps(esd[start_idx:], freq[start_idx:])
    # # freq, psd = welch(normalized_signal, fs=fs_mean)
    # freq, esd = compute_esd(normalized_signal, fs_mean)    
    # freq_fft, signal_fft = compute_fft(normalized_signal, fs_mean)
    # fft_mag = np.abs(signal_fft)**2
    # esd_metric = simps(esd, freq)
    # esd_fft_metric = simps(fft_mag, freq_fft)

    # # Plot
    



    
    
