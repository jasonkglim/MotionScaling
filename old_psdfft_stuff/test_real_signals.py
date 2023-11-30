from process import high_butter, compute_esd, compute_fft, compute_osd, compute_psd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
from scipy.signal import welch
import csv

# Which files to read in
param_set = [(0.0, 0.2, 4), (0.75, 0.8, 1), (0.75, 0.2, 4)]

fig, axes = plt.subplots(3, len(param_set), figsize=(24, 12))

for i, params in enumerate(param_set):
    latency = params[0]
    scale = params[1]
    target_num = params[2]      

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

    # Get desired segment
    d = f"d{target_num}"
    start_idx = click_indices[target_num-2]+1 if target_num > 1 else 0
    end_idx = click_indices[target_num-1]+1
    segment = df.iloc[start_idx:end_idx]
    time = np.array(segment["time"])
    time = time - time[0]
    signal = np.array(segment[d])
    normalized_signal = signal / np.max(signal)
        
     # Calculate ESD
    fc = 0.1 # Hz

    # filter stuff
    # order = 5
    # duration = len(signal) / fs_mean
    # padding_duration = 0.1 * duration # seconds
    # num_padding_samples = int(padding_duration * fs_mean) # per side
    # padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')
    # filtered_signal = high_butter(padded_signal, fs_mean, fc, order)
    # filtered_signal = filtered_signal[num_padding_samples:-num_padding_samples]

    # Calculate Overshoot distance
    osd = compute_osd(signal, time)  
    # norm_osd.append(compute_osd(normalized_signal, segment["time"]))
    
    freq, psd = compute_psd(normalized_signal, fs_mean)

    # Compute esd through fft
    freq_fft, signal_fft = compute_fft(normalized_signal, fs_mean)
    fft_mag = np.abs(signal_fft)**2

    # # Integrate over specified interval for total energy
    start_freq = 0.5
    start_idx = np.argmax(freq > start_freq)
    fft_start_idx = np.argmax(freq_fft > start_freq)
    psd_metric = simps(psd[start_idx:], freq[start_idx:]) * 100
    # esd_metric_set.append(esd_metric)
    # psd_metric_set.append(psd_metric)
    esd_fft_metric = simps(fft_mag[fft_start_idx:], freq_fft[fft_start_idx:])
    # esd_fft_metric_set.append(esd_fft_metric)

    with open(f"l{latency}s{scale}_t{target_num}_psd.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list(zip(freq, psd)))

    axes[0, i].plot(time, normalized_signal)
    axes[0, i].axhline(0, color='black')
    axes[0, i].set_title(f"Normalized Signal, latency {latency}, scale {scale}, target {target_num}")

    axes[1, i].plot(freq, psd, marker='o')
    axes[1, i].set_xlim(-5, 40)
    #axes[1, i].axvline(0, linestyle='--')
    axes[1, i].set_title(f"PSD scaled x100, Integral(0.5 Hz:] = {psd_metric}")

    # axes[2, i].scatter(freq, esd)
    # axes[2, i].set_xlim(-5, 40)
    # #axes[2, i].axvline(0, linestyle='--')
    # axes[2, i].set_title(f"ESD, Integral[0:] = {esd_metric}")

    axes[2, i].semilogy(freq_fft, fft_mag, marker='o')
    axes[2, i].set_xlim(-5, 40)
    axes[2, i].axvline(0, linestyle='--')
    axes[2, i].set_title(f"FFT mag^2, Integral(0.5 Hz:] = {esd_fft_metric}")

    # fig.suptitle(f"Latency {latency}, Scale {scale}, OSD = {osd:.3f}, PSD = {sum(psd_metric_set):.3f}, ESD = {sum(esd_metric_set):.3f}, ESD/FFT = {sum(esd_fft_metric_set):.3f}, Target Dist = {sum(target_distances):.3f}")
plt.tight_layout()
plt.savefig(f"figures/set1_psd/comparing_key_psdfftinterval.png")
plt.show()
        
        
    # error_metric = c1 * sum(target_distances) + c2 * sum(esd_metric_set)
    # metric_data.append([latency, scale, error_metric, completion_time])
        


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
    



    
    
