from process import high_butter, compute_esd, compute_fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
from scipy.signal import welch


# Which files to read in
param_set = [(0.75, 1.0, 2), (0.75, 0.2, 4)]

# Set up figure
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
    print(start_idx, end_idx)
    segment = df.iloc[start_idx:end_idx]
    time = segment["time"]
    signal = segment[d]

    # Calculate ESD
    fc = 0.1 # Hz
    order = 5
    # duration = len(signal) / fs_mean
    # padding_duration = 0.1 * duration # seconds
    # num_padding_samples = int(padding_duration * fs_mean) # per side
    # padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')
    # filtered_signal = high_butter(padded_signal, fs_mean, fc, order)
    # filtered_signal = filtered_signal[num_padding_samples:-num_padding_samples]
    #freq, esd = compute_esd(signal, fs_mean)
    #freq_fft, esd_fft = compute_esd_from_fft(signal, fs_mean) 
    # Integrate over specified interval for total energy
    # start_freq = fc
    # start_idx = np.argmax(freq >= start_freq)
    # esd_metric = simps(esd[start_idx:], freq[start_idx:])
    freq, psd = welch(signal, fs=fs_mean)
    freq_fft, signal_fft = compute_fft(signal, fs_mean)
    fft_mag = np.abs(signal_fft)

    # Plot
    axes[0, i].plot(time, signal)
    axes[0, i].set_title(f"Original Signal: Latency {latency}, Scale {scale}, Target {target_num}")

    axes[1, i].plot(freq, psd)
    axes[1, i].set_title("PSD")

    axes[2, i].plot(freq_fft, fft_mag)
    axes[2, i].set_xlim(0, 40)
    axes[2, i].set_title("FFT")


plt.tight_layout()
plt.show()
    
    
