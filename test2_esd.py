import process
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# Define the parameters
duration = 10  # seconds
sampling_frequency = 93  # Hz
num_samples = int(duration * sampling_frequency)
time = np.linspace(0, duration, num_samples)

signals = []

# Generate the signals
signal1 = 10 - (time / duration) * 10  # Linear function from 100 to 0
signal2 = 5 - (time / duration) * 5   # Linear function from 50 to 0
signals.append(signal1)
signals.append(signal2)

# Decaying sine function with high frequency
frequency_high = 1  # High frequency
amplitude_high = 10
decay_high = 0.5
signal3 = amplitude_high * (np.cos(2 * np.pi * frequency_high * time))**2 * np.exp(-decay_high * time)

# Decaying sine function with lower frequency
frequency_low = 1   # Lower frequency
amplitude_low = 5
decay_low = 0.5
signal4 = amplitude_low * (np.cos(2 * np.pi * frequency_low * time))**2 * np.exp(-decay_low * time)

signals.append(signal3)
signals.append(signal4)

# signals.append( np.cos(2*np.pi* 0.01 * time) )
# signals.append( np.cos(2*np.pi* 0.1 * time) )
# signals.append( np.cos(2*np.pi* 1 * time) )
# signal =  np.cos(2*np.pi* 2 * time)
padding_duration = 5 # seconds
num_padding_samples = padding_duration * sampling_frequency # per side


cutoff_frequency = 1
filt_order = 6


# Make figure
fig, axes = plt.subplots(3, len(signals), figsize=(24, 12))

for i, signal in enumerate(signals):

    padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')
    filtered_signal = process.high_butter(signal, sampling_frequency, cutoff_frequency, order=filt_order)
    filtered_padded_signal = process.high_butter(padded_signal, sampling_frequency, cutoff_frequency, order=filt_order)
    filtered_padded_signal = filtered_padded_signal[num_padding_samples:-num_padding_samples]
    freq, esd = process.compute_esd(signal, sampling_frequency)
    esd_metric = simps(esd, freq)
    # freq_unpad, esd_unpad = process.compute_esd(filtered_signal, sampling_frequency)
    freq_pad, esd_pad = process.compute_esd(filtered_padded_signal, sampling_frequency)
    start_freq = 1
    start_idx = np.argmax(freq >= start_freq)
    esd_metric_interval = simps(esd[start_idx:], freq[start_idx:])
    # esd_metric_unpad = simps(esd_unpad, freq_unpad)
    esd_metric_pad = simps(esd_pad, freq_pad)

    axes[0, i].plot(time, signal)
    axes[0, i].set_title("Original Signal")

    axes[1, i].semilogy(freq, esd)
    axes[1, i].axvline(start_freq)
    axes[1, i].set_title(f"ESD, Sum = {esd_metric:.3f}, Interval Sum ({start_freq}) = {esd_metric_interval:.3f}")

    axes[2, i].semilogy(freq_pad, esd_pad)
    axes[2, i].set_title(f"ESD of Filtered Signal (fc {cutoff_frequency}, order {filt_order}), Sum = {esd_metric_pad:.3f}")

    # plt.subplot(423)
    # plt.plot(time, filtered_signal)
    # plt.title("Filtered Original Signal")

    # plt.subplot(424)
    # plt.plot(time, filtered_padded_signal)
    # plt.title("Unpadded Filtered Padded Signal")

    # plt.subplot(425)
    # plt.semilogy(freq, esd)
    # plt.title(f"ESD of Original Signal = {esd_metric}")

    # plt.subplot(426)
    # plt.semilogy(freq_unpad, esd_unpad)
    # plt.title(f"ESD of unpadded filtered signal = {esd_metric_unpad}")

    # plt.subplot(427)
    # plt.semilogy(freq_pad, esd_pad)
    # plt.title(f"ESD of padded filtered signal = {esd_metric_pad}")


plt.tight_layout()
plt.savefig(f"figures/test_plots_esd_fc1.png")
plt.show()


# for i in range(len(signals)):

#     cutoff_frequency = 1
#     freq_unfilt, esd_unfilt = process.compute_esd(signals[i], sampling_frequency)
#     filtered_signal = process.high_butter(signals[i], sampling_frequency, cutoff_frequency, order=1)
#     freq_filt, esd_filt = process.compute_esd(filtered_signal, sampling_frequency)
    
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(221)
#     plt.plot(time, signals[i])
#     plt.title("Original Signal")

#     plt.subplot(222)
#     plt.plot(time, filtered_signal)
#     plt.title(f"Filtered Signal, fc = {cutoff_frequency}")
    
#     plt.subplot(223)
#     plt.semilogy(freq_unfilt, esd_unfilt)
#     plt.title("ESD of Original Signal")
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Energy/Frequency')

#     plt.subplot(224)
#     plt.semilogy(freq_filt, esd_filt)
#     plt.title("ESD of Filtered Signal")
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Energy/Frequency')

#     plt.tight_layout()
#     # plt.savefig(f"figures/test_plot{i+1}.png")
#     plt.show()
    
