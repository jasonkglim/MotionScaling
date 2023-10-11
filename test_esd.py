import process
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Define the parameters
duration = 10  # seconds
sampling_frequency = 93  # Hz
num_samples = int(duration * sampling_frequency)
time = np.linspace(0, duration, num_samples)

signals = []

# Generate the signals
# signal1 = 10 - (time / duration) * 10  # Linear function from 100 to 0
# signal2 = 5 - (time / duration) * 5   # Linear function from 50 to 0
# signals.append(signal1)
# signals.append(signal2)

# Decaying sine function with high frequency
frequency_high = 1  # High frequency
amplitude_high = 10
decay_high = 0.5
# signal3 = amplitude_high * (np.cos(2 * np.pi * frequency_high * time))**2 * np.exp(-decay_high * time)

# # Decaying sine function with lower frequency
# frequency_low = 0.1   # Lower frequency
# amplitude_low = 10
# decay_low = 0.5
# signal4 = amplitude_low * (np.cos(2 * np.pi * frequency_low * time))**2 * np.exp(-decay_low * time)

# signals.append(signal3)
# signals.append(signal4)

# signals.append( np.cos(2*np.pi* 0.01 * time) )
# signals.append( np.cos(2*np.pi* 0.1 * time) )
# signals.append( np.cos(2*np.pi* 1 * time) )

for i in range(1, 2):
    signal = np.cos(2*np.pi* 2 * time)  #amplitude_high * (np.cos(2 * np.pi * frequency_high * time))**2 * np.exp(-decay_high * time)
    padding_duration = duration / 2 # seconds
    num_padding_samples = int(padding_duration * sampling_frequency) # per side
    padded_signal = np.pad(signal, (num_padding_samples, num_padding_samples), 'constant')

    cutoff_frequency = 0.1
    filt_order = 6
    filtered_signal = process.high_butter(signal, sampling_frequency, cutoff_frequency, order=filt_order)
    filtered_padded_signal = process.high_butter(padded_signal, sampling_frequency, cutoff_frequency, order=filt_order)
    signal_fft = np.abs(np.fft.fft(signal))
    filtered_signal_fft = np.abs(np.fft.fft(filtered_signal))
    padded_signal_fft = np.abs(np.fft.fft(padded_signal))
    padded_filtered_signal_fft = np.abs(np.fft.fft(filtered_padded_signal[num_padding_samples:-num_padding_samples]))
    padded_freq = np.fft.fftfreq(len(padded_signal), 1.0 / sampling_frequency)
    freq = np.fft.fftfreq(len(signal), 1.0 / sampling_frequency)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(421)
    plt.plot(time, signal)
    plt.title("Original Signal")

    plt.subplot(422)
    plt.plot(padded_signal)
    plt.title(f"Padded Signal")

    plt.subplot(423)
    plt.plot(time, filtered_signal)
    plt.title(f"Filtered Original Signal, order = {filt_order}, fc = {cutoff_frequency}")

    plt.subplot(424)
    plt.plot(time, filtered_padded_signal[num_padding_samples:-num_padding_samples])
    # plt.ylim(-1, 1)
    plt.title("Unpadded Filtered Padded Signal")

    plt.subplot(425)
    plt.plot(freq, signal_fft)
    plt.xlim(0, 10)
    plt.title("FFT of Original Signal")

    plt.subplot(426)
    plt.xlim(0, 10)
    plt.plot(freq, filtered_signal_fft)
    plt.title("FFT of Filtered Signal")

    plt.subplot(427)
    plt.plot(padded_freq, padded_signal_fft)
    plt.xlim(0, 10)
    plt.title("FFT of Padded Signal")

    plt.subplot(428)
    plt.plot(freq, padded_filtered_signal_fft)
    plt.xlim(0, 10)
    plt.title("FFT of Filtered Padded Signal")

    

    plt.tight_layout()
    # plt.savefig(f"figures/test_plot{i+1}.png")
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
    
