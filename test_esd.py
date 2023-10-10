import process
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
duration = 10  # seconds
sampling_frequency = 93  # Hz
num_samples = int(duration * sampling_frequency)
time = np.linspace(0, duration, num_samples)

signals = []

# Generate the signals
signal1 = 100 - (time / duration) * 100  # Linear function from 100 to 0
signal2 = 50 - (time / duration) * 50    # Linear function from 50 to 0
signals.append(signal1)
signals.append(signal2)

# Decaying sine function with high frequency
frequency_high = 20  # High frequency
amplitude_high = 100
decay_high = 0.1
signal3 = amplitude_high * np.sin(2 * np.pi * frequency_high * time) * np.exp(-decay_high * time)

# Decaying sine function with lower frequency
frequency_low = 5   # Lower frequency
amplitude_low = 100
decay_low = 0.05
signal4 = amplitude_low * np.sin(2 * np.pi * frequency_low * time) * np.exp(-decay_low * time)

signals.append(signal3)
signals.append(signal4)

for i in range(4):

    cutoff_frequency = 1
    freq_unfilt, esd_unfilt = process.compute_esd(signals[i], sampling_frequency)
    filtered_signal = process.hfiltfilt(signals[i], sampling_frequency, cutoff_frequency, order=10)
    freq_filt, esd_filt = process.compute_esd(filtered_signal, sampling_frequency)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(221)
    plt.plot(time, signals[i])
    plt.title("Original Signal")

    plt.subplot(222)
    plt.plot(time, filtered_signal)
    plt.title(f"Filtered Signal, fc = {cutoff_frequency}")
    
    plt.subplot(223)
    plt.semilogy(freq_unfilt, esd_unfilt)
    plt.title("ESD of Original Signal")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy/Frequency')

    plt.subplot(224)
    plt.semilogy(freq_filt, esd_filt)
    plt.title("ESD of Filtered Signal")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Energy/Frequency')

    plt.tight_layout()
    plt.savefig(f"test_plot{i+1}.png")
    plt.show()
    
