import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from distribution_estimation import OnlineEmpiricalCDF


# generates an example signal of noisy measurements of delay following gaussian distribution
def generate_gauss_signal(duration, sampling_frequency, mean, std_dev):
    # Calculate the number of samples
    num_samples = int(duration * sampling_frequency)

    # Generate time values
    time = np.linspace(0, duration, num_samples, endpoint=False)

    # Generate signal sampled from a Gaussian distribution
    measurements = np.random.normal(mean, std_dev, num_samples)
    
    # Plot the original and delayed signals
    plt.plot(time, measurements, label='Delay Measurements')
    plt.xlabel('Time')
    plt.ylabel('Measured Delay')
    plt.legend()
    plt.show()

    return time, measurements


# Example usage
duration = 10  # seconds
sampling_frequency = 50  # Hz
mean_delay = 0.5  # seconds
std_dev_delay = 0.1  # seconds

time, delay_measurements = generate_gauss_signal(duration, sampling_frequency, mean_delay, std_dev_delay)


# Example usage
bin_width = 0.01
empirical_cdf(delay_measurements, bin_width)
online_cdf = OnlineEmpiricalDistribution()

# Simulate an online scenario
for data_point in delay_measurements:
    online_cdf.update(data_point)


plt.show()
