import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import distribution_estimation as de


# generates an example signal of noisy measurements of delay following gaussian distribution
def generate_gauss_signal(duration, sampling_frequency, mean, std_dev):
    # Calculate the number of samples
    num_samples = int(duration * sampling_frequency)

    # Generate time values
    time = np.linspace(0, duration, num_samples, endpoint=False)

    # Generate signal sampled from a Gaussian distribution
    measurements = np.random.normal(mean, std_dev, num_samples)
    
    # # Plot the original and delayed signals
    # plt.plot(time, measurements, label='Delay Measurements')
    # plt.xlabel('Time')
    # plt.ylabel('Measured Delay')
    # plt.legend()
    # plt.show()

    return time, measurements


# Example usage
duration = 10  # seconds
sampling_frequency = 50  # Hz
mean_delay = 0.5  # seconds
std_dev_delay = 0.1  # seconds

time, delay_measurements = generate_gauss_signal(duration, sampling_frequency, mean_delay, std_dev_delay)

online_histogram = de.OnlineHistogram()

# Set up plot animation for iterative updating
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Create the line plot for the signal
line_signal, = axs[0].plot([], [], marker='o')
axs[0].set_title("Signal")

# Create the step plot for the empirical CDF
step_cdf, = axs[1].step([], [], where='post')
axs[1].set_xlabel('Signal Values')
axs[1].set_ylabel('Cumulative Probability')
axs[1].set_title('Empirical CDF')
axs[1].axhline(0.9)
axs[1].axvline(0.6318)
percentile_line = axs[1].axvline(0, color='r', linestyle='--')

# Create the bar plot for the PDF
bar_pmf = axs[2].bar([], [], width=[], alpha=0.5)
axs[2].set_xlabel('Signal Values')
axs[2].set_ylabel('Probability Density')
axs[2].set_title('Empirical PDF')

# Set tight layout
plt.tight_layout()

# Update function for the animation
def update_plot(frame):

    # update histogram and calculate empirical distributions
    new_data_point = delay_measurements[frame]
    pdf, edges = online_histogram.update(new_data_point)
    bin_width = np.diff(edges)
    cdf = np.cumsum(pdf)*bin_width
    pmf = pdf*bin_width

    # Update the line plot for the signal
    line_signal.set_data(range(frame + 1), delay_measurements[:frame + 1])

    # Update the step plot for the empirical CDF
    step_cdf.set_data(edges[:-1], cdf)
    if frame > 1:
        percentile_line.set_xdata([de.value_at_percentile(cdf, edges, 90)])

    # Update the bar plot for the PMF
    axs[2].clear()
    bar_pmf = axs[2].bar(edges[:-1], pmf, width=bin_width, alpha=0.5)
    axs[2].set_xlabel('Signal Values')
    axs[2].set_ylabel('Probability Density')
    axs[2].set_title('Empirical PDF')

    # Adjust y-axis limits dynamically
    for ax in axs:
        ax.relim()
        ax.autoscale_view()

    return line_signal, step_cdf, bar_pmf

# Create the animation
num_frames = len(delay_measurements)
ani = FuncAnimation(fig, update_plot, frames=num_frames, interval=50, blit=True, repeat=False)

# Show the animation
plt.show()

# offline_histogram = de.OnlineHistogram(delay_measurements)
# offline_histogram.plot_pmf_cdf()
