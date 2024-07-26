import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import distribution_estimation as de
from scipy.stats import norm
import pandas as pd


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

def bimodal_gauss_signal(mean1, std1, mean2, std2):
    # Calculate the number of samples
    num_samples = 250

    # Generate time values
    time = np.linspace(0, duration, num_samples, endpoint=False)

    # Generate signal sampled from a Gaussian distribution
    s1 = np.random.normal(mean1, std1, num_samples)
    s2 = np.random.normal(mean2, std2, num_samples)

    measurements = np.concatenate((s1, s2))
    
    # # Plot the original and delayed signals
    # plt.plot(time, measurements, label='Delay Measurements')
    # plt.xlabel('Time')
    # plt.ylabel('Measured Delay')
    # plt.legend()
    # plt.show()

    return measurements


# Example usage
duration = 10  # seconds
sampling_frequency = 50  # Hz
mean_delay = 0.5  # seconds
std_dev_delay = 0.1  # seconds

# time, delay_measurements = generate_gauss_signal(duration, sampling_frequency, mean_delay, std_dev_delay)
mean1 = 0.5
std1 = 0.1
mean2 = 0.7
std2 = 0.05
delay_measurements = bimodal_gauss_signal(mean1, std1, mean2, std2)
df = pd.read_csv(f"delay_modeling/network_test_data/JaeyOCU/trial1.csv")
real_data = df["tcp_rtt (ms)"]
bin_mode = 'auto'
window = 10
online_histogram = de.OnlineHistogram(bin_mode=bin_mode, window=window)

# Set up plot animation for iterative updating
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"histogram window: {window}, bin mode: {bin_mode}")
# Create the line plot for the signal
line_signal, = axs[0].plot([], [], marker='o')
axs[0].set_title("Signal")

# Create the step plot for the empirical CDF
step_cdf, = axs[1].step([], [], where='post')
axs[1].set_xlabel('Signal Values')
axs[1].set_ylabel('Cumulative Probability')
axs[1].set_title('Empirical CDF')
axs[1].axhline(0.9)
ref_percentile_line = axs[1].axvline(norm.ppf(0.9, loc=mean1, scale=std1))
axs[1].set_xlim(0, 1.0)
axs[1].set_ylim(0, 1.0)
percentile_line = axs[1].axvline(0, color='r', linestyle='--')
# Plot reference CDF
ref_x1 = np.linspace(mean1 - 4*std1, mean1 + 4*std1, 1000)
ref_x2 = np.linspace(mean2 - 4*std2, mean2 + 4*std2, 1000)
ref_cdf1 = norm.cdf(ref_x1, loc=mean1, scale=std1)
ref_cdf2 = norm.cdf(ref_x2, loc=mean2, scale=std2)
ref_cdf_line, = axs[1].plot(ref_x1, ref_cdf1)

# Create the bar plot for the PDF
bar_pmf = axs[2].bar([], [], width=[], alpha=0.5)
axs[2].set_xlabel('Signal Values')
axs[2].set_ylabel('Probability Density')
axs[2].set_title('Empirical PDF')
axs[2].set_xlim(0, 1.0)
axs[2].set_ylim(0, 1.0)


# Set tight layout
plt.tight_layout()

# Update function for the animation
def update_plot(frame):

    # update histogram and calculate empirical distributions
    # new_data_point = real_data[frame]
    new_data_point = delay_measurements[frame]
    pdf, edges = online_histogram.update(new_data_point)
    bin_width = np.diff(edges)
    cdf = np.cumsum(pdf)*bin_width
    pmf = pdf*bin_width

    # Update the line plot for the signal
    # line_signal.set_data(range(frame + 1), real_data[:frame + 1])
    line_signal.set_data(range(frame + 1), delay_measurements[:frame + 1])

    # Update the step plot for the empirical CDF
    step_cdf.set_data(edges[:-1], cdf)
    if frame > 1:
        percentile_line.set_xdata([de.value_at_percentile(cdf, edges, 90)])

    if frame == 250:
        ref_percentile_line.set_xdata([norm.ppf(0.9, loc=mean2, scale=std2)])
        ref_cdf_line.set_data(ref_x2, ref_cdf2)

    # Update the bar plot for the PMF
    axs[2].clear()
    bar_pmf = axs[2].bar(edges[:-1], pmf, width=bin_width, alpha=0.5)
    axs[2].set_xlabel('Signal Values')
    axs[2].set_ylabel('Probability Density')
    axs[2].set_title('Empirical PDF')

    # Adjust y-axis limits dynamically
    #    for ax in axs:
    axs[0].relim()
    axs[0].autoscale_view()

    return line_signal, step_cdf, bar_pmf

# Create the animation
# plt.rcParams['animation.convert_path'] = 'usr/bin/convert'
num_frames = len(delay_measurements)
ani = animation.FuncAnimation(fig, update_plot, frames=num_frames, interval=50, blit=True, repeat=False)


# Show the animation
# plt.show()


# writer = PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
ani.save(f'de_animations/realdata_JaeyOCU_trial1_binmode_{bin_mode}_window_{window}.gif', writer="pillow", fps=20)


# offline_histogram = de.OnlineHistogram(delay_measurements)
# offline_histogram.plot_pmf_cdf()
