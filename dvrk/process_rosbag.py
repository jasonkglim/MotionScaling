import rosbag
import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle
import cv2
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft, fftfreq, fftshift
# from sensor_msgs.msg import CompressedImage

# Define a Butterworth bandpass filter
def butter_bandpass(data, lowcut, highcut, fs, order=4):
	nyquist = 0.5 * fs
	low = lowcut / nyquist
	high = highcut / nyquist
	b, a = butter(order, [low, high], btype='band')
	y = filtfilt(b, a, data)
	return y


def compute_fft(signal, fs):

	# Compute fft of signal
	signal = np.array(signal)
	signal_fft = fftshift(fft(signal))
	freq = fftshift(fftfreq(len(signal), 1.0 / fs))

	# # ESD is magnitued squared
#    esd = (np.abs(signal_fft))**2

	return freq, signal_fft

def process_rosbag(bag_path, psm_save_filepath=None, video_save_filepath=None):
	'''
	Reads in and processes data from rosbag. 
	Saves PSM force data to pickle file and image data to video file if filepath is not none
	Returns PSM force data
	'''
	
	# Open the ROS bag using a context manager
	with rosbag.Bag(bag_path, 'r') as bag:
		# # Read topic info if necessary
		topics = bag.get_type_and_topic_info()[1].keys()
		print(topics)
		topic_list = [item for item in list(topics) if 'PSM' in item]
		print(topic_list)
		types = bag.get_type_and_topic_info()[0]
		# for topic in topics:
		# 	num_msgs = bag.get_type_and_topic_info()[1][topic][1]
		# 	print(f"Num messages in {topic}: {num_msgs}")

		# Initialize data storage for PSM1 and PSM2
		all_data = {}
		for topic in topic_list:
			all_data[topic.strip("/dvrk/").replace('/wrench_body_current', '')] = {"time": [], "force_mag": []}

		# Loop over all topics and messages
		t0 = None
		for topic, msg, _ in bag.read_messages(topics=topic_list):
			# Set initial time reference
			if t0 is None:
				t0 = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
			timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 - t0
			# print(msg)

			# If PSM topic, calculate forces
			# Determine which PSM the message corresponds to
			psm_id = topic.strip("/dvrk/").replace('/wrench_body_current', '')

			# Calculate force magnitude
			force_vector = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
			force_magnitude = np.linalg.norm(force_vector)
			
			# Store the timestamp and force magnitude
			all_data[psm_id]["time"].append(timestamp)
			all_data[psm_id]["force_mag"].append(force_magnitude)

	# Convert to pandas dataframe and save psm data
	for key, data in all_data.items():
		if key != "stereo_image":
			all_data[key] = pd.DataFrame(data)
	if psm_save_filepath is not None:
		with open(psm_save_filepath, 'wb') as f:
			pickle.dump(all_data, f)
	# print("PSM force data saved to ", save_file_prefix + '.pkl')

	return all_data

# Process user_trial_data 
def process_user_trial_data(user, test_name):
	'''
	Process trial data from video, i.e. time score and number of drops
	'''
	# Load data from csv file
	data = pd.read_csv(f"dvrk/trial_data/user_{user}/{user}_trial_data.csv")
	
	

# Process a set of bag files for user study.
if __name__=="__main__":

	scale = [1, 2, 3, 4]
	delay = [2, 5]
	test_names = [f"scale{s}_delay{d}" for s in scale for d in delay]
	# test_names = ["force_test", "force_test2", "force_test3"]
	user_list = ["neelay", "nikhil"]

	for user in user_list:
		os.makedirs(f"dvrk/trial_data/user_{user}", exist_ok=True)
		for test in test_names:
			# if test == "scale_1e-1_delay_2e-1" or test == "scale_5e-1_delay_2e-1":
			# 	continue
			bag_path = f"C:/Users/jlimk/Documents/dvrk_trial_data/user_{user}/rosbags/{test}.bag"
			psm_save_filepath = f"dvrk/trial_data/user_{user}/{test}_psm_data.pkl"
			# video_save_filepath = f""
			print(user, test)
			all_data = process_rosbag(bag_path, psm_save_filepath)

		# # plot PSM forces over time
		# fig, axs = plt.subplots(1, 4, figsize=(16, 6))
		# force_fft = {}
		# i = 1
		# for key, data in all_data.items():
		# 	if key != "stereo_image":
		# 		# print(len(data))
		# 		print(key)
		# 		axs[0].plot(data["time"], data["force_mag"], label=key)
		# 		fs = len(data) / data["time"].iloc[-1]
		# 		lcf = 0.1
		# 		hcf = 5
		# 		force_filtered = butter_bandpass(data["force_mag"], lowcut=lcf, highcut=hcf, fs=fs, order=3)
		# 		force_filtered_positive = np.maximum(force_filtered, 0)
		# 		filtered_mean = np.mean(force_filtered_positive)
		# 		axs[3].plot(data["time"], force_filtered, label=key+" Filtered")
		# 		freq, force_fft[key] = compute_fft(data["force_mag"], fs=fs)
		# 		window = 1000
		# 		idx = (int(len(freq)/2) - 1, int(len(freq)/2) + window)
		# 		axs[i].plot(freq[idx[0]:idx[1]], force_fft[key][idx[0]:idx[1]], label=key)
		# 		axs[i].legend()
		# 		i += 1
		# 		axs[0].set_xlabel('Time (s)')
		# 		axs[0].set_ylabel('Force (N)')
		# 		# axs[1].set_xlabel('Time (s)')
		# 		# axs[1].set_ylabel('Force (N)')
		# 		print(filtered_mean)
		# 		# axs[1].axhline(filtered_mean, label=key+" Mean")
		# 		axs[0].legend()
		# 		# axs[1].set_xlabel('Frequency (Hz)')
		# 		# axs[1].set_ylabel('FT')
		# # print(force_fft.keys())
		# # fft_diff = force_fft["PSM2"] - force_fft["PSM1"][:-1]
		# # axs[3].plot(freq[idx[0]:idx[1]], fft_diff[idx[0]:idx[1]], label='FFT Difference')
		# # axs[3].set_xlabel('Frequency (Hz)')
		# # axs[3].set_ylabel('FFT')
		# axs[3].set_title(f"Filtered, lcf = {lcf}, hcf = {hcf}")
		# # 	# else:
		# 		# plt.scatter(data["time_valid"], np.zeros(len(data["time_valid"])))
		# plt.suptitle("Force Magnitude of PSM arms")
		# plt.legend()
		# plt.tight_layout()
		# plt.savefig(f'dvrk/rosbags/processed_data/{test_name}_fft.png')
		# plt.show()

