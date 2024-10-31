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
		# print(topics)
		topic_list = [item for item in list(topics) if 'PSM' in item]
		# print(topic_list)
		types = bag.get_type_and_topic_info()[0]
		for topic in topic_list:
			num_msgs = bag.get_type_and_topic_info()[1][topic][1]
			print(f"Num messages in {topic}: {num_msgs}")

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
		all_data[key] = pd.DataFrame(data)
	if psm_save_filepath is not None:
		with open(psm_save_filepath, 'wb') as f:
			pickle.dump(all_data, f)
	print("PSM force data saved to ", psm_save_filepath)

	return all_data	
	

# Process a set of bag files for user study.
if __name__=="__main__":

	# get all names of folders in base_folder, in format user_<name>
	base_folder = "C:/Users/jlimk/Documents/dvrk_trial_data"
	user_list = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
	user_list = ["user_pradhit"]
	# loop over all users folders
	for user in user_list:
		print("Processing ", user)
		bag_folder = f"{base_folder}/{user}/rosbags"
		# Create local folder to store processed data
		os.makedirs(f"dvrk/trial_data/{user}", exist_ok=True)
		# Loop over all files in bag_folder

		for test in os.listdir(bag_folder):
			test = test.strip(".bag")
			print("Processing test ", test)
			bag_path = f"C:/Users/jlimk/Documents/dvrk_trial_data/{user}/rosbags/{test}.bag"
			psm_save_filepath = f"dvrk/trial_data/{user}/{test}_psm_data.pkl"
			# video_save_filepath = f""
			print(user, test)
			all_data = process_rosbag(bag_path, psm_save_filepath)