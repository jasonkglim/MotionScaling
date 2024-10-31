### Code for analyzing dvrk trial data, generating performance metric datasets and heatmaps
import numpy as np
import pandas as pd
import pickle
import itertools
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import filtfilt, butter
import rosbag

# Define a Butterworth bandpass filter
def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Processes rosbag data and saves PSM force data to pickle file
def process_rosbag(bag_path, psm_save_filepath=None, video_save_filepath=None):
	'''
	Reads in and processes data from rosbag. 
	Saves PSM force data to pickle file and image data to video file if filepath is not none
	Returns PSM force data as dataframe
	'''
	
	# Open the ROS bag using a context manager
	with rosbag.Bag(bag_path, 'r') as bag:
		# # Read topic info if necessary
		topics = bag.get_type_and_topic_info()[1].keys()
		# print(topics)
		topic_list = [item for item in list(topics) if 'PSM' in item]
		# print(topic_list)
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
	print('PSM force data processed')
	return all_data

# Function for plotting psm and peg board forces
def plot_forces(psm_force_data, fts_data, save_filepath=None):
	'''
	Plots PSM and Peg Board forces over time
	'''
	fig, axs = plt.subplots(1, 2, figsize=(12, 6))
	fig.suptitle("Force Data")

	axs[0].set_title("Peg Board Force")
	axs[0].plot(fts_data[' Time '], fts_data[' Fx '], label="Fx")
	axs[0].plot(fts_data[' Time '], fts_data[' Fy '], label="Fy")
	axs[0].plot(fts_data[' Time '], fts_data[' Fz '], label="Fz")
	axs[0].legend()

	axs[1].set_title("PSM Arm Forces")
	for psm_id, psm_data in psm_force_data.items():
		axs[1].plot(psm_data["time"], psm_data["force_mag"], label=psm_id)
	axs[1].legend()
	if save_filepath is not None:
		plt.savefig(save_filepath, facecolor='w')
	plt.show()


if __name__ == "__main__":
	trial_data_base_folder = "C:/Users/jlimk/Documents/dvrk_trial_data"
	force_avgs = {'delay2': [], 'delay5': []}
	all_user_data = []
	user_list = ["soofiyan", "nikhil", "neelay", "calvin"]
	# scale, latency paramter combinations
	scale = [1, 2, 3, 4]
	delay = [2, 5]
	for user in user_list:
		print("Processing user ", user)

		# metric data for current user
		metrics = ["board_force", "psm_force", "time_score", "drop_penalty", "overall_score"]
		# metrics = ["board_force", "psm_force"]
		metric_data = pd.DataFrame(columns=["scale", "latency"] + metrics)
		metric_data_save_filepath = f"../dvrk/trial_data/user_{user}/metric_data"

		# Process trial video data (time score and drop penalties)
		video_data = pd.read_csv(f"{trial_data_base_folder}/user_{user}/{user}_trial_data.csv")

		# Loop over scale, delay parameters for each individual trial
		for s, d in itertools.product(scale, delay):

			print("Processing scale = ", s, " delay = ", d)
			# Skip certain trials
			if user == "sarah" and s == 1 and d == 2: continue
			test_name = f"scale{s}_delay{d}"

			# Extract the proper row from video data and process (time score and drop penalties)
			cur_video_data = video_data[(video_data['scale'] == s) & (video_data['latency'] == d)]
			rosbag_start_time = int(cur_video_data["rosbag_start_time"])
			end_time_str = str(cur_video_data["end_time"].values[0])
			start_time_str = str(cur_video_data["start_time"].values[0])
			if len(end_time_str.split(':')) == 1:
				end_time = int(end_time_str.split(':')[0])
			elif len(end_time_str.split(':')) == 2:
				end_time = int(end_time_str.split(':')[0])*60 + int(end_time_str.split(':')[1])
			if len(start_time_str.split(':')) == 1:
				start_time = int(start_time_str.split(':')[0])
			elif len(start_time_str.split(':')) == 2:
				start_time = int(start_time_str.split(':')[0])*60 + int(start_time_str.split(':')[1])
			end_time = end_time - rosbag_start_time
			start_time = start_time - rosbag_start_time
			completion_time = end_time - start_time - int(cur_video_data["time_credit"])
			time_score = completion_time / int(cur_video_data["num_transfers"]) # avg time per transfer

			# Load rosbag and fts data
			try:
				# psm_force_data = process_rosbag(f"{trial_data_base_folder}/user_{user}/rosbags/{test_name}.bag")
				fts_data_filepath = f"{trial_data_base_folder}/user_{user}/fts_files/{test_name}.csv"
				fts_data = pd.read_csv(fts_data_filepath, skiprows=6)
			except FileNotFoundError as e:
				print(f"File not found: {e}")
				continue
			
			# Convert to seconds and zero time data, remove bias and z axis drift
			fts_data[' Time '] = fts_data[' Time '].apply(lambda x: int(x.split(':')[0]) * 3600 + int(x.split(':')[1]) * 60 + float(x.split(':')[2]))
			fts_data[' Time '] = fts_data[' Time '] - fts_data[' Time '][0]
			fts_data[' Fx '] = fts_data[' Fx '] - fts_data[' Fx '].iloc[0]
			fts_data[' Fy '] = fts_data[' Fy '] - fts_data[' Fy '].iloc[0]
			fts_data[' Fz '] = fts_data[' Fz '] - fts_data[' Fz '].iloc[0]
			drift_slope = (fts_data[' Fz '].iloc[-1] - fts_data[' Fz '].iloc[0]) / (len(fts_data)-1)
			drift = [drift_slope * i + fts_data[' Fz '].iloc[0] for i in range(len(fts_data))]
			fts_data.loc[:, ' Fz '] = fts_data[' Fz '] - drift
			# Clip fts data to trial start and end times
			fts_data_full = fts_data.copy()
			fts_start_idx = fts_data[' Time '].sub(start_time).abs().idxmin()
			fts_end_idx = fts_data[' Time '].sub(end_time).abs().idxmin()
			fts_data = fts_data.loc[fts_start_idx:fts_end_idx]
			fts_data[' Time '] = fts_data[' Time '] - fts_data[' Time '].iloc[0] # rezero time data
			# Calculate Peg Board forces
			total_force_adjusted = np.sqrt(fts_data[' Fx ']**2 + fts_data[' Fy ']**2 + fts_data[' Fz ']**2)
			board_total_force = np.mean(total_force_adjusted) # Average force magnitude after adjusting for drift

			# # Calculate avg filtered psm forces
			# filtered_mean = []
			# for psm_id, psm_data in psm_force_data.items():
			# 	# Clip psm data
			# 	psm_data_full = psm_data.copy()
			# 	# psm_start_idx = psm_data['time'].sub(start_time).abs().idxmin()
			# 	# psm_end_idx = psm_data['time'].sub(end_time).abs().idxmin()
			# 	# psm_data = psm_data.loc[psm_start_idx:psm_end_idx].copy()  # Make a copy to avoid SettingWithCopyWarning
			# 	psm_data.loc[:, 'time'] = psm_data['time'] - psm_data['time'].iloc[0]  # Rezero time data
			
			# 	# Filter psm data with bandpass filter
			# 	order = 3
			# 	lcf = 0.1
			# 	hcf = 5
			# 	fs = len(psm_data_full) / psm_data_full["time"].iloc[-1]
			# 	force_filtered = butter_bandpass(psm_data["force_mag"], lowcut=lcf, highcut=hcf, fs=fs, order=order)
			# 	psm_data.loc[:, "force_filtered"] = force_filtered  # Add filtered force to dict for plotting later
			# 	force_filtered_positive = np.maximum(force_filtered, 0)
			# 	psm_data.loc[:, "force_filtered_positive"] = force_filtered_positive
			# 	filtered_mean.append(np.mean(force_filtered_positive))
			# 	# print(psm_id, " force_filtered_positive mean = ", np.mean(force_filtered_positive))
			# 	# psm_total_force += filtered_mean # sum of average filtered psm forces
			# psm_total_force = np.mean(filtered_mean)
			psm_total_force = 0

			# Print metrics
			print("Time score = ", time_score)
			print("Drop penalty = ", int(cur_video_data["num_drops"]))
			print("Peg board force = ", board_total_force)
			print("PSM total force = ", psm_total_force)
			print("Overall score = ???")

			# Plot full peg board and psm forces to ensure clip is correct
			# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
			# fig.suptitle("Full Force Data")

			# axs[0].set_title("Peg Board Force")
			# axs[0].plot(fts_data_full[' Time '], fts_data_full[' Fx '], label="Fx")
			# axs[0].plot(fts_data_full[' Time '], fts_data_full[' Fy '], label="Fy")
			# axs[0].plot(fts_data_full[' Time '], fts_data_full[' Fz '], label="Fz")
			# axs[0].axvline(start_time, color='r', linestyle='--', label='Start Time')
			# axs[0].axvline(end_time, color='g', linestyle='--', label='End Time')
			# axs[0].legend()

			# axs[1].set_title("PSM Arm Forces")
			# for psm_id, psm_data in psm_force_data.items():
			# 	axs[1].plot(psm_data["time"], psm_data["force_mag"], label=psm_id)
			# axs[1].axvline(start_time, color='r', linestyle='--', label='Start Time')
			# axs[1].axvline(end_time, color='g', linestyle='--', label='End Time')
			# axs[1].legend()
			# plt.show()

			# Plot Peg Board and PSM force magnitude over time
			# plot_forces(psm_force_data, fts_data)

			# Append metrics to data
			metric_data.loc[len(metric_data)] = [s, d, board_total_force, psm_total_force, time_score, int(cur_video_data["num_drops"]), 0]
			# metric_data.loc[len(metric_data)] = [s, d, board_total_force, psm_total_force, time_score, 0, 0]

			all_user_data.append(metric_data)
			
		# Save metric data?
		# metric_data.to_csv(metric_data_save_filepath + '.csv')

		# # Plot metric heatmaps
		# fig, ax = plt.subplots(2, 2, figsize=(18, 6))
		# # plt.figure()
		# title = (f"{user} force metrics")
		# fig.suptitle(title)
		# board_force_heatmap = metric_data.pivot(
		# 	index='latency', columns='scale', values='board_force'
		# )
		# sns.heatmap(board_force_heatmap, cmap='YlGnBu', ax=ax[0], annot=True, fmt='.3f')
		# ax[0].set_title('Peg Board Force')
		# ax[0].set_xlabel('Scale')
		# ax[0].set_ylabel('Latency')
		# # annotate_extrema(board_force_heatmap.values, ax[0], extrema_type='max')

		# psm_force_heatmap = metric_data.pivot(
		# 	index='latency', columns='scale', values='psm_force'
		# )
		# sns.heatmap(psm_force_heatmap, cmap='YlGnBu', ax=ax[1], annot=True, fmt='.3f')
		# ax[1].set_title('PSM Force')
		# ax[1].set_xlabel('Scale')
		# ax[1].set_ylabel('Latency')
		# # annotate_extrema(psm_force_heatmap.values, ax[1], extrema_type='max')

		# plt.tight_layout()
		# folder = "../figures/dvrk"
		# os.makedirs(folder, exist_ok=True)
		# filepath = f"{folder}/{user}_force"
		# plt.savefig(filepath, facecolor='w')
		# plt.show()

		# Calculate average board_force and psm_force across scale for latency 2 and 5
		force_avgs['delay2'].append(metric_data[metric_data['latency'] == 2].mean()['board_force'])
		force_avgs['delay5'].append(metric_data[metric_data['latency'] == 5].mean()['board_force'])

		

	# Plot average board force across scale for each user
	plt.figure(figsize=(10, 6))
	for data in all_user_data:
		scales = data['scale'].unique()
		forces = data[data['latency'] == 5]['board_force']
		plt.plot(scales, forces, marker='o')

	plt.legend([f"User {i}" for i in range(len(user_list))])
	plt.xlabel('Scale')
	plt.ylabel('Average Force')
	plt.title('Safety vs. Scale (Latency = .5 s)')
	plt.grid(True)
	plt.show()
	# plt.figure()
	# for data in all_user_data:
	# 	scales = data['scale'].unique()
	# 	forces = data[data['latency'] == 5]['board_force']
	# 	plt.plot(scales, forces)

	# plt.legend([f"User {i}" for i in range(len(user_list))])
	# plt.show()

	# Plot time score vs. scale for each user
	plt.figure(figsize=(10, 6))
	for data in all_user_data:
		scales = data['scale'].unique()
		time_scores = data[data['latency'] == 5]['time_score']
		plt.plot(scales, time_scores, marker='o')
	plt.legend([f"User {i}" for i in range(len(user_list))])
	plt.xlabel('Scale')
	plt.ylabel('Time Score')
	plt.title('Time Score vs. Scale (Latency = .5 s)')
	plt.grid(True)
	plt.show()
	