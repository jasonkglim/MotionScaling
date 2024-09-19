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

def process_rosbag(bag_path, save_file_prefix, process_video=False):
	'''
	Reads in and processes data from rosbag. 
	Saves PSM force data to pickle file and image data to video file if process_video is set to true
	Returns PSM force data

	'''
	
	# Open the ROS bag using a context manager
	with rosbag.Bag(bag_path, 'r') as bag:
		topics = bag.get_type_and_topic_info()[1].keys()
		for topic in topics:
			num_msgs = bag.get_type_and_topic_info()[1][topic][1]
			print(f"Num messages in {topic}: {num_msgs}")
		# Initialize data storage for PSM1 and PSM2
		all_data = {
			"PSM1": {"time": [], "force_mag": []},
			"PSM2": {"time": [], "force_mag": []},
			"stereo_image": {"time": [], "time_valid": []}
		}

		# Track the initial timestamp
		t0 = None
		
		# Process relevant topics in the bag
		frame_count = 0
		frames_to_skip = []
		cv_images = []
		topic_list = ['/dvrk/PSM1/wrench_body_current', 
					  '/dvrk/PSM2/wrench_body_current',
					  '/stereo/left/image/compressed']
		for topic, msg, _ in bag.read_messages(topics=topic_list):
			# Set initial time reference
			if t0 is None:
				t0 = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
			timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 - t0
                  
			# If PSM topic, calculate forces
			if topic.startswith("/dvrk"):

				# Determine which PSM the message corresponds to
				psm_id = topic.strip("/dvrk/").replace('/wrench_body_current', '')

				# Calculate force magnitude
				force_vector = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
				force_magnitude = np.linalg.norm(force_vector)
				
				# Store the timestamp and force magnitude
				all_data[psm_id]["time"].append(timestamp)
				all_data[psm_id]["force_mag"].append(force_magnitude)
            
			# If topic is stereo, save data to video
			elif process_video and topic.startswith("/stereo"):
                # Check if the image data is non-empty
				all_data["stereo_image"]["time"].append(timestamp)
				if msg.data:
					all_data["stereo_image"]["time_valid"].append(timestamp)
					np_arr = np.frombuffer(msg.data, np.uint8)
					cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

					# Check if cv_image is valid
					if cv_image is not None and cv_image.size > 0:
						# print(f"Frame {frame_count}: Successfully decoded. Shape: {cv_image.shape}")
						height, width, _ = cv_image.shape

						# Write the frame to the video
						cv_images.append(cv_image)
						frame_count += 1
					else:
						print(f"Frame {frame_count}: Decoding failed or generated an empty image.")
				else:
					# print(f"Frame {frame_count}: Empty image data, skipping frame.")
					frames_to_skip.append(frame_count)
					frame_count += 1
		
	if process_video:
		output_video_path = save_file_prefix + ".avi"
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		# print("Duration: ", timestamp)
		# print("Total frames: ", frame_count)
		# print("Num of frames skipped: ", len(frames_to_skip))
		fps_estimated = len(cv_images)/timestamp
		print("Approximated fps: ", fps_estimated)
		out = cv2.VideoWriter(output_video_path, fourcc, fps_estimated, (width, height))
		# Convert to pandas DataFrames
		for image in cv_images:
			out.write(image)
		# # Release the video writer object
		if out is not None:
			out.release()
		print(f"Video saved to {output_video_path}")

	# Convert to pandas dataframe
	for key, data in all_data.items():
		if key != "stereo_image":
			all_data[key] = pd.DataFrame(data)
			# force_filtered = butter_bandpass(all_data[key]["force_mag"], lowcut=lcf, highcut=hcf, fs=fs, order=3)
			# force_filtered_positive = np.maximum(force_filtered, 0)
			# filtered_mean = np.mean(force_filtered_positive)


	with open(save_file_prefix + '.pkl', 'wb') as f:
		pickle.dump(all_data, f)

	return all_data


if __name__=="__main__":
	scale = [2, 3, 4]
	delay = [2, 5]

	for s, d in itertools.product(scale, delay):
		bag_path = f'dvrk/rosbags/scale_{s}e-1_delay_{d}e-1.bag'
		all_data = process_rosbag(bag_path)

		# plot PSM forces over time
		plt.figure()
		for psm_id, data in all_data.items():
			plt.plot(data["time"], data["force_mag"], label=psm_id)
		plt.title("Force Magnitude of PSM arms")
		plt.xlabel('Time (s)')
		plt.ylabel('Force (N)')
		plt.savefig(bag_path.rstrip('bag') + 'png')
		plt.show()

	bag = rosbag.Bag('dvrk/rosbags/force_test.bag')
	topics = bag.get_type_and_topic_info()[1].keys()
	print(topics)
	types = []
	for val in bag.get_type_and_topic_info()[1].values():
		types.append(val[0])
	print(types)

	test_name = 'scale_2e-1_delay_2e-1'
	all_data = process_rosbag(f'dvrk/rosbags/{test_name}.bag', f'dvrk/rosbags/processed_data/{test_name}')

	# plot PSM forces over time
	fig, axs = plt.subplots(1, 4, figsize=(16, 6))
	force_fft = {}
	i = 1
	for key, data in all_data.items():
		if key != "stereo_image":
			# print(len(data))
			print(key)
			axs[0].plot(data["time"], data["force_mag"], label=key)
			fs = len(data) / data["time"].iloc[-1]
			lcf = 0.1
			hcf = 5
			force_filtered = butter_bandpass(data["force_mag"], lowcut=lcf, highcut=hcf, fs=fs, order=3)
			force_filtered_positive = np.maximum(force_filtered, 0)
			filtered_mean = np.mean(force_filtered_positive)
			axs[3].plot(data["time"], force_filtered, label=key+" Filtered")
			freq, force_fft[key] = compute_fft(data["force_mag"], fs=fs)
			window = 1000
			idx = (int(len(freq)/2) - 1, int(len(freq)/2) + window)
			axs[i].plot(freq[idx[0]:idx[1]], force_fft[key][idx[0]:idx[1]], label=key)
			axs[i].legend()
			i += 1
			axs[0].set_xlabel('Time (s)')
			axs[0].set_ylabel('Force (N)')
			# axs[1].set_xlabel('Time (s)')
			# axs[1].set_ylabel('Force (N)')
			print(filtered_mean)
			# axs[1].axhline(filtered_mean, label=key+" Mean")
			axs[0].legend()
			# axs[1].set_xlabel('Frequency (Hz)')
			# axs[1].set_ylabel('FT')
	# print(force_fft.keys())
	# fft_diff = force_fft["PSM2"] - force_fft["PSM1"][:-1]
	# axs[3].plot(freq[idx[0]:idx[1]], fft_diff[idx[0]:idx[1]], label='FFT Difference')
	# axs[3].set_xlabel('Frequency (Hz)')
	# axs[3].set_ylabel('FFT')
	axs[3].set_title(f"Filtered, lcf = {lcf}, hcf = {hcf}")
	# 	# else:
			# plt.scatter(data["time_valid"], np.zeros(len(data["time_valid"])))
	plt.suptitle("Force Magnitude of PSM arms")
	plt.legend()
	plt.tight_layout()
	plt.savefig(f'dvrk/rosbags/processed_data/{test_name}_fft.png')
	plt.show()

