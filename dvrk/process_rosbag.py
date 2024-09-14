import rosbag
import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle

def process_rosbag(bag_path):
	# Open the ROS bag using a context manager
	with rosbag.Bag(bag_path, 'r') as bag:
		topics = bag.get_type_and_topic_info()[1].keys()
		
		# Initialize data storage for PSM1 and PSM2
		all_data = {
			"PSM1": {"time": [], "force_mag": []},
			"PSM2": {"time": [], "force_mag": []}
		}

		# Track the initial timestamp
		t0 = None
		
		# Process relevant topics in the bag
		for topic, msg, _ in bag.read_messages(topics=['/dvrk/PSM1/wrench_body_current', '/dvrk/PSM2/wrench_body_current']):
			# Set initial time reference
			if t0 is None:
				t0 = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9

			# Determine which PSM the message corresponds to
			psm_id = topic.strip("/dvrk/").replace('/wrench_body_current', '')

			# Calculate force magnitude
			force_vector = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
			force_magnitude = np.linalg.norm(force_vector)
			
			# Calculate relative timestamp
			timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 - t0
			
			# Store the timestamp and force magnitude
			all_data[psm_id]["time"].append(timestamp)
			all_data[psm_id]["force_mag"].append(force_magnitude)
		
		# Convert to pandas DataFrames
		for psm_id, data in all_data.items():
			all_data[psm_id] = pd.DataFrame(data)

	with open(bag_path.rstrip('bag') + 'pkl', 'wb') as f:
		pickle.dump(all_data, f)
	return all_data


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
