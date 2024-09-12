import rosbag
import os

bag = rosbag.Bag('dvrk/rosbags/scale_2e-1_delay_5e-1.bag')
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for val in bag.get_type_and_topic_info()[1].values():
	types.append(val[0])

i = 0 
for topic, msg, t in bag.read_messages(topics=['/dvrk/PSM1/wrench_body_current', '/dvrk/PSM2/wrench_body_current']):
	if i==2: break
	print(topic)
	print(msg.wrench.force)
	i+=1
	