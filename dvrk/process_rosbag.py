import rosbag
import os

bag = rosbag.Bag('dvrk/rosbags/scale_2e-1_delay_5e-1.bag')
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for val in bag.get_type_and_topic_info()[1].values():
	types.append(val[0])

first = True
t0 = 0
for topic, msg, t in bag.read_messages(topics=['/dvrk/PSM1/wrench_body_current', '/dvrk/PSM2/wrench_body_current']):
	if first:
		t0 = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
		first = False
	whichPSM = topic.strip("/dvrk/wrench_body_current")
	print(msg.wrench.force)
	fx = msg.wrench.force.x
	fy = msg.wrench.force.y
	fz = msg.wrench.force.z
	t = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 - t0
	# if topic == '/dvrk/PSM1/wrench_body_current':
		
	