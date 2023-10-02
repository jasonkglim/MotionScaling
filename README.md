Run mouse_target.py to play game
Run process.py to plot data

Notes:

data_files:
played games for 4 param combinations, noticing issue with clutch

data_files1:
played same 4 param combos, trying to replicate clutch issue, still present, especially noticeable for last combo, which is 0 latency, 0.2 scale. Clutching off and then moving mouse and then clutching on makes mouse jump, which is not behavior intended

data_files2:
No longer noticing effect... could be due to inherent system lag with VMware instead of code bug?

data_files3:
Definitely noticing effect now for params 0 latency, 0.2 scale. Definitely seems like an issue with inherent lag in the system.