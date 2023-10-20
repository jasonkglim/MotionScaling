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

Confirmed that it is general system lag causing issue. Ran program on laptop with 0 latency, 0.2 scale, and everything was fine at first. Then I noticed the issue a little and noticed that the clutch command was lagging, in other words I would press spacebar and there was some delay before seeing the indicator switch, which shouldn't be the case for 0 latency.

### Notes on psd/esd/fft metric stuff

- metrics still not aligning with what I want... ESD/PSD/FFT metrics will be very high unexpectedly for seemingly good signals..
- Maybe still too affected by DC component (currently looking at integral over whole spectrum)
- PSD/ESD have a spike at low freq that is not well sampled. only about three data points at 0, 0.3, 0.7 Hz, spike almost always occurs at 0.3 Hz. Always 0 Hz -> low, 0.3 Hz -> high peak, 0.7 Hz -> back down low. I think it is this spike that causes the weird differences, it can be very high for some signals that look smooth but not high for other smooth signals..
- can we get better resolution around this lower freq range?
- with FFT, the first freq after 0 is around 0.1-0.3 as well. But for FFT, 0 is always greatest, then it reduces.
- I think using the fft and removing 0 component might be best solution?
