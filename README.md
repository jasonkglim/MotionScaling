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

### Looking at fft integral metric after removing 0 component only
- heatmap of using fft only looks pretty wrong... trend is almost opposite of what we should expect
- for l=0, s=0.2, which should probably be best looking signal, metric total is 7.7k, but then drops to 3.5-4k for s=0.4, 0.6, 0.8, and then down to 2k for s = 1.0. All of these signals look "good" and I would expect them to give similar metrics..
- Also, why is metric for l=0.3, s=0.2, so high (9k), when signals look smooth and good? The magnitude at the lower frequencies just seem to be so much higher..
- Also, l=0.75, s=0.2, target 4 makes no sense. Metric is extremely high, ~4k, but signal is very smooth??
- Okay, so I guess using FFT with 0 removed does NOT seem like it works..

- Confirmed that FFT is dependent on signal duration. For identical sine functions, signal with twice duration yields approx twice value for integral. (Interestingly for 2x duration signal, peak value is 4x, with 2x resolution, so peak is narrower)


### Looking at PSD metric alone
- Metric is very high for all 0 latency signals.. may be dependent on the way it gets sampled?
- Issue could be errors in taking integral? Since resolution is so poor at lower frequencies of importance, could be a lot of error in integral approximation
- Even so, looking at the peaks of PSD, doesn't really align with what seems like a "good" signal..
- Conclusion: PSD doesn't seem to represent what I want as far as measuring the oscillations in the signal.. Seemingly good signals will give very high values compared to signals that are clearly 'worse' qualitatively..
