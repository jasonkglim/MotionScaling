from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(256)

sp = fftshift(fft(np.sin(t)))

freq = fftshift(fftfreq(t.shape[-1]))

plt.plot(freq, sp.real, freq, sp.imag)
plt.show()
