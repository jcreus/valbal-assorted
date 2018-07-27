import numpy as np
import matplotlib.pyplot as plt
import functions as fn

freq = 1

a, b = fn.biquad_lowpass(freq, 0.5, 20)

vs = np.linspace(0.1, 5, 100)
ds = []
for v in vs:
    vals = v*np.arange(1000)*0.05

    filt = fn.biquad_filter(vals, a, b)
    """plt.plot(vals)
    plt.plot(filt)
    plt.show()
    exit()"""
    delay = -(filt[-1]/(0.05*v)-vals[-1]/(0.05*v))
    ds.append(delay)

plt.plot(vs, ds)
#plt.plot(vals)
#plt.plot(filt)
plt.show()
