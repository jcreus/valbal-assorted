import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import functions as fn

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

#df = pd.read_hdf('../data/ssi71/ssi71.h5', stop=20*3600*5)

df = pd.read_hdf('smol.h5')##', stop=20*3600*5)

p1 = df.raw_pressure_1.values
p2 = df.raw_pressure_2.values
p3 = df.raw_pressure_3.values
p4 = df.raw_pressure_4.values


f = lambda p: 44330 * (1-(p/101325.)**(1/5.255))
g = lambda p: 940.9411 * p**-0.8097

freq = 1
a, b = fn.biquad_lowpass(freq, 0.5, 20)
a2, b2 = fn.biquad_lowpass_derivative(1/60., 0.5, 20)

def e(t, x):
    x = np.concatenate((x,np.zeros(4)+x[0]))
    y = np.zeros(x.size) + x[0]
    dy = np.zeros(x.size)
    v = np.zeros(x.size-3)
    last = t[0]
    rej = []
    span = None
    pp = np.zeros(x.size)
    for i in range(0,x.size-4):
        slope = g(y[i-1])
        ts = 0.6/freq+(t[i]-last)
        vel = slope*(abs(x[i] - y[i-1] - 0 * ts * dy[i-1])-3.4*5)/(ts)
        if vel > 10:
            x[i] = y[i-1]
            #last += 1
            span = i
            pp[i] = np.nan
        else:
            if span is not None:
                #print(i, x.size)
                #plt.axvspan(span, i, color='black', alpha=0.2)
                span = None
            last = t[i]
            pp[i] = x[i]
        v[i] = vel
        y[i] = 1/a[0]*(b[0]*x[i] + b[1]*x[i-1]  + b[2]*x[i-2] - a[1]*y[i-1] - a[2]*y[i-2])
        dy[i] = 1/a2[0]*(b2[0]*x[i] + b2[1]*x[i-1]  + b2[2]*x[i-2] + b2[3]*x[i-3] - a2[1]*dy[i-1] - a2[2]*dy[i-2])
 
    return v, pp

t = df.index - df.index[0]
t = (t/ np.timedelta64(1, 's'))
v, y = e(t, p2)
print("hm", np.min(v), np.mean(v), np.max(v), np.median(v))
#plt.plot(p1)
plt.plot(df.altitude_barometer.values)
plt.plot(f(y))
plt.figure()
plt.plot(v)
plt.show()
exit()

v, y = e(p2)
#plt.plot(p2)
plt.plot(y)

v, y = e(p3)
#plt.plot(p3)
plt.plot(y)

#plt.plot(f(y))
plt.figure()
plt.plot(v)
plt.show()
exit()


f1 = butter_highpass_filter(p1, 1, 20)

print("std", np.std(f1[600000:1000000]))
#plt.plot(           f1[600000:1000000])
hmm = butter_highpass_filter(fn.biquad_filter(p1, a, b), 1, 20)
print("std", np.std(hmm[600000:1000000]))
plt.plot(hmm[600000:1000000])

plt.show()
exit()
abs = np.abs
diffs = np.maximum.reduce([abs(p1-p2), abs(p1-p3), abs(p1-p4), abs(p2-p3), abs(p2-p4), abs(p3-p4)])
plt.plot(diffs)
plt.figure()

plt.plot(df.raw_pressure_1.values)
plt.plot(df.raw_pressure_2.values)
plt.plot(df.raw_pressure_3.values)
plt.plot(df.raw_pressure_4.values)
plt.show()
