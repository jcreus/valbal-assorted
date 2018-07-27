import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import functions as fn

df = pd.read_hdf('smol.h5')##', stop=20*3600*5)

def s(a,b):
    a.tofile('/home/joan/balloons-VALBAL/hootl/Filters/arrs/'+b+'.bin')

t = df.index - df.index[0]
t = (t/ np.timedelta64(1, 'ms'))
t = t.astype(np.uint32).values.astype(np.uint32)
print(t)
print(t[0], t[300])
p1 = df.raw_pressure_1.values
p2 = df.raw_pressure_2.values
p3 = df.raw_pressure_3.values
p4 = df.raw_pressure_4.values
s(p1, "p1")
s(p2, "p2")
s(p3, "p3")
s(p4, "p4")
s(t, "t")
print(len(t), len(p3))
