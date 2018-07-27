import pandas as pd
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

df = pd.read_hdf('ssi63atmo.h5')#,start=35000,stop=428000)#,start=20*60*60*20, stop=20*60*60*40)
plt.plot(df.altitude_barometer.values)
plt.show()


h = df.altitude_barometer.values
val = df.valve_time_total.values
bal = df.ballast_time_total.values

klin = 10
sample_time = 1 #sample time in minutes
intv = int(60*sample_time)
t = intv/20
navg = 20
rho = 1e-7

h = h[::intv]

val = np.diff(val[::intv])/1000*0.001
bal = np.diff(bal[::intv])/1000*0.0006
val = np.append([0],val)
bal = np.append([0],bal)
val2 = np.convolve(val,np.ones(navg)/navg,mode='full')[:-(navg-1)] 
bal2 = np.convolve(bal,np.ones(navg)/navg,mode='full')[:-(navg-1)]
h = h - h[0]; 
T = h.size
print(T)

h = h/abs(np.mean(h))
dl = cvx.Variable(T)
l = cvx.Variable(T)
v = cvx.Variable(T)
Val = cvx.Variable(T)
Bal = cvx.Variable(T)
a = cvx.Variable(1)
b = cvx.Variable(1)
turb = cvx.Variable(T)
const = 	[l[1:] == l[:-1] + dl[:-1] + Bal[:-1] + Val[:-1],
			v[1:] == v[:-1] + 1/klin/4*60*(l[:-1] - v[:-1]) + turb[:-1],
			h[1:] == h[:-1] + v[:-1]]
const.append(Bal == a*bal2)
const.append(Val == a*val2)

obj = cvx.Minimize(0.001*cvx.sum_squares(turb) + 100*cvx.sum_squares(cvx.diff(dl)))
prob = cvx.Problem(obj,const)
r = prob.solve()#solver = 'ECOS',verbose=True,reltol=1e-10,abstol=1e-10,max_iters=200)
print(r)
temps = df.atmo_temp_dp[::intv]
l = np.asarray(l.value.T)[0,:]
dl = np.asarray(dl.value.T)[0,:]
#plt.plot(temps, np.sign(l)*)dl)
#m, b = np.polyfit(temps, np.sign(l)*(np.asarray(l.value.T).T), 1)
#plt.plot(np.unique(temps), m*np.unique(temps)+b)
plt.xlabel('gradient (dK/dPa)')
plt.ylabel('dl/dt')
plt.show()
