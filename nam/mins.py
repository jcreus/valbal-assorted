import pygrib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.spatial
import scipy.interpolate

from datetime import tzinfo, timedelta, datetime

class UTC(tzinfo):
    def utcoffset(self, dt):
        return timedelta(0)
    def tzname(self, dt):
        return "UTC"
    def dst(self, dt):
        return timedelta(0)
utc = UTC()

launch_time = datetime(2017, 12, 9, 9, 39, tzinfo=utc)
landing_time = datetime(2017, 12, 14, 11, 13, tzinfo=utc)

tree = None

varlist = ['Geometric vertical velocity','U component of wind','V component of wind','Geopotential Height','Albedo','Sensible heat net flux','Latent heat net flux','Downward short-wave radiation flux','Upward short-wave radiation flux','Upward long-wave radiation flux','Temperature',]
vv = ['zvel','uvel','vvel','geop_height','albedo','sensible','latent','downward','shortwave','longwave','atmo_temp']
#varlist = ['Geopotential Height']
#vv = ['height']
varlist = ['Temperature']
vv = ['atmo_temp']

def read_file(fname):
    global tree
    f = pygrib.open(fname) # open

    assert len(varlist) == len(vv)
    d = f.select()
    grbs = []
    for x in varlist:
        if type(x) == int:
            grbs.append([d[x-1]])
        else:
            grbs.append(f.select(name=x))

    if tree is None:
        lats, lons = grbs[0][0].latlons()

        lats = lats.flatten()
        lons = lons.flatten()

        grid = np.zeros(shape=(len(lats),2),dtype=np.float32)
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            grid[i,0] = lat
            grid[i,1] = lon

        tree = scipy.spatial.cKDTree(grid, copy_data=True)

    data = []
    for grb in grbs:
        cnt = max(1,len(list(filter(lambda x: x['typeOfLevel'] == "isobaricInhPa", grb))))
        gg = []
        for row in grb:
            if cnt > 1 and row['typeOfLevel'] != 'isobaricInhPa': continue
            gg.append((row.level, row.values.flatten()))
        data.append(gg)

    del grbs
    del f

    return data 

import time
def get_at_point(grbs, idx, pres):#lat, lon, pres):

    out = []
    for grb in grbs:
        levels = []
        data = []
        for level, x in grb:
            levels.append(level)
            data.append(x[idx])
        if len(data) == 1:
            out.append(data[0])
            out.append(0)
        else:
            spl = scipy.interpolate.InterpolatedUnivariateSpline(levels, data)
            out.append(spl(pres))
            out.append(spl.derivative(1)(pres))
            return spl.derivative(1)(pres)

    return out

def get_at_points(grbs, idx, pres):#lat, lon, pres):

    out = []
    for grb in grbs:
        levels = []
        data = []
        for level, x in grb:
            levels.append(level)
            data.append(x[idx])
        if len(data) == 1:
            out.append(data[0])
            out.append(0)
        else:
            #pres = np.linspace(min(levels),max(levels),100)
            spl = scipy.interpolate.InterpolatedUnivariateSpline(levels, data)
            out.append(spl(pres))
            out.append(spl.derivative(1)(pres))
            return spl.derivative(1)(pres)

    return out


if __name__ == '__main__':
    files = sorted(map(lambda x: int(x.split('.')[0]), os.listdir('data')))
    cur = files[0]
    nxt = files[1]
    i = 0
    last_idx = -1
    last_data = None

    df = pd.read_hdf('/home/joan/valbal/data/ssi63/smol.h5')
    fields = []

    index = df.index.astype(np.int64) // 10**9
    index += 7531200

    fmt = 'data/%d.grb2'
    cur_data = read_file(fmt % cur)

    print("Starting")

    t0 = index[0]
    mm = []
    nn = []
    oo = []
    bb = []
    aalt = []
    for j in range(len(df.index.values)):
        t = index.values[j]
        print(round((t-t0)/3600.,2))
        if t >= nxt:
            del cur_data
            cur_data = read_file(fmt % nxt)
            cur = nxt
            try:
                nxt = files[i+2]
            except:
                nxt = cur + 3600*10
                print("finishing...")
            i += 1
        assert (t-cur) < 10900
        assert (t-cur) >= 0
        lat = df.lat_gps.values[j]
        lon = df.long_gps.values[j]
        dist, idx = tree.query([lat, lon],1)
        pres = (df.raw_pressure_1.values[j] + df.raw_pressure_2.values[j] + \
                df.raw_pressure_3.values[j] + df.raw_pressure_4.values[j])/400.
        actual = get_at_point(cur_data, idx, pres)
        oo.append(actual)
        bb.append(df.ballast_time_total.values[j])
        aalt.append(df.altitude_barometer.values[j])

        pres = np.linspace(112, 186, 20)
        data = get_at_points(cur_data, idx, pres)#, df.altitude_barometer.values[j]>14000)
        mm.append(min(data))
        nn.append(max(data))

    plt.axhline(y=0, color='gray')
    plt.subplot(211)
    for k in range(0, 430000, 20000): plt.axvline(x=k, color='gray',alpha=0.3)
    plt.plot(mm, color='black')
    plt.plot(nn, color='black')
    plt.plot(oo, color='red')
    bb = np.array(bb)
    bb = bb/np.max(bb)*((np.max(mm)-np.min(mm))*1.25)*3
    plt.plot(bb, color='green')
    plt.subplot(212)
    for k in range(0, 430000, 20000): plt.axvline(x=k, color='gray',alpha=0.3)
    plt.plot(aalt)
    plt.show()
    #for v in fields:
    #    df[v] = pd.Series(ar[v], index=df.index)

    #df1.to_hdf('ssi63atmo.h5', 'df', complevel=5)
    exit()
    grbs = read_file('data/1512849600.grb2')
    print(get_at_point(grbs, 37.42, 122.16, 1000.))
