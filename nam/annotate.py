import pygrib
import sys
import pandas as pd
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

tree = None

varlist = ['Vertical velocity','U component of wind','V component of wind','Geopotential Height','Albedo','Sensible heat net flux','Latent heat net flux','Downward short-wave radiation flux','Downward long-wave radiation flux', 'Upward short-wave radiation flux','Upward long-wave radiation flux','Temperature','Total Cloud Cover','Relative humidity','Absolute vorticity','Icing severity','Orography','Pressure reduced to MSL']
vv = ['zvel','uvel','vvel','geop_height','albedo','sensible','latent','downward_short','downward_long','upward_short','upward_long','atmo_temp','cloud_cover','humidity','vorticity','icing','orography','pressure_msl']
#varlist = ['Geopotential Height']
#vv = ['height']

def read_file(fname):
    print('opening',fname)
    global tree
    f = pygrib.open(fname) # open

    print(list(zip(varlist,vv)))
    assert len(varlist) == len(vv)

    d = f.select()
    #for x in d:
    #    print(x)
    #exit()
    grbs = []
    for x in varlist:
        if type(x) == int:
            grbs.append([d[x-1]])
        else:
            try: grbs.append(f.select(name=x))
            except:
                print('Not including',x)
                grbs.append(None)
            #print(grbs[-1])
    #exit()

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
        if grb is None:
            data.append(None)
            continue
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
    for i, grb in enumerate(grbs):
        if grb is None:
            out.append(None)
            out.append(None)
            continue
        levels = []
        data = []
        for level, x in grb:
            levels.append(level)
            data.append(x[idx])
        if np.all(np.array(levels) == 0):
            data = [data[0]] # hack for heat fluxes and cloud cover
        if len(data) == 1:
            out.append(data[0])
            out.append(0)
        else:
            print("num", i)
            print("levels", levels)
            print("data", data)
            spl = scipy.interpolate.InterpolatedUnivariateSpline(levels, data)
            out.append(spl(pres))
            out.append(spl.derivative(1)(pres))

    return out


if __name__ == '__main__':
    files = sorted(map(lambda x: int(x.split('.')[0]), os.listdir('data')))
    df = pd.read_hdf('/home/ssi/flight-data/%s.smol.h5' % sys.argv[1])
    index = df.index.astype(np.int64) // 10**9
    print("hmm", index[0])
    print(files)
    fst = np.where(np.array(files) >= index[0])[0][0] - 1
    cur = files[fst]
    nxt = files[fst+1]
    print("hmm", index[0]-cur, index[0]-nxt)
    print("starting gwith", fst)
    i = fst
    last_idx = -1
    last_data = None

    fields = []
    ar = {}
    for v in vv:
        ar[v] = np.zeros(len(df.index))
        ar[v+"_dp"] = np.zeros(len(df.index))
        fields.extend([v, v+"_dp"])
    #index += 7531200

    fmt = 'data/%d.grb2'
    cur_data = read_file(fmt % cur)

    print("Starting")

    t0 = index[0]
    for j in range(len(df.index.values)):
        t = index.values[j]
        print(round((t-t0)/3600.,2))
        if t >= nxt:
            print(t, nxt, "switching!")
            del cur_data
            cur_data = read_file(fmt % nxt)
            cur = nxt
            try:
                nxt = files[i+2]
            except:
                nxt = cur + 3600*40
                print("finishing...")
            i += 1
        assert (t-cur) < 10900
        assert (t-cur) >= 0
        lat = df.lat_gps.values[j]
        lon = df.long_gps.values[j]
        dist, idx = tree.query([lat, lon],1)
        pres = (df.raw_pressure_1.values[j] + df.raw_pressure_2.values[j] + \
                df.raw_pressure_3.values[j] + df.raw_pressure_4.values[j])/400.

        data = get_at_point(cur_data, idx, pres)

        for v, d in zip(fields, data):
            print("var",v,"lol",d)
            if d is None:
                d = ar[v][j-1]
            ar[v][j] = d

    df1 = df.assign(index=df.index, **ar)
    #for v in fields:
    #    df[v] = pd.Series(ar[v], index=df.index)

    df1.to_hdf('/home/ssi/flight-data/%s.atmo.h5' % sys.argv[1], 'df', complevel=5)
    exit()
    grbs = read_file('data/1512849600.grb2')
    print(get_at_point(grbs, 37.42, 122.16, 1000.))
