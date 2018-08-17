import pandas as pd
import numpy as np
import sys

df = pd.read_hdf(sys.argv[1])#, stop=3600)
#df['raw_pressure'] = (1/4.)*(df.raw_pressure_1+df.raw_pressure_2+df.raw_pressure_3+df.raw_pressure_4)
#var = ['overpressure','overpressure_vref','altitude_barometer','raw_pressure','lat_gps','long_gps','speed_gps','temp_int','voltage_primary','ballast_time_total','valve_time_total','las_v', 'valve_state', 'ballast_state']
#var = ['raw_pressure_1', 'raw_pressure_2', 'raw_pressure_3', 'raw_pressure_4', 'altitude_barometer']
var = ['altitude_barometer', 'altitude_gps', 'ascent_rate', 'lat_gps', 'long_gps', 'temp_int','ballast_time_total', 'valve_time_total', 'solar_elevation', 'raw_pressure_1', 'raw_pressure_2', 'raw_pressure_3', 'raw_pressure_4']
if '63' not in sys.argv[1]:
    var.append('las_v')
    var.append('las_fused_v')

print('initial time', df.index[0])

"""d = {}
for v in var:
    d[v] = np.mean"""
log = df[var].resample('60s').mean()# agg(d)
#log['ballast_state'] += 0
#log['valve_state'] += 0
#log.to_csv('logan.csv')
print(dir(log), len(log))
print(log)
print(list(log))

log.to_hdf(sys.argv[1].replace('.h5','.smol.h5'), 'df')
print("done")
