import tarfile
import os
import sys
import tempfile
import parse_nam218 as parser
import datetime
from datetime import tzinfo, timedelta, datetime

# See PYthon docs
class UTC(tzinfo):
    def utcoffset(self, dt):
        return timedelta(0)
    def tzname(self, dt):
        return "UTC"
    def dst(self, dt):
        return timedelta(0)
utc = UTC()


#d = '/media/joan/My Passport1/balloondata/www1.ncdc.noaa.gov/pub/has/model/HAS010869317' if len(sys.argv) == 1 else sys.argv[1]
d = '/home/joan/newsim/raw/nomads.ncdc.noaa.gov/data/namanl/201612'

#f = []
f = [x for x in os.listdir(d) if x.endswith('.grb') and not "006." in x]

def pname(x):
    return map(int,[x[11:15],x[15:17],x[17:19],x[20:22], x[26:28]])

def pname2(x):
    return map(int,[x[8:12],x[12:14],x[14:16], x[17:19], x[22:25]])

def kk(a):
    v = pname(a)
    b = datetime(v[0],v[1],v[2],v[3],tzinfo=utc)
    return (b+timedelta(hours=v[4]), b+timedelta(hours=v[4]*1.01))

f = sorted(f, key=lambda a: kk(a)[1])
for i, fi in enumerate(f):
    secs = int((kk(fi)[0]-datetime(1970,1,1,tzinfo=utc)).total_seconds())
    if os.path.exists('data48/%d.longwave.bin' % secs):
        print "skipping", secs
        continue
    parser.dofile(d+'/'+fi, 'data48/%d' % secs)
    continue
    nxt = pname(f[i+1]) if i+1 < len(f) else None
    if nxt:
        nxt = datetime(nxt[0], nxt[1], nxt[2], nxt[3], tzinfo=utc)-timedelta(seconds=0.1)
        #nxt = (nxt[0]-2016)*365+nxt[1]*30+nxt[2]+nxt[3]/24.-0.1/24
    t = tarfile.open(os.path.join(d, fi))
    for tt in t.getnames():
        v = pname2(tt)
        v = datetime(v[0], v[1], v[2], v[3], tzinfo=utc)+timedelta(hours=v[4])
        #v = (v[0]-2016)*365+v[1]*30+v[2]+(v[3]+v[4])/24.
        if nxt and v < nxt:
            print "doing",v
            """
            tfile = tempfile.NamedTemporaryFile()
            secs = int((v-datetime(1970,1,1,tzinfo=utc)).total_seconds())
            fake = t.extractfile(tt)
            ss = fake.read()
            tfile.write(ss)
            parser.dofile(tfile.name, 'data/%d' % secs)
            tfile.close()"""
    #print t.getnames()
    #print pname(fi), map(pname2,t.getnames())
