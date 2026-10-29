"""NOAA GEFS ensemble for a race pack -> site/<race>/app/data/ens.json (RACE=kona|taupo|roth) (read by the modeller's Ensemble option).

Runs on GitHub Actions twice a day (see .github/workflows/kona-ensemble.yml). Does nothing unless race day is 0 to 16 days away.
Data: NOAA Global Ensemble Forecast System, 0.5 degree, control + 30 members (US Government work, public domain),
fetched with the NOMADS grib filter for a small box around the course only.

Local test:  RACE=taupo RACE_DATE=2027-03-06 python3 scripts/gefs_race.py
"""
import os, sys, json, math, time, datetime as dt, urllib.request, urllib.error, tempfile

RACE = os.environ.get('RACE', 'kona')
# every course: IANA time zone, the pack's weather points and a grib filter box (lon 0-360), from races.json (generated from the packs)
RACES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'races.json')))
RACE_DATE = os.environ.get('RACE_DATE', '')
OUT = os.environ.get('OUT', 'site/%s/app/data/ens.json' % RACE)
def tz_hours(zone, date):
    from zoneinfo import ZoneInfo
    d = dt.datetime.fromisoformat((date or dt.date.today().isoformat()) + 'T12:00').replace(tzinfo=ZoneInfo(zone))
    return d.utcoffset().total_seconds() / 3600
TZ = tz_hours(RACES[RACE]['tz'], RACE_DATE)   # hours to add to UTC for the race clock on race day (summer time included)
BASE = os.environ.get('NOMADS', 'https://nomads.ncep.noaa.gov')
AWS = os.environ.get('GEFS_AWS', 'https://noaa-gefs-pds.s3.amazonaws.com')   # NOAA Open Data on AWS: same files, no rate limit
H0, H1 = 5, 19                         # local hours the modeller uses
WXP = [tuple(p) for p in RACES[RACE]['wxp']]
BOX = RACES[RACE]['box']
MEMBERS = ['gec00'] + ['gep%02d' % i for i in range(1, 31)]
PAUSE = float(os.environ.get('PAUSE', '1.0'))   # per worker; 3 workers keeps us under NOMADS' ~120 requests a minute
WORKERS = int(os.environ.get('WORKERS', '6'))

def log(*a): print(*a, flush=True)

LAST_ERR = {}
def get(url, tries=3, headers=None):
    h = {'User-Agent': 'RaceRecon (ianwhiteperformance.com)'}
    if headers: h.update(headers)
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=45) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            LAST_ERR['e'] = 'HTTP %s from %s' % (e.code, url.split('?')[0][:90])
            if e.code == 404: return None
            time.sleep(3 * (k + 1))
        except Exception as e:
            LAST_ERR['e'] = '%s from %s' % (type(e).__name__, url.split('?')[0][:90])
            time.sleep(3 * (k + 1))
    return None

# fields wanted from each file, as named in the .idx inventory (variable, level prefix)
WANT = [('UGRD', '10 m above ground'), ('VGRD', '10 m above ground'), ('TMP', '2 m above ground'), ('RH', '2 m above ground'),
        ('PRES', 'surface'), ('TCDC', 'entire atmosphere'), ('GUST', 'surface')]

def aws_path(cycle, member, lead):
    return '%s/gefs.%s/%02d/atmos/pgrb2ap5/%s.t%02dz.pgrb2a.0p50.f%03d' % (AWS, cycle.strftime('%Y%m%d'), cycle.hour, member, cycle.hour, lead)

def fetch_aws(cycle, member, lead):
    """Read the file's .idx inventory, then download only the wanted fields with HTTP byte ranges."""
    url = aws_path(cycle, member, lead)
    idx = get(url + '.idx')
    if not idx: return None
    rows = []
    for line in idx.decode('utf-8', 'replace').splitlines():
        f = line.split(':')
        if len(f) >= 5 and f[1].isdigit(): rows.append((int(f[1]), f[3], f[4]))
    out = b''
    for i, (off, var, lev) in enumerate(rows):
        if not any(var == v and lev.startswith(l) for v, l in WANT): continue
        end = rows[i + 1][0] - 1 if i + 1 < len(rows) else ''
        part = get(url, headers={'Range': 'bytes=%d-%s' % (off, end)})
        if part: out += part
    return out or None

def exists(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'KonaRaceModeller'}), timeout=30) as r:
            return r.status == 200
    except Exception:
        return False

def valid_hours(race):
    """UTC times covering 05:00-19:00 Hawaii time on race day, every 3 hours with one either side."""
    start = dt.datetime.combine(race, dt.time(H0)) - dt.timedelta(hours=TZ)   # local -> UTC
    end = dt.datetime.combine(race, dt.time(H1)) - dt.timedelta(hours=TZ)
    t = start.replace(hour=(start.hour // 3) * 3)
    out = []
    while t <= end + dt.timedelta(hours=3):
        out.append(t); t += dt.timedelta(hours=3)
    return out

def pick_cycle(race, now):
    for back in range(0, 48, 6):
        c = (now - dt.timedelta(hours=back)).replace(minute=0, second=0, microsecond=0)
        c = c.replace(hour=(c.hour // 6) * 6)
        leads = [int((v - c).total_seconds() // 3600) for v in valid_hours(race)]
        if min(leads) < 0 or max(leads) > 384: continue
        leads = [l if l <= 240 else (l // 6) * 6 for l in leads]
        last = max(leads)
        path = '%s/pub/data/nccf/com/gens/prod/gefs.%s/%02d/atmos/pgrb2ap5/gep30.t%02dz.pgrb2a.0p50.f%03d' % (BASE, c.strftime('%Y%m%d'), c.hour, c.hour, last)
        if exists(aws_path(c, 'gep30', last) + '.idx') or exists(path): return c, sorted(set(leads))
    return None, None

def fetch(cycle, member, lead):
    b = fetch_aws(cycle, member, lead)
    if b: return b
    return fetch_nomads(cycle, member, lead)

def fetch_nomads(cycle, member, lead):
    q = dict(file='%s.t%02dz.pgrb2a.0p50.f%03d' % (member, cycle.hour, lead),
             lev_10_m_above_ground='on', lev_2_m_above_ground='on', lev_surface='on', lev_entire_atmosphere='on',
             var_UGRD='on', var_VGRD='on', var_TMP='on', var_RH='on', var_PRES='on', var_TCDC='on', var_GUST='on',
             subregion='', dir='/gefs.%s/%02d/atmos/pgrb2ap5' % (cycle.strftime('%Y%m%d'), cycle.hour), **BOX)
    url = BASE + '/cgi-bin/filter_gefs_atmos_0p50a.pl?' + '&'.join('%s=%s' % (k, v) for k, v in q.items())
    time.sleep(PAUSE * 3)   # NOMADS only, as a fallback: stay well under its rate limit
    return get(url)

def decode(buf):
    """GRIB2 bytes -> {name: [values at the five course points]} using ecCodes, bilinear on the regular grid."""
    import eccodes
    out = {}
    with tempfile.NamedTemporaryFile(suffix='.grb2') as f:
        f.write(buf); f.flush()
        with open(f.name, 'rb') as fh:
            while True:
                h = eccodes.codes_grib_new_from_file(fh)
                if h is None: break
                try:
                    # identify fields by WMO code (discipline, category, number, level type, level): robust to ecCodes naming
                    g = lambda k: eccodes.codes_get(h, k, int)
                    lvt = g('typeOfFirstFixedSurface'); lv = g('level') if lvt == 103 else 0
                    key = {(0, 2, 2, 103, 10): 'u', (0, 2, 3, 103, 10): 'v', (0, 0, 0, 103, 2): 't', (0, 1, 1, 103, 2): 'rh',
                           (0, 3, 0, 1, 0): 'p', (0, 2, 22, 1, 0): 'g', (0, 6, 1, 10, 0): 'cf'}.get((g('discipline'), g('parameterCategory'), g('parameterNumber'), lvt, lv))
                    if not key: continue
                    ni, nj = eccodes.codes_get(h, 'Ni'), eccodes.codes_get(h, 'Nj')
                    la1, lo1 = eccodes.codes_get(h, 'latitudeOfFirstGridPointInDegrees'), eccodes.codes_get(h, 'longitudeOfFirstGridPointInDegrees')
                    di, dj = eccodes.codes_get(h, 'iDirectionIncrementInDegrees'), eccodes.codes_get(h, 'jDirectionIncrementInDegrees')
                    jpos = eccodes.codes_get(h, 'jScansPositively'); vals = eccodes.codes_get_values(h)
                    res = []
                    for lat, lon in WXP:
                        x = ((lon % 360) - lo1) / di; y = ((lat - la1) / dj) if jpos else ((la1 - lat) / dj)
                        i0 = max(0, min(ni - 2, int(math.floor(x)))); j0 = max(0, min(nj - 2, int(math.floor(y)))); fx = min(1, max(0, x - i0)); fy = min(1, max(0, y - j0))
                        V = lambda i, j: vals[j * ni + i]
                        res.append((1 - fx) * (1 - fy) * V(i0, j0) + fx * (1 - fy) * V(i0 + 1, j0) + (1 - fx) * fy * V(i0, j0 + 1) + fx * fy * V(i0 + 1, j0 + 1))
                    out[key] = res
                finally:
                    eccodes.codes_release(h)
    return out

def hourly(cycle, leads, per_lead, race):
    """3- or 6-hourly values -> hourly 05-19 local, wind interpolated as vectors."""
    times = [cycle + dt.timedelta(hours=l) for l in leads]
    def at(k, pt, t):
        for a in range(len(times) - 1):
            if times[a] <= t <= times[a + 1]:
                w = (t - times[a]).total_seconds() / max(1, (times[a + 1] - times[a]).total_seconds())
                va, vb = per_lead[a].get(k), per_lead[a + 1].get(k)
                if va is None or vb is None: return None
                return va[pt] + (vb[pt] - va[pt]) * w
        return None
    pts = []
    for pt in range(len(WXP)):
        P = {k: [] for k in ('ws', 'wd', 'wg', 't', 'rh', 'p', 'cf')}
        for hr in range(H0, H1 + 1):
            t = dt.datetime.combine(race, dt.time(hr)) + dt.timedelta(hours=-TZ)
            u, v = at('u', pt, t), at('v', pt, t)
            ws = None if u is None or v is None else math.hypot(u, v)
            P['ws'].append(None if ws is None else round(ws, 2))
            P['wd'].append(None if ws is None else round((math.degrees(math.atan2(-u, -v)) + 360) % 360))   # direction the wind comes from
            g = at('g', pt, t); P['wg'].append(round(g, 2) if g is not None else (round(ws * 1.4, 2) if ws is not None else None))
            T = at('t', pt, t); P['t'].append(None if T is None else round(T - 273.15, 2))
            rh = at('rh', pt, t); P['rh'].append(None if rh is None else round(rh, 1))
            p = at('p', pt, t); P['p'].append(None if p is None else round(p / 100, 1))
            cf = at('cf', pt, t); P['cf'].append(None if cf is None else round(max(0, min(1, cf / 100)), 2))
        P['ghi'] = [None] * len(P['ws'])
        pts.append(P)
    return pts

def main():
    if not RACE_DATE:
        log('RACE_DATE not set; nothing to do.'); return 0
    race = dt.date.fromisoformat(RACE_DATE)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    days = (dt.datetime.combine(race, dt.time(7)) + dt.timedelta(hours=-TZ) - now).total_seconds() / 86400
    if days < -1 or days > 16.5:
        log('Race day is %.1f days away; the ensemble only covers the last 16 days. Nothing to do.' % days); return 0
    cycle, leads = pick_cycle(race, now)
    if not cycle:
        log('No complete GEFS run covers race day yet.'); return 0
    log('GEFS run %s, leads %s' % (cycle.strftime('%Y-%m-%d %HZ'), leads))
    try:   # the site already has this run: nothing new, so no commit and no redeploy
        old = json.load(open(OUT))
        if old.get('date') == RACE_DATE and old.get('run') == cycle.strftime('%Y-%m-%d %H:00 UTC') and len(old.get('members') or []) >= 10:
            log('The site already has the %s run; nothing to do.' % old['run']); return 0
    except Exception:
        pass
    from concurrent.futures import ThreadPoolExecutor
    def one(m):
        per = []
        for l in leads:
            b = fetch(cycle, m, l)
            per.append(decode(b) if b else {})
        ok = all('u' in x and 'v' in x and 't' in x and 'rh' in x for x in per)
        why = '' if ok else ' (fields found per time step: %s%s)' % ([sorted(x) for x in per][:3], '; last error: ' + LAST_ERR['e'] if LAST_ERR.get('e') else '')
        log('member %s %s%s' % (m, 'ok' if ok else 'incomplete, skipped', why))
        return {'name': m, 'h0': H0, 'pts': hourly(cycle, leads, per, race)} if ok else None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        members = [x for x in ex.map(one, MEMBERS) if x]
    log('fetched %d members in %.1f min' % (len(members), (time.time() - t0) / 60))
    if len(members) < 10:
        log('Only %d complete members; keeping the previous file.' % len(members)); return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({'date': RACE_DATE, 'run': cycle.strftime('%Y-%m-%d %H:00 UTC'), 'model': 'NOAA GEFS 0.5°',
               'made': now.strftime('%Y-%m-%d %H:%M UTC'), 'members': members}, open(OUT, 'w'), separators=(',', ':'))
    log('wrote %s with %d members' % (OUT, len(members)))
    return 0

if __name__ == '__main__':
    sys.exit(main())
