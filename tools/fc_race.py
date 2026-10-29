"""High-resolution race-day forecast for a race pack -> site/<race>/app/data/fc.json (read by the modeller's Forecast option).

Kona and 70.3 Hawaii use the National Weather Service relay only (2.5 km, forecaster-edited). This script covers the rest (races.json, 'fc'):
  hrrr            US mainland courses: NOAA HRRR 3 km (to +48 h from the 00/06/12/18 UTC runs), public domain, AWS Open Data. The NWS relay stays alongside.
  dwd             ICON-D2 where the course sits inside its domain, otherwise ICON-EU (Roth, Frankfurt, Berlin, Nice, 70.3 Nice, Wales, Weymouth, London).
  ecmwf           Taupō, 70.3 Taupō, Sydney, Tokyo.
  roth            DWD ICON-D2 (2.2 km, to +48 h) when race day is in range, otherwise DWD ICON-EU (6.5 km, to +120 h).
                  Deutscher Wetterdienst open data, CC BY 4.0 (commercial use allowed with attribution). https://opendata.dwd.de
  taupo, taupo703 ECMWF IFS open data (0.25 degree, to +240 h from the 00 and 12 UTC runs), CC BY 4.0. https://data.ecmwf.int
Output has the same shape as one member of ens.json: {date, run, model, made, h0, pts:[{ws,wd,wg,t,rh,p,cf,ghi}]} at the pack's
weather points, hourly 05-19 local race time. Wind as speed (m/s) and the direction it comes from; t in °C; rh %; p hPa; cf 0-1.

Local test:  RACE=roth RACE_DATE=2026-10-17 OUT=/tmp/fc.json python3 scripts/fc_race.py
"""
import os, sys, json, math, time, bz2, tempfile, datetime as dt, urllib.request, urllib.error

RACE = os.environ.get('RACE', 'roth')
RACES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'races.json')))
RACE_DATE = os.environ.get('RACE_DATE', '')
OUT = os.environ.get('OUT', 'site/%s/app/data/fc.json' % RACE)
def tz_hours(zone, date):
    from zoneinfo import ZoneInfo
    d = dt.datetime.fromisoformat((date or dt.date.today().isoformat()) + 'T12:00').replace(tzinfo=ZoneInfo(zone))
    return d.utcoffset().total_seconds() / 3600
CFG = dict(RACES[RACE]); CFG['src'] = CFG.get('fc'); TZ = tz_hours(CFG['tz'], RACE_DATE); CFG['tz'] = TZ; WXP = [tuple(p) for p in CFG['wxp']]
H0, H1 = 5, 19
UA = {'User-Agent': 'EnduranceModeller (ianwhiteperformance.com)'}
LAST = {}

def log(*a): print(*a, flush=True)
def get(url, headers=None, tries=3):
    h = dict(UA); h.update(headers or {})
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90) as r: return r.read()
        except urllib.error.HTTPError as e:
            LAST['e'] = 'HTTP %s %s' % (e.code, url[-90:])
            if e.code == 404: return None
        except Exception as e:
            LAST['e'] = '%s %s' % (type(e).__name__, url[-90:])
        time.sleep(3 * (k + 1))
    return None

def sample(buf):
    """GRIB2 bytes -> {shortName: values at the weather points}: inverse-distance blend of the four nearest grid points (works on any grid)"""
    import eccodes
    out = {}
    with tempfile.NamedTemporaryFile(suffix='.grb2') as f:
        f.write(buf); f.flush()
        with open(f.name, 'rb') as fh:
            while True:
                h = eccodes.codes_grib_new_from_file(fh)
                if h is None: break
                try:
                    sn = eccodes.codes_get(h, 'shortName'); res = []
                    for lat, lon in WXP:
                        nn = eccodes.codes_grib_find_nearest(h, lat, lon, npoints=4)
                        w = [1 / max(0.05, n['distance']) ** 2 for n in nn]
                        res.append(sum(wi * n['value'] for wi, n in zip(w, nn)) / sum(w))
                        if not LAST.get('nn'): LAST['nn'] = '%s nearest grid point %.3f,%.3f for %.3f,%.3f (%.1f km)' % (sn, nn[0]['lat'], nn[0]['lon'], lat, lon, nn[0]['distance']); log(LAST['nn'])
                    out[sn] = res
                finally:
                    eccodes.codes_release(h)
    return out

def window_utc(race):
    a = dt.datetime.combine(race, dt.time(H0)) - dt.timedelta(hours=TZ)
    b = dt.datetime.combine(race, dt.time(H1)) - dt.timedelta(hours=TZ)
    return a, b

# ---------------- DWD ICON-D2 / ICON-EU ----------------
DWD = 'https://opendata.dwd.de/weather/nwp'
DVARS = ['t_2m', 'relhum_2m', 'u_10m', 'v_10m', 'vmax_10m', 'clct', 'ps']
def dwd_url(model, run, lead, var):
    if model == 'icon-d2':
        return '%s/icon-d2/grib/%02d/%s/icon-d2_germany_regular-lat-lon_single-level_%s_%03d_2d_%s.grib2.bz2' % (DWD, run.hour, var, run.strftime('%Y%m%d%H'), lead, var)
    return '%s/icon-eu/grib/%02d/%s/icon-eu_europe_regular-lat-lon_single-level_%s_%03d_%s.grib2.bz2' % (DWD, run.hour, var, run.strftime('%Y%m%d%H'), lead, var.upper())
def dwd(race, now):
    a, b = window_utc(race)
    for model, step, maxl, cyc in (('icon-d2', 3, 48, 3), ('icon-eu', 6, 120, 6)):
        if model == 'icon-d2' and not CFG.get('d2'): continue   # ICON-D2 covers central Europe only (Wales is west of it)
        for back in range(0, 24, cyc):
            run = (now - dt.timedelta(hours=back + 2)).replace(minute=0, second=0, microsecond=0); run = run.replace(hour=(run.hour // cyc) * cyc)
            l0 = int((a - run).total_seconds() // 3600) - 1; l1 = int((b - run).total_seconds() // 3600) + 1
            if l0 < 0 or l1 > maxl: continue
            if model == 'icon-eu' and l1 > 78:   # 3-hourly beyond 78 h
                leads = sorted(set([l for l in range(max(0, l0), 79)] + [l - l % 3 for l in range(79, l1 + 3)]))
            else:
                leads = list(range(max(0, l0), l1 + 1))
            if get(dwd_url(model, run, leads[-1], 't_2m'), headers={'Range': 'bytes=0-10'}) is None: continue
            log(model, 'run', run, 'leads', leads[0], 'to', leads[-1])
            per = []
            from concurrent.futures import ThreadPoolExecutor
            def one(l):
                d = {}
                for v in DVARS:
                    raw = get(dwd_url(model, run, l, v))
                    if raw is None: continue
                    s = sample(bz2.decompress(raw))
                    if s: d[v] = list(s.values())[0]
                return d
            with ThreadPoolExecutor(8) as ex: per = list(ex.map(one, leads))
            ok = sum(1 for d in per if all(k in d for k in ('t_2m', 'relhum_2m', 'u_10m', 'v_10m')))
            log('time steps complete', ok, 'of', len(per))
            if ok < len(per) * 0.8: continue
            res = {'icon-d2': 'DWD ICON-D2 2.2 km', 'icon-eu': 'DWD ICON-EU 6.5 km'}[model]
            conv = lambda d: dict(u=d.get('u_10m'), v=d.get('v_10m'), t=[x - 273.15 for x in d['t_2m']] if 't_2m' in d else None, rh=d.get('relhum_2m'),
                                  g=d.get('vmax_10m'), p=[x / 100 for x in d['ps']] if 'ps' in d else None, cf=[x / 100 for x in d['clct']] if 'clct' in d else None)
            return run, [run + dt.timedelta(hours=l) for l in leads], [conv(d) for d in per], res
    return None

# ---------------- ECMWF IFS open data ----------------
ECM = 'https://data.ecmwf.int/forecasts'
EPAR = {'10u': 'u', '10v': 'v', '2t': 't', '2d': 'td', 'sp': 'p', 'tcc': 'cf', '10fg': 'g'}
def ecmwf(race, now):
    a, b = window_utc(race)
    for back in range(0, 48, 12):
        run = (now - dt.timedelta(hours=back + 8)).replace(minute=0, second=0, microsecond=0); run = run.replace(hour=(run.hour // 12) * 12)
        l0 = int((a - run).total_seconds() // 3600); l1 = int((b - run).total_seconds() // 3600)
        if l0 < 0 or l1 > 240: continue
        st = 3 if l1 <= 144 else 6
        leads = list(range(max(0, l0 - l0 % st), l1 + st, st))
        base = '%s/%s/%02dz/ifs/0p25/oper/%s%02d0000-%dh-oper-fc.grib2' % (ECM, run.strftime('%Y%m%d'), run.hour, run.strftime('%Y%m%d'), run.hour, leads[-1])
        if get(base[:-6] + '.index') is None: continue
        log('ECMWF IFS run', run, 'leads', leads)
        def one(l):
            url = '%s/%s/%02dz/ifs/0p25/oper/%s%02d0000-%dh-oper-fc' % (ECM, run.strftime('%Y%m%d'), run.hour, run.strftime('%Y%m%d'), run.hour, l)
            idx = get(url + '.index')
            if not idx: return {}
            d = {}
            for line in idx.decode().splitlines():
                try: j = json.loads(line)
                except Exception: continue
                if j.get('levtype') != 'sfc' or j.get('param') not in EPAR: continue
                part = get(url + '.grib2', headers={'Range': 'bytes=%d-%d' % (j['_offset'], j['_offset'] + j['_length'] - 1)})
                if part:
                    s = sample(part)
                    if s: d[EPAR[j['param']]] = list(s.values())[0]
            return d
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex: per = list(ex.map(one, leads))
        ok = sum(1 for d in per if all(k in d for k in ('u', 'v', 't', 'td')))
        log('time steps complete', ok, 'of', len(per), LAST.get('e', ''))
        if ok < len(per) * 0.8: continue
        def conv(d):
            if 't' not in d or 'td' not in d: return dict(d, t=None, rh=None)
            t = [x - 273.15 for x in d['t']]; td = [x - 273.15 for x in d['td']]
            rh = [100 * math.exp(17.625 * q / (243.04 + q)) / math.exp(17.625 * T / (243.04 + T)) for T, q in zip(t, td)]   # Magnus
            return dict(u=d['u'], v=d['v'], t=t, rh=rh, g=d.get('g'), p=[x / 100 for x in d['p']] if 'p' in d else None, cf=d.get('cf'))
        return run, [run + dt.timedelta(hours=l) for l in leads], [conv(d) for d in per], 'ECMWF IFS 0.25°'
    return None

# ---------------- NOAA HRRR (3 km, US mainland; public domain), AWS Open Data ----------------
HRRR = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
HWANT = {('UGRD', '10 m above ground'): 'u', ('VGRD', '10 m above ground'): 'v', ('TMP', '2 m above ground'): 't', ('RH', '2 m above ground'): 'rh',
         ('GUST', 'surface'): 'g', ('PRES', 'surface'): 'p', ('TCDC', 'entire atmosphere'): 'cf', ('DSWRF', 'surface'): 'sw'}
def hrrr(race, now):
    """HRRR runs every hour to +18 h and to +48 h at 00, 06, 12 and 18 UTC. Byte ranges from the .idx inventory, wanted fields only."""
    a, b = window_utc(race)
    for back in range(1, 25):
        run = (now - dt.timedelta(hours=back)).replace(minute=0, second=0, microsecond=0)
        maxl = 48 if run.hour % 6 == 0 else 18
        l0 = int((a - run).total_seconds() // 3600); l1 = int((b - run).total_seconds() // 3600)
        if l0 < 0 or l1 > maxl: continue
        leads = list(range(l0, l1 + 1))
        url = lambda l: '%s/hrrr.%s/conus/hrrr.t%02dz.wrfsfcf%02d.grib2' % (HRRR, run.strftime('%Y%m%d'), run.hour, l)
        if get(url(leads[-1]) + '.idx') is None: continue
        log('HRRR run', run, 'leads', leads[0], 'to', leads[-1])
        def one(l):
            idx = get(url(l) + '.idx')
            if not idx: return {}
            lines = idx.decode('utf-8', 'replace').splitlines(); d = {}
            for i, line in enumerate(lines):
                p = line.split(':')
                if len(p) < 6 or (p[3], p[4]) not in HWANT: continue
                start = int(p[1]); end = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else ''
                part = get(url(l), headers={'Range': 'bytes=%d-%s' % (start, end)})
                if part:
                    s = sample(part)
                    if s: d[HWANT[(p[3], p[4])]] = list(s.values())[0]
            return d
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(6) as ex: per = list(ex.map(one, leads))
        ok = sum(1 for d in per if all(k in d for k in ('u', 'v', 't', 'rh')))
        log('time steps complete', ok, 'of', len(per), LAST.get('e', ''))
        if ok < len(per) * 0.8: continue
        conv = lambda d: dict(u=d.get('u'), v=d.get('v'), t=[x - 273.15 for x in d['t']] if 't' in d else None, rh=d.get('rh'), g=d.get('g'),
                              p=[x / 100 for x in d['p']] if 'p' in d else None, cf=[x / 100 for x in d['cf']] if 'cf' in d else None, sw=d.get('sw'))
        return run, [run + dt.timedelta(hours=l) for l in leads], [conv(d) for d in per], 'NOAA HRRR 3 km'
    return None

def hourly(times, per, race):
    def at(k, pt, t):
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                w = (t - times[i]).total_seconds() / max(1, (times[i + 1] - times[i]).total_seconds())
                va, vb = per[i].get(k), per[i + 1].get(k)
                if va is None or vb is None: return None
                return va[pt] + (vb[pt] - va[pt]) * w
        return None
    pts = []
    for pt in range(len(WXP)):
        P = {k: [] for k in ('ws', 'wd', 'wg', 't', 'rh', 'p', 'cf')}
        for hr in range(H0, H1 + 1):
            t = dt.datetime.combine(race, dt.time(hr)) - dt.timedelta(hours=TZ)
            u, v = at('u', pt, t), at('v', pt, t); ws = None if u is None or v is None else math.hypot(u, v)
            P['ws'].append(None if ws is None else round(ws, 2))
            P['wd'].append(None if ws is None else round((math.degrees(math.atan2(-u, -v)) + 360) % 360))
            g = at('g', pt, t); P['wg'].append(round(max(g, ws or 0), 2) if g is not None else (round(ws * 1.4, 2) if ws is not None else None))
            T = at('t', pt, t); P['t'].append(None if T is None else round(T, 2))
            rh = at('rh', pt, t); P['rh'].append(None if rh is None else round(max(0, min(100, rh)), 1))
            p = at('p', pt, t); P['p'].append(None if p is None else round(p, 1))
            cf = at('cf', pt, t); P['cf'].append(None if cf is None else round(max(0, min(1, cf)), 2))
        P['ghi'] = [None] * len(P['ws'])
        for j, hr in enumerate(range(H0, H1 + 1)):   # HRRR carries downward shortwave; the app estimates sun from cloud otherwise
            sw = at('sw', pt, dt.datetime.combine(race, dt.time(hr)) - dt.timedelta(hours=TZ))
            if sw is not None: P['ghi'][j] = round(max(0, sw))
        pts.append(P)
    return pts

def main():
    if not RACE_DATE: log('RACE_DATE not set; nothing to do.'); return 0
    race = dt.date.fromisoformat(RACE_DATE)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    if not CFG.get('src'): log('No high-resolution forecast for this course (the NWS relay serves it).'); return 0
    r = {'dwd': dwd, 'ecmwf': ecmwf, 'hrrr': hrrr}[CFG['src']](race, now)
    if not r: log('No high-resolution run covers race day yet.', LAST.get('e', '')); return 0
    run, times, per, model = r
    pts = hourly(times, per, race)
    if any(v is None for p in pts for v in p['t'] + p['ws']): log('Incomplete hours in the forecast; keeping the previous file.'); return 1
    try:
        old = json.load(open(OUT))
        if old.get('date') == RACE_DATE and old.get('run') == run.strftime('%Y-%m-%d %H:00 UTC') and old.get('model') == model: log('Already published.'); return 0
    except Exception: pass
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    json.dump({'date': RACE_DATE, 'run': run.strftime('%Y-%m-%d %H:00 UTC'), 'model': model, 'made': now.strftime('%Y-%m-%d %H:%M UTC'), 'h0': H0, 'pts': pts},
              open(OUT, 'w'), separators=(',', ':'))
    log('wrote', OUT, model, run)
    return 0

if __name__ == '__main__': sys.exit(main())
