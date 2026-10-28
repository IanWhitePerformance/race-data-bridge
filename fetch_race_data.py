#!/usr/bin/env python3
"""Fetch the open 3D data for one race course and write it under race-data/<id>/.

Runs on a GitHub Actions runner (full internet). Sources, all open licences:
  terrain   AWS Terrain Tiles (Terrarium PNG, Mapzen/Tilezen; SRTM 30 m and better where available), z11 over the box, z13 along the course
            https://registry.opendata.aws/terrain-tiles/
  osm       OpenStreetMap via the Overpass API (ODbL): buildings, roads, water, woods, land use, single trees near the course
  nz        LINZ aerial imagery and 1 m LiDAR DEM from the AWS Open Data buckets nz-imagery / nz-elevation (CC BY 4.0), no key
  bayern    Bavarian DOP20 orthophotos (WMS) and DGM1 1 m terrain (OpenData, CC BY 4.0) from geodaten.bayern.de
  profile   Open-Meteo elevation along the course line (Copernicus 90 m), as a check on the pack profile

Every source is optional: a failure is logged in race-data/<id>/report.json and the rest carries on.
Usage: python3 fetch_race_data.py race-data/courses/taupo.json
"""
import sys,os,json,math,time,io,re,threading,traceback
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
from PIL import Image
import rasterio,rasterio.windows,rasterio.transform
from rasterio.transform import from_origin,from_bounds
from rasterio.warp import reproject,Resampling,transform_bounds
from rasterio.crs import CRS

UA={'User-Agent':'race-modeller-data-bridge/1.0 (open data fetch for a race planning tool)'}
spec=json.load(open(sys.argv[1]));pid=spec['id'];OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1]))),'data',pid)
os.makedirs(OUT,exist_ok=True)
REPORT={'id':pid,'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sources':{}}
LAT=np.array([p[0] for p in spec['line']]);LON=np.array([p[1] for p in spec['line']])
lat0,lon0=spec['lat0'],spec['lon0'];kx=111.32*math.cos(math.radians(lat0));kz=110.574
M=spec.get('margin_km',3.0);S,N=LAT.min()-M/kz,LAT.max()+M/kz;W,E=LON.min()-M/kx,LON.max()+M/kx
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def done(src,**kw):REPORT['sources'][src]=dict(status='ok',**kw);log(src,'ok',kw)
def failed(src,e):REPORT['sources'][src]=dict(status='failed',error=str(e)[:400]);log(src,'FAILED',e)
def save_report():json.dump(REPORT,open(os.path.join(OUT,'report.json'),'w'),indent=1)
def dist_km(lat,lon):
    """km from a point to the course line (nearest vertex, fine at 150 m spacing)"""
    return float(np.min(np.hypot((LAT-lat)*kz,(LON-lon)*kx)))
def get(url,**kw):
    for k in range(4):
        try:
            r=requests.get(url,headers=UA,timeout=kw.pop('timeout',120),**kw)
            if r.status_code==200:return r
            if r.status_code in(429,503,504,502):time.sleep(5*(k+1));continue
            r.raise_for_status()
        except requests.RequestException as e:
            if k==3:raise
            time.sleep(3*(k+1))
    raise RuntimeError('gave up '+url)

# ---------------- web mercator tiles ----------------
R=6378137.0;ORG=math.pi*R
def tile_xy(lat,lon,z):
    n=2**z;x=(lon+180)/360*n;y=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n;return x,y
def tile_bounds_3857(x,y,z):
    res=2*ORG/(256*2**z)*256;return (-ORG+x*res,ORG-(y+1)*res,-ORG+(x+1)*res,ORG-y*res)
def tile_centre(x,y,z):
    n=2**z;lon=(x+0.5)/n*360-180;lat=math.degrees(math.atan(math.sinh(math.pi*(1-2*(y+0.5)/n))));return lat,lon
def tiles_for(z,corridor_km=None):
    x0,y1=tile_xy(S,W,z);x1,y0=tile_xy(N,E,z);out=[]
    for x in range(int(x0),int(x1)+1):
        for y in range(int(y0),int(y1)+1):
            if corridor_km is not None:
                la,lo=tile_centre(x,y,z);half=2*ORG/(256*2**z)*256/2/1000*math.cos(math.radians(la))
                if dist_km(la,lo)>corridor_km+half*1.5:continue
            out.append((x,y))
    return out
def mosaic(z,tiles,fetch,nodata,bands=1,dtype='int16'):
    xs=[t[0] for t in tiles];ys=[t[1] for t in tiles];X0,X1,Y0,Y1=min(xs),max(xs),min(ys),max(ys)
    Wp,Hp=(X1-X0+1)*256,(Y1-Y0+1)*256
    arr=np.full((bands,Hp,Wp),nodata,dtype);got=0;lock=threading.Lock()
    def one(t):
        nonlocal got
        try:
            a=fetch(t[0],t[1],z)
            if a is None:return
            with lock:
                arr[:,(t[1]-Y0)*256:(t[1]-Y0+1)*256,(t[0]-X0)*256:(t[0]-X0+1)*256]=a;got+=1
        except Exception as e:log('tile',z,t,'failed',str(e)[:80])
    with ThreadPoolExecutor(8) as ex:list(ex.map(one,tiles))
    res=2*ORG/(256*2**z);tr=from_origin(-ORG+X0*256*res,ORG-Y0*256*res,res,res)
    return arr,tr,got
def write_tif(path,arr,tr,crs,nodata=None,photometric=None):
    if arr.ndim==2:arr=arr[None]
    prof=dict(driver='GTiff',height=arr.shape[1],width=arr.shape[2],count=arr.shape[0],dtype=arr.dtype,crs=crs,transform=tr,compress='deflate',tiled=True,blockxsize=256,blockysize=256,predictor=2 if arr.dtype!=np.uint8 else 1)
    if nodata is not None:prof['nodata']=nodata
    if arr.shape[0]==3:prof.update(compress='jpeg',jpeg_quality=88,photometric='ycbcr');prof.pop('predictor',None)
    with rasterio.open(path,'w',**prof) as d:d.write(arr)
    log('wrote',os.path.basename(path),arr.shape,round(os.path.getsize(path)/1e6,1),'MB')

# ---------------- 1. terrain tiles ----------------
def terrarium(x,y,z):
    r=get(f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',timeout=60)
    im=np.asarray(Image.open(io.BytesIO(r.content)).convert('RGB'),np.float32)
    return np.round(im[...,0]*256+im[...,1]+im[...,2]/256-32768).astype('int16')[None]
def do_terrain():
    for z,corr,name in((11,None,'terrain_z11_box.tif'),(13,2.5,'terrain_z13_course.tif')):
        tiles=tiles_for(z,corr);log('terrain z',z,'tiles',len(tiles))
        arr,tr,got=mosaic(z,tiles,terrarium,-32768)
        write_tif(os.path.join(OUT,name),arr,tr,CRS.from_epsg(3857),nodata=-32768)
        done('terrain_z%d'%z,tiles=len(tiles),got=got,file=name)

# ---------------- 2. OpenStreetMap via Overpass ----------------
def osm_to_geojson(elements):
    nodes={e['id']:(e['lon'],e['lat']) for e in elements if e['type']=='node'}
    ways={e['id']:e for e in elements if e['type']=='way'}
    def is_area(t):
        if not t:return False
        if t.get('area')=='yes':return True
        if 'building' in t or 'landuse' in t or 'leisure' in t or 'amenity' in t or 'water' in t:return True
        if t.get('natural') in('water','wood','scrub','heath','grassland','sand','beach','wetland'):return True
        if t.get('waterway') in('riverbank','dock'):return True
        return False
    feats=[];used=set()
    rels=[e for e in elements if e['type']=='relation' and e.get('tags',{}).get('type')=='multipolygon']
    def rings_from(members):
        segs=[]
        for m in members:
            w=ways.get(m['ref'])
            if not w:continue
            pts=[nodes[n] for n in w.get('nodes',[]) if n in nodes]
            if len(pts)>=2:segs.append(pts)
        rings=[]
        while segs:
            cur=segs.pop(0)
            while cur[0]!=cur[-1]:
                for i,s in enumerate(segs):
                    if s[0]==cur[-1]:cur=cur+s[1:];segs.pop(i);break
                    if s[-1]==cur[-1]:cur=cur+s[::-1][1:];segs.pop(i);break
                else:break
            if len(cur)>=4 and cur[0]==cur[-1]:rings.append(cur)
        return rings
    def area(r):return 0.5*abs(sum(r[i][0]*r[i+1][1]-r[i+1][0]*r[i][1] for i in range(len(r)-1)))
    def inside(pt,r):
        x,y=pt;ins=False
        for i in range(len(r)-1):
            (x0,y0),(x1,y1)=r[i],r[i+1]
            if (y0>y)!=(y1>y) and x<(x1-x0)*(y-y0)/(y1-y0+1e-300)+x0:ins=not ins
        return ins
    for rel in rels:
        outer=rings_from([m for m in rel['members'] if m['type']=='way' and m.get('role','outer') in('outer','')])
        inner=rings_from([m for m in rel['members'] if m['type']=='way' and m.get('role')=='inner'])
        for m in rel['members']:
            if m['type']=='way':used.add(m['ref'])
        if not outer:continue
        polys=[]
        for o in sorted(outer,key=area,reverse=True):
            holes=[i for i in inner if inside(i[0],o)];polys.append([o]+holes)
        geom={'type':'Polygon','coordinates':polys[0]} if len(polys)==1 else {'type':'MultiPolygon','coordinates':polys}
        feats.append({'type':'Feature','id':'relation/%d'%rel['id'],'properties':dict(rel.get('tags',{}),**{'@id':'relation/%d'%rel['id']}),'geometry':geom})
    for w in ways.values():
        t=w.get('tags')
        if not t or (w['id'] in used and not t.get('highway')):continue
        pts=[nodes[n] for n in w.get('nodes',[]) if n in nodes]
        if len(pts)<2:continue
        if pts[0]==pts[-1] and len(pts)>=4 and is_area(t):geom={'type':'Polygon','coordinates':[pts]}
        else:geom={'type':'LineString','coordinates':pts}
        feats.append({'type':'Feature','id':'way/%d'%w['id'],'properties':dict(t,**{'@id':'way/%d'%w['id']}),'geometry':geom})
    for e in elements:
        if e['type']=='node' and e.get('tags'):feats.append({'type':'Feature','id':'node/%d'%e['id'],'properties':dict(e['tags'],**{'@id':'node/%d'%e['id']}),'geometry':{'type':'Point','coordinates':[e['lon'],e['lat']]}})
    return {'type':'FeatureCollection','features':feats}
def do_osm():
    if spec.get('geofabrik'):   # regional extract first: Overpass rejects or times out on long course corridors from CI runners
        gj=geofabrik_osm();json.dump(gj,open(os.path.join(OUT,'osm.geojson'),'w'),separators=(',',':'))
        kinds={}
        for f in gj['features']:
            p=f['properties'];k='building' if 'building' in p else 'highway' if 'highway' in p else 'water' if (p.get('natural')=='water' or 'waterway' in p or 'water' in p) else 'other'
            kinds[k]=kinds.get(k,0)+1
        done('osm',source='geofabrik',features=len(gj['features']),kinds=kinds,file='osm.geojson');return
    line=','.join('%.5f,%.5f'%(la,lo) for la,lo in zip(LAT[::2],LON[::2]))
    q=f'''[out:json][timeout:600][maxsize:1073741824];
(
  way["building"](around:900,{line});
  way["highway"](around:1500,{line});
  way["natural"="water"](around:3500,{line});
  relation["natural"="water"](around:3500,{line});
  way["waterway"](around:3500,{line});
  way["landuse"](around:3000,{line});
  relation["landuse"](around:3000,{line});
  way["natural"~"^(wood|scrub|heath|grassland|sand|beach|wetland)$"](around:3000,{line});
  relation["natural"~"^(wood|scrub)$"](around:3000,{line});
  way["leisure"~"^(park|pitch|golf_course|nature_reserve)$"](around:2000,{line});
  node["natural"="tree"](around:400,{line});
);
out body; >; out skel qt;'''
    for ep in('https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter','https://maps.mail.ru/osm/tools/overpass/api/interpreter'):
        try:
            log('overpass',ep);r=requests.post(ep,data={'data':q},headers=UA,timeout=900);r.raise_for_status();j=r.json();break
        except Exception as e:log('overpass failed',ep,str(e)[:120]);j=None
    if j is None:
        gj=geofabrik_osm()
        if gj is None:raise RuntimeError('all Overpass endpoints failed and no Geofabrik extract is set for this course')
        src='geofabrik'
    else:
        els=j.get('elements',[]);json.dump(j,open(os.path.join(OUT,'osm_overpass.json'),'w'),separators=(',',':'))
        gj=osm_to_geojson(els);src='overpass'
    json.dump(gj,open(os.path.join(OUT,'osm.geojson'),'w'),separators=(',',':'))
    els=gj['features']
    kinds={}
    for f in gj['features']:
        p=f['properties'];k='building' if 'building' in p else 'highway' if 'highway' in p else 'water' if (p.get('natural')=='water' or 'waterway' in p or 'water' in p) else 'wood' if p.get('natural') in('wood','scrub') or p.get('landuse') in('forest','orchard') else 'landuse' if 'landuse' in p else 'tree' if p.get('natural')=='tree' else 'other'
        kinds[k]=kinds.get(k,0)+1
    done('osm',source=src,features=len(gj['features']),kinds=kinds,file='osm.geojson')

KEEP={'ele','tourism','building','highway','natural','waterway','landuse','leisure','water','amenity','name','surface','access','layer','bridge','tunnel','building:levels','height','tourism','place',
      'roof:shape','roof:material','roof:colour','roof:levels','building:material','building:colour','species','genus','leaf_type','leaf_cycle','wood','barrier','wall','lanes','oneway'}   # roof, wall and tree-species tags for the 3D detail; hedges and walls for the UK
def geofabrik_osm():
    """Regional extract from download.geofabrik.de (OpenStreetMap, ODbL), cut to the course box and exported to GeoJSON with osmium."""
    url=spec.get('geofabrik')
    if not url:return None
    import subprocess
    urls=url if isinstance(url,list) else [url]   # a course across a state or country border lists both extracts
    box=os.path.join('/tmp',pid+'_box.osm.pbf');gjp=os.path.join('/tmp',pid+'_box.geojson');parts=[]
    for k,u in enumerate(urls):
        pbf=os.path.join('/tmp',pid+'_region%d.osm.pbf'%k);bk=os.path.join('/tmp',pid+'_box%d.osm.pbf'%k)
        log('geofabrik',u)
        with requests.get(u,headers=UA,stream=True,timeout=900) as r:
            r.raise_for_status()
            with open(pbf,'wb') as f:
                for ch in r.iter_content(1<<22):f.write(ch)
        log('geofabrik MB',round(os.path.getsize(pbf)/1e6))
        subprocess.run(['osmium','extract','-b','%f,%f,%f,%f'%(W,S,E,N),pbf,'-o',bk,'--overwrite','-s','smart'],check=True);os.remove(pbf);parts.append(bk)
    if len(parts)>1:subprocess.run(['osmium','merge']+parts+['-o',box,'--overwrite'],check=True)
    else:os.replace(parts[0],box)
    subprocess.run(['osmium','export',box,'-o',gjp,'--overwrite','-f','geojson','--geometry-types=point,linestring,polygon'],check=True)
    raw=json.load(open(gjp));out=[]
    for f in raw['features']:
        p=f.get('properties') or {}
        if not any(k in p for k in('building','highway','natural','waterway','landuse','leisure','water','place','tourism')):continue
        LM=p.get('natural') in('peak','volcano') or p.get('place') in('city','town','village') or p.get('tourism')=='viewpoint' or p.get('waterway')=='waterfall' or p.get('natural')=='waterfall'
        if f['geometry']['type']=='Point' and p.get('natural')!='tree' and not LM:continue
        if LM and f['geometry']['type']=='Point':
            c=f['geometry']['coordinates']
            if p.get('name') and dist_km(c[1],c[0])<=15:out.append({'type':'Feature','geometry':f['geometry'],'properties':{k:v for k,v in p.items() if k in KEEP or k in('ele','population')}})
            continue
        if f['geometry']['type']=='LineString' and not ('highway' in p or 'waterway' in p):continue   # osmium also writes closed areas as lines
        if 'building' in p or f['geometry']['type']=='Point':
            c=f['geometry']['coordinates']
            while isinstance(c[0],list):c=c[0]   # first vertex, whatever the geometry type
            if dist_km(c[1],c[0])>(0.9 if 'building' in p else 0.4):continue
        out.append({'type':'Feature','geometry':f['geometry'],'properties':{k:v for k,v in p.items() if k in KEEP}})
    log('geofabrik features kept',len(out),'of',len(raw['features']))
    return {'type':'FeatureCollection','features':out}

# ---------------- 3. open-meteo profile ----------------
def do_profile():
    z=[]
    for i in range(0,len(LAT),100):
        r=get('https://api.open-meteo.com/v1/elevation',params={'latitude':','.join('%.5f'%v for v in LAT[i:i+100]),'longitude':','.join('%.5f'%v for v in LON[i:i+100])},timeout=60)
        z+=r.json()['elevation'];time.sleep(0.3)
    json.dump({'lat':LAT.tolist(),'lon':LON.tolist(),'z':z,'source':'Open-Meteo elevation API (Copernicus DEM 90 m)'},open(os.path.join(OUT,'profile_openmeteo.json'),'w'))
    done('profile',points=len(z))

# ---------------- 4a. New Zealand: LINZ open data on AWS ----------------
NZI='https://nz-imagery.s3.ap-southeast-2.amazonaws.com/';NZE='https://nz-elevation.s3.ap-southeast-2.amazonaws.com/'
def stac_collections(root):
    cat=get(root+'catalog.json',timeout=120).json();out=[]
    for l in cat.get('links',[]):
        if l.get('rel')=='child':out.append(l['href'] if l['href'].startswith('http') else root+l['href'].lstrip('./'))
    log('stac children',len(out),out[:3])
    return out
def bbox_hits(b):
    return not (b[2]<W or b[0]>E or b[3]<S or b[1]>N)
def pick_collections(root,want):
    """collections whose extent covers the course box; want='rgb' (imagery), 'dem' (bare earth)"""
    hits=[]
    hrefs=stac_collections(root)
    def fetch(h):
        try:return h,get(h,timeout=60).json()
        except Exception:return h,None
    with ThreadPoolExecutor(24) as ex:got=list(ex.map(fetch,hrefs))
    for href,c in got:
        if c is None:continue
        b=c.get('extent',{}).get('spatial',{}).get('bbox',[[0,0,0,0]])[0]
        if not bbox_hits(b):continue
        cid=c.get('id','');title=c.get('title','')
        txt=' '.join([cid,title,href,c.get('description','')[:200]])   # LINZ ids are ULIDs: the title and the path carry the details
        gsd=None
        m=re.search(r'(\d+(?:\.\d+)?)\s*m\b',title) or re.search(r'_(\d+(?:\.\d+)?)m',href)
        if m:gsd=float(m.group(1))
        if want=='rgb' and not ('rgb' in href.lower() or 'aerial' in title.lower() or 'photo' in title.lower()):continue
        if re.search(r'infrared|nir|draft',title.lower()):continue
        if want=='dsm' and not re.search(r'dsm|surface model',(href+' '+title).lower()):continue
        if want=='dem' and not re.search(r'dem|elevation model',(href+' '+title).lower()):continue
        if want=='dem' and re.search(r'dsm|surface model',(href+' '+title).lower()):continue
        yrs=re.findall(r'(20\d\d)',title+' '+href);yr=max(map(int,yrs)) if yrs else 0
        hits.append(dict(href=href,id=cid,title=title,gsd=gsd,year=yr,bbox=b,links=c.get('links',[]),base=href.rsplit('/',1)[0]+'/',cover=line_cover([b])))
    log('collections in box',want,[(h['title'][:60],h['gsd'],h['year']) for h in hits[:12]])
    return hits
def line_cover(boxes):
    """share of the course line (every 150 m point) inside any of the boxes (lon/lat bbox lists)"""
    inside=np.zeros(len(LAT),bool)
    for b in boxes:inside|=(LON>=b[0])&(LON<=b[2])&(LAT>=b[1])&(LAT<=b[3])
    return float(inside.mean())
def collection_items(c):
    items=[]
    for l in c['links']:
        if l.get('rel')=='item':
            href=l['href'] if l['href'].startswith('http') else c['base']+l['href'].lstrip('./')
            items.append(href)
    return items
def items_in_box(c,asset_pick):
    """item JSONs fetched 32 at a time (a regional collection has thousands); returns (bbox, asset href) inside the course box"""
    hrefs=collection_items(c);log('items to scan',c['title'][:50],len(hrefs))
    def one(it):
        try:j=get(it,timeout=60).json()
        except Exception:return None
        b=j.get('bbox');a=asset_pick(j.get('assets',{}))
        if b and a and bbox_hits(b):return (b,a['href'] if a['href'].startswith('http') else c['base']+a['href'].lstrip('./'))
        return None
    with ThreadPoolExecutor(32) as ex:out=[x for x in ex.map(one,hrefs) if x]
    return out
def read_remote(url,s,n,w,e,nrow,ncol,bands,dst,dst_crs='EPSG:4326'):
    """reproject a remote COG into the lat/lon grid, picking the overview nearest the target resolution"""
    tr=from_bounds(w,s,e,n,ncol,nrow)
    with rasterio.open(url) as d:
        tb=transform_bounds(d.crs,'EPSG:4326',*d.bounds)
        if tb[2]<w or tb[0]>e or tb[3]<s or tb[1]>n:return False
        target=(e-w)*kx*1000/ncol;ov=d.overviews(1);lvl=None;sc=1
        for i,o in enumerate(ov):
            if d.res[0]*o<=target*1.05:lvl=i;sc=o
        src=rasterio.open(url,overview_level=lvl) if lvl is not None else d
        for b in range(bands):
            reproject(rasterio.band(src,b+1),dst[b],dst_transform=tr,dst_crs=dst_crs,resampling=Resampling.average,src_nodata=d.nodata,dst_nodata=0 if bands==3 else -32768,init_dest_nodata=False)
        if src is not d:src.close()
    return True
def spots():
    sp=[(s[0],s[1],s[2],s[3]) for s in spec.get('spots',[])]
    # densest built-up cells near the course, from the OSM buildings
    p=os.path.join(OUT,'osm.geojson')
    if os.path.exists(p):
        bc=[]
        for f in json.load(open(p))['features']:
            if 'building' in f['properties'] and f['geometry']['type']=='Polygon':
                c=f['geometry']['coordinates'][0][0];bc.append((c[1],c[0]))
        if bc:
            bc=np.array(bc);d=np.array([dist_km(la,lo) for la,lo in bc]);bc=bc[d<0.6]
            if len(bc):
                cell=np.round(bc/0.02).astype(int);u,cnt=np.unique(cell,axis=0,return_counts=True)
                for j in np.argsort(-cnt):
                    la,lo=u[j]*0.02
                    if all(math.hypot((la-s[0])*kz,(lo-s[1])*kx)>1.5 for s in sp):sp.append((float(la),float(lo),2.5,'town'))
                    if len(sp)>=7:break
    return sp[:max(7,int(spec.get('max_spots',7)))]   # twins list more than seven
def do_nz():
    img=pick_collections(NZI,'rgb');img=[c for c in img if (c['gsd'] or 0.3)<=0.6]
    img.sort(key=lambda c:(-round(c['cover'],1),-c['year'],c['gsd'] or 9));log('nz imagery collections',[(c['title'][:50],round(c['cover'],2),c['year'],c['gsd']) for c in img[:8]])
    dem=pick_collections(NZE,'dem');dem.sort(key=lambda c:(-round(c['cover'],1),-c['year']));log('nz dem collections',[(c['title'][:50],round(c['cover'],2),c['year']) for c in dem[:8]])
    if img:
        srcs=[];used=[]
        for c in img[:6]:
            got=items_in_box(c,lambda A:A.get('visual') or next(iter(A.values()),None))
            if got:srcs+=got;used.append(c['title'])
            cv=line_cover([b for b,_ in srcs]);log('imagery cover of the course',round(cv,2))
            if cv>0.95:break
        img=[c for c in img if c['title'] in used] or img
        log('nz imagery items in box',len(srcs))
        # box at 8 m, detail spots at 0.75 m
        ncol=int((E-W)*kx*1000/8);nrow=int((N-S)*kz*1000/8);dst=np.zeros((3,nrow,ncol),np.uint8)
        for b,href in srcs:
            try:read_remote(href,S,N,W,E,nrow,ncol,3,dst)
            except Exception as e:log('item failed',href[-60:],str(e)[:80])
        write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,from_bounds(W,S,E,N,ncol,nrow),CRS.from_epsg(4326))
        k=0
        for la,lo,sz,nm in spots():
            s_,n_=la-sz/2/kz,la+sz/2/kz;w_,e_=lo-sz/2/kx,lo+sz/2/kx;px=int(sz*1000/0.75);d2=np.zeros((3,px,px),np.uint8);hit=False
            for b,href in srcs:
                if b[2]<w_ or b[0]>e_ or b[3]<s_ or b[1]>n_:continue
                try:hit|=read_remote(href,s_,n_,w_,e_,px,px,3,d2)
                except Exception as e:log('spot item failed',str(e)[:80])
            if hit and (d2.max(0)>0).mean()>0.3:write_tif(os.path.join(OUT,'ortho_spot_%d.tif'%k),d2,from_bounds(w_,s_,e_,n_,px,px),CRS.from_epsg(4326));k+=1
        done('nz_imagery',collections=[c['title'] for c in img[:4]],items=len(srcs),spots=k,cover=round(line_cover([b for b,_ in srcs]),2))
    else:failed('nz_imagery','no LINZ imagery collection covers the box')
    if dem:
        srcs=[];used=[]
        for c in dem[:6]:
            got=items_in_box(c,lambda A:next(iter(A.values()),None))
            if got:srcs+=got;used.append(c['title'])
            cv=line_cover([b for b,_ in srcs]);log('dem cover of the course',round(cv,2))
            if cv>0.95:break
        dem=[c for c in dem if c['title'] in used] or dem
        log('nz dem items in box',len(srcs))
        for res_m,name,corr in((10,'dem_box_10m.tif',None),(3,'dem_course_3m.tif',0.35)):
            ncol=int((E-W)*kx*1000/res_m);nrow=int((N-S)*kz*1000/res_m)
            if ncol*nrow>120e6:log('skip',name,'too large');continue
            dst=np.full((1,nrow,ncol),-32768,np.int16);hit=0
            for b,href in srcs:
                try:hit+=read_remote(href,S,N,W,E,nrow,ncol,1,dst)
                except Exception as e:log('dem item failed',str(e)[:80])
            if corr is not None:  # keep only the corridor so the file stays small
                gy=np.linspace(N,S,nrow)[:,None];gx=np.linspace(W,E,ncol)[None,:]
                m=np.ones((nrow,ncol),bool)
                for r0 in range(0,nrow,256):
                    for c0 in range(0,ncol,256):
                        la=float(gy[min(nrow-1,r0+128),0]);lo=float(gx[0,min(ncol-1,c0+128)])
                        if dist_km(la,lo)>corr+0.3:m[r0:r0+256,c0:c0+256]=False
                dst[0][~m]=-32768
            write_tif(os.path.join(OUT,name),dst,from_bounds(W,S,E,N,ncol,nrow),CRS.from_epsg(4326),nodata=-32768)
        done('nz_dem',collections=[c['title'] for c in dem[:4]],items=len(srcs),cover=round(line_cover([b for b,_ in srcs]),2))
    else:failed('nz_dem','no LINZ DEM collection covers the box')

# ---------------- 4b. Bavaria: DOP20 WMS and DGM1 open data ----------------
def do_bayern():
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326','EPSG:25832',always_xy=True)
    x0,y0=T.transform(W,S);x1,y1=T.transform(E,N)
    # DOP20 (0.2 m) via WMS GetMap in 2048 px chunks: box at 8 m, spots at 0.8 m
    wms='https://geoservices.bayern.de/od/wms/dop/v1/dop20'
    def getmap(xa,ya,xb,yb,px,py):
        r=get(wms,params=dict(SERVICE='WMS',VERSION='1.3.0',REQUEST='GetMap',LAYERS='by_dop20c',STYLES='',CRS='EPSG:25832',BBOX='%f,%f,%f,%f'%(xa,ya,xb,yb),WIDTH=px,HEIGHT=py,FORMAT='image/jpeg'),timeout=180)
        if not r.headers.get('content-type','').startswith('image'):raise RuntimeError('WMS answered '+r.text[:120])
        return np.asarray(Image.open(io.BytesIO(r.content)).convert('RGB'),np.uint8).transpose(2,0,1)
    try:
        res_m=8;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);dst=np.zeros((3,nrow,ncol),np.uint8);n=0
        for r0 in range(0,nrow,2048):
            for c0 in range(0,ncol,2048):
                h=min(2048,nrow-r0);w=min(2048,ncol-c0);xa=x0+c0*res_m;yb=y1-r0*res_m
                dst[:,r0:r0+h,c0:c0+w]=getmap(xa,yb-h*res_m,xa+w*res_m,yb,w,h);n+=1
        write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,from_origin(x0,y1,res_m,res_m),CRS.from_epsg(25832))
        k=0
        for la,lo,sz,nm in spots():
            cx,cy=T.transform(lo,la);px=int(sz*1000/0.8);im=getmap(cx-sz*500,cy-sz*500,cx+sz*500,cy+sz*500,min(px,4096),min(px,4096))
            write_tif(os.path.join(OUT,'ortho_spot_%d.tif'%k),im,from_origin(cx-sz*500,cy+sz*500,sz*1000/im.shape[2],sz*1000/im.shape[1]),CRS.from_epsg(25832));k+=1
        done('bayern_dop20',chunks=n,spots=k)
    except Exception as e:failed('bayern_dop20',e)
    # DGM1: 1 km tiles named by UTM32 km easting/northing on the open data download server; several naming patterns tried
    try:
        ex=range(int(x0//1000),int(x1//1000)+1);ny=range(int(y0//1000),int(y1//1000)+1);got=0;tried=0;miss=[];srcs=[]
        os.makedirs(os.path.join(OUT,'dgm1'),exist_ok=True)
        pats=['https://download1.bayernwolke.de/a/dgm/dgm1/{e}_{n}.tif','https://download1.bayernwolke.de/a/dgm/dgm1/dgm1_{e}_{n}.tif','https://download1.bayernwolke.de/a/dgm/dgm1/{e}{n}.tif']
        okpat=None
        for e_ in ex:
            for n_ in ny:
                # only tiles within 1.5 km of the course
                lo,la=Transformer.from_crs('EPSG:25832','EPSG:4326',always_xy=True).transform(e_*1000+500,n_*1000+500)
                if dist_km(la,lo)>1.5:continue
                tried+=1;ok=False
                for pat in([okpat] if okpat else pats):
                    url=pat.format(e=e_,n=n_)
                    try:
                        r=requests.get(url,headers=UA,timeout=120)
                        if r.status_code==200 and r.content[:2] in(b'II',b'MM'):
                            p=os.path.join(OUT,'dgm1','%d_%d.tif'%(e_,n_));open(p,'wb').write(r.content);srcs.append(p);got+=1;ok=True;okpat=pat;break
                    except Exception:pass
                if not ok:miss.append((e_,n_))
        if got:
            # one corridor mosaic at 2 m so the pack build reads a single file
            # heights in decimetres (int16) so the 1 m detail survives; each tile warps into the mosaic without blanking the others
            res_m=3;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);acc=np.full((nrow,ncol),np.nan,np.float32);tr=from_origin(x0,y1,res_m,res_m)
            for p in srcs:
                with rasterio.open(p) as d:
                    tmp=np.full((nrow,ncol),np.nan,np.float32)
                    reproject(rasterio.band(d,1),tmp,dst_transform=tr,dst_crs='EPSG:25832',resampling=Resampling.average,src_nodata=d.nodata,dst_nodata=np.nan)
                    m=np.isfinite(tmp)&(tmp>-500)&(tmp<5000);acc[m]=tmp[m]
            dst=np.where(np.isfinite(acc),np.round(acc*10),-32768).astype(np.int16)[None]
            log('dgm1 mosaic valid share',round(float(np.isfinite(acc).mean()),3))
            write_tif(os.path.join(OUT,'dem_course_3m_dm.tif'),dst,tr,CRS.from_epsg(25832),nodata=-32768)
            old=os.path.join(OUT,'dem_course_3m.tif')
            if os.path.exists(old):os.remove(old)
            for p in srcs:os.remove(p)
            done('bayern_dgm1',tiles=got,tried=tried,pattern=okpat,file='dem_course_3m_dm.tif',units='decimetres')
        else:failed('bayern_dgm1','no DGM1 tile answered (%d tried); first misses %s'%(tried,miss[:3]))
    except Exception as e:failed('bayern_dgm1',e)

# ---------------- 4c. New Zealand: surface model minus terrain = heights of buildings and trees (the Kona LiDAR method) ----------------
def course_xy(crs):
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326',crs,always_xy=True);x,y=T.transform(LON,LAT)
    # densify the 150 m line to 25 m so the corridor distance is right
    X=[];Y=[]
    for i in range(len(x)-1):
        n=6;X+=list(np.linspace(x[i],x[i+1],n,endpoint=False));Y+=list(np.linspace(y[i],y[i+1],n,endpoint=False))
    return np.array(X),np.array(Y)
class NdsmAcc:
    """Accumulates building heights and tree tops tile by tile, so a long LiDAR run can write as it goes.
    add(dsm_href, dem_href or None, dem_scale): dem None means the raster already holds height above ground."""
    def __init__(self,crs_hint=None,corr_m=900,near_m=200,cap=45000):
        osm=json.load(open(os.path.join(OUT,'osm.geojson')))['features'] if os.path.exists(os.path.join(OUT,'osm.geojson')) else []
        self.blds=[f for f in osm if 'building' in f['properties'] and f['geometry']['type'] in('Polygon','MultiPolygon')]
        self.BH={};self.TREES=[];self.crs=None;self.corr_m=corr_m;self.near_m=near_m;self.cap=cap;self.n=0;self.fail=0;self.stats=[]
    def _init(self,crs):
        from scipy.spatial import cKDTree
        from shapely.geometry import shape
        from shapely.ops import transform as stransform
        from pyproj import Transformer
        self.crs=crs;cx,cy=course_xy(crs);self.tree_ctr=cKDTree(np.c_[cx,cy])
        self.T2=Transformer.from_crs(crs,'EPSG:4326',always_xy=True);Tb=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
        self.bgeo=[]
        for f in self.blds:
            try:g=stransform(lambda x,y,z=None:Tb.transform(x,y),shape(f['geometry']))
            except Exception:continue
            c=f['geometry']['coordinates']
            while isinstance(c[0][0],list):c=c[0]
            self.bgeo.append(('%.6f,%.6f'%(c[0][1],c[0][0]),g))
    def add(self,dsm_h,dem_h=None,dsc=1.0,label=''):
        from scipy.ndimage import maximum_filter,binary_dilation
        from rasterio import features as rfeat
        self.n+=1
        try:
            with rasterio.open(dsm_h) as ds:
                if self.crs is None:self._init(ds.crs)
                b=ds.bounds;cxm=(b.left+b.right)/2;cym=(b.bottom+b.top)/2;half=math.hypot(b.right-b.left,b.top-b.bottom)/2
                if self.tree_ctr.query([cxm,cym])[0]>half+self.corr_m:return
                a=ds.read(1).astype(np.float32);nd=ds.nodata;H,Wd=a.shape;tr=ds.transform
                if dem_h:
                    with rasterio.open(dem_h) as dm:
                        g=dm.read(1,window=dm.window(*b),boundless=True,out_shape=a.shape).astype(np.float32)*dsc;gnd=dm.nodata
                else:g=np.zeros_like(a);gnd=None
            bad=(a<-100)|(g<-100)|(a>9000)|(g>9000)
            if nd is not None:bad|=(a==nd)
            if gnd is not None:bad|=(g==gnd*dsc)
            h=a-g;h[bad]=0;h[h<0]=0
            if len(self.stats)<6:self.stats.append(dict(tile=label or os.path.basename(str(dsm_h)),valid=round(float((~bad).mean()),3),p50=round(float(np.percentile(h[~bad],50)),2) if (~bad).any() else None,p99=round(float(np.percentile(h[~bad],99)),2) if (~bad).any() else None));log('ndsm tile stats',self.stats[-1])
            ys=np.arange(0,H,10);xs=np.arange(0,Wd,10);XX,YY=np.meshgrid(tr.c+tr.a*(xs+0.5),tr.f+tr.e*(ys+0.5))
            dd=self.tree_ctr.query(np.c_[XX.ravel(),YY.ravel()])[0].reshape(XX.shape);dd=np.repeat(np.repeat(dd,10,0),10,1)[:H,:Wd]
            corr=dd<self.corr_m
            bm=np.zeros((H,Wd),bool);inb=[]
            for key,gm in self.bgeo:
                bx=gm.bounds
                if bx[2]<b.left or bx[0]>b.right or bx[3]<b.bottom or bx[1]>b.top:continue
                inb.append((key,gm))
            if inb:
                lab=rfeat.rasterize([(gm.buffer(-0.8) if gm.area>60 else gm,i+1) for i,(key,gm) in enumerate(inb)],out_shape=(H,Wd),transform=tr,fill=0,dtype='int32')
                full=rfeat.rasterize([(gm,1) for key,gm in inb],out_shape=(H,Wd),transform=tr,fill=0,dtype='uint8').astype(bool);bm=binary_dilation(full,iterations=2)
                idx=lab.ravel();vals=h.ravel();o=np.argsort(idx,kind='stable');si=idx[o];sv=vals[o];cuts=np.searchsorted(si,np.arange(1,len(inb)+2))
                for i,(key,gm) in enumerate(inb):
                    v=sv[cuts[i]:cuts[i+1]]
                    if len(v)<4:continue
                    v=v[v>0.5]
                    if len(v)<0.6*max(1,cuts[i+1]-cuts[i]):continue
                    p90=float(np.percentile(v,90));p25=float(np.percentile(v,25));p98=float(np.percentile(v,98))
                    pit=1 if p90-p25>1.2 else 0
                    if 2<=p90<=80:self.BH[key]=[round(p98 if pit else p90,1),pit,round(max(2.2,p25) if pit else p90,1)]   # [top, pitched, eave]
            # a crown has area: at least 40 % of the 5 x 5 window above 3 m (drops wires, poles and sign posts, which are a pixel wide)
            from scipy.ndimage import uniform_filter
            crown=uniform_filter((h>3).astype(np.float32),size=5)>=0.4
            cand=(h>4)&(~bm)&corr&crown&(h==maximum_filter(h,size=5))
            r,c=np.nonzero(cand)
            if len(r):
                X=tr.c+tr.a*(c+0.5);Y=tr.f+tr.e*(r+0.5);lo,la=self.T2.transform(X,Y)
                for q in range(len(r)):self.TREES.append((float(la[q]),float(lo[q]),float(h[r[q],c[q]]),float(dd[r[q],c[q]])))
            if self.n%10==0:log('ndsm tile',self.n,'buildings so far',len(self.BH),'tree tops so far',len(self.TREES))
        except Exception as e:
            self.fail+=1
            if self.fail<=3:log('ndsm tile failed',repr(e)[:200]);traceback.print_exc()
    def write(self):
        T=sorted(self.TREES,key=lambda t:t[3]);near=[t for t in T if t[3]<=self.near_m];far=[t for t in T if t[3]>self.near_m]
        rng=np.random.default_rng(3);room=max(0,self.cap-len(near))
        if len(far)>room:far=[far[i] for i in sorted(rng.choice(len(far),room,replace=False))]
        keep=near+far
        json.dump({'source':'surface model minus terrain model','buildings':self.BH},open(os.path.join(OUT,'ndsm_buildings.json'),'w'),separators=(',',':'))
        json.dump({'source':'surface model minus terrain model','trees':[[round(a,6),round(b,6),round(c,1)] for a,b,c,_ in keep],'total_found':len(T)},open(os.path.join(OUT,'ndsm_trees.json'),'w'),separators=(',',':'))
        return len(self.BH),len(keep),len(T)
def ndsm_products(pairs,crs_hint=None,corr_m=900,near_m=200,cap=45000):
    """pairs: list of (dsm_href, dem_href, dem_scale) on the same grid. Writes ndsm_buildings.json and ndsm_trees.json."""
    acc=NdsmAcc(crs_hint,corr_m,near_m,cap)
    for dsm_h,dem_h,dsc in pairs:acc.add(dsm_h,dem_h,dsc)
    return acc.write()
def nz_items(want,limit=6):
    cols=pick_collections(NZE,want);cols.sort(key=lambda c:(-round(c['cover'],1),-c['year']))
    log('nz',want,'collections',[(c['title'][:60],round(c['cover'],2),c['year']) for c in cols[:8]])
    srcs=[];used=[]
    for c in cols[:limit]:
        got=items_in_box(c,lambda A:next(iter(A.values()),None))
        if got:srcs+=[(b,h,c['title']) for b,h in got];used.append(c['title'])
        if line_cover([b for b,_,_ in srcs])>0.95:break
    return srcs,used
def do_nz_ndsm():
    dsm,du=nz_items('dsm');dem,eu=nz_items('dem')
    if not dsm:return failed('nz_ndsm','no LINZ DSM collection covers the box')
    key=lambda b:tuple(round(v,3) for v in b)
    D={key(b):h for b,h,_ in dem};pairs=[];miss=0
    for b,h,_ in dsm:
        # only sheets near the course
        cla=(b[1]+b[3])/2;clo=(b[0]+b[2])/2;half=math.hypot((b[2]-b[0])*kx,(b[3]-b[1])*kz)/2
        if dist_km(cla,clo)>half+1.0:continue
        if key(b) in D:pairs.append((h,D[key(b)],1.0))
        else:miss+=1
    log('dsm/dem sheet pairs',len(pairs),'dsm sheets without a matching dem sheet',miss)
    nb,nt,tot=ndsm_products(pairs)
    done('nz_ndsm',dsm=du[:3],dem=eu[:3],sheets=len(pairs),unmatched=miss,buildings_with_height=nb,trees_kept=nt,trees_found=tot)

# ---------------- 4d. Bavaria: which surface products the open data server offers (LoD2 buildings, DOM) ----------------
def do_bayern_probe():
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326','EPSG:25832',always_xy=True);x,y=T.transform(float(LON[len(LON)//3]),float(LAT[len(LAT)//3]))
    e1,n1=int(x//1000),int(y//1000);e2,n2=e1-e1%2,n1-n1%2
    urls=['https://download1.bayernwolke.de/a/lod2/citygml/%d_%d.gml'%(e2,n2),'https://download1.bayernwolke.de/a/lod2/citygml/%d_%d.gml'%(e1,n1),
          'https://download1.bayernwolke.de/a/dom20/DOM/%d_%d.tif'%(e1,n1),'https://download1.bayernwolke.de/a/dom40/DOM/%d_%d.tif'%(e1,n1),
          'https://download1.bayernwolke.de/a/dom20/dom20/%d_%d.tif'%(e1,n1),'https://download1.bayernwolke.de/a/laser/%d_%d.laz'%(e1,n1),
          'https://geodaten.bayern.de/odd/a/lod2/citygml/meta/metalink/09576.meta4','https://geodaten.bayern.de/odd/a/dom20/meta/metalink/09576.meta4',
          'https://geodaten.bayern.de/odd/a/laser/meta/metalink/09576.meta4','https://geodaten.bayern.de/odd/a/dgm/dgm1/meta/metalink/09576.meta4']
    res=[]
    for u in urls:
        try:
            r=requests.get(u,headers=UA,timeout=60,stream=True);head=r.raw.read(400,decode_content=True);res.append([u,r.status_code,r.headers.get('content-type',''),r.headers.get('content-length',''),head[:300].decode('latin1')]);r.close()
        except Exception as e:res.append([u,'error',str(e)[:120]])
    json.dump(res,open(os.path.join(OUT,'bayern_probe.json'),'w'),indent=1)
    done('bayern_probe',answers=[[a[0].rsplit('/',2)[-2]+'/'+a[0].rsplit('/',1)[-1],a[1]] for a in res])

# ---------------- 4e. Bavaria: LoD2 3D buildings (CityGML, measured heights and roof shapes, CC BY 4.0) ----------------
def do_bayern_lod2(corr_km=0.9):
    import xml.etree.ElementTree as ET
    from pyproj import Transformer
    from shapely.geometry import shape,Polygon,Point
    from shapely.strtree import STRtree
    from shapely.ops import transform as stransform
    T=Transformer.from_crs('EPSG:4326','EPSG:25832',always_xy=True);Ti=Transformer.from_crs('EPSG:25832','EPSG:4326',always_xy=True)
    osm=json.load(open(os.path.join(OUT,'osm.geojson')))['features']
    keys=[];polys=[]
    for f in osm:
        if 'building' not in f['properties'] or f['geometry']['type'] not in('Polygon','MultiPolygon'):continue
        c=f['geometry']['coordinates']
        while isinstance(c[0][0],list):c=c[0]
        try:g=stransform(lambda x,y,z=None:T.transform(x,y),shape(f['geometry']))
        except Exception:continue
        keys.append('%.6f,%.6f'%(c[0][1],c[0][0]));polys.append(g)
    tree=STRtree(polys);log('osm buildings to match',len(polys))
    x0,y0=T.transform(W,S);x1,y1=T.transform(E,N)
    tiles=[]
    for e_ in range(int(x0//2000)*2,int(x1//1000)+2,2):
        for n_ in range(int(y0//2000)*2,int(y1//1000)+2,2):
            lo,la=Ti.transform(e_*1000+1000,n_*1000+1000)
            if dist_km(la,lo)<=corr_km+1.5:tiles.append((e_,n_))
    log('lod2 tiles near the course',len(tiles))
    BH={};nb=0;got=0
    ln=lambda t:t.rsplit('}',1)[-1]
    def parse_tile(e_,n_):
        url='https://download1.bayernwolke.de/a/lod2/citygml/%d_%d.gml'%(e_,n_)
        r=requests.get(url,headers=UA,timeout=300,stream=True)
        if r.status_code!=200:return []
        r.raw.decode_content=True;out=[]
        for ev,el in ET.iterparse(r.raw,events=('end',)):
            if ln(el.tag)!='Building':continue
            zs=[];ground=[];roof=None;rz=[]
            for sub in el.iter():
                t=ln(sub.tag)
                if t=='roofType' and sub.text:roof=sub.text.strip()
                if t=='posList' and sub.text:
                    v=[float(q) for q in sub.text.split()];zs+=v[2::3]
                if t=='RoofSurface':
                    for pl in sub.iter():
                        if ln(pl.tag)=='posList' and pl.text:rz+=[float(q) for q in pl.text.split()][2::3]
            for gs in el.iter():
                if ln(gs.tag)=='GroundSurface':
                    for pl in gs.iter():
                        if ln(pl.tag)=='posList' and pl.text:
                            v=[float(q) for q in pl.text.split()];ground.append(list(zip(v[0::3],v[1::3])))
                            zg=min(v[2::3])
            if ground and zs:
                try:
                    P=Polygon(max(ground,key=len));z0=min(zs);h=max(zs)-z0;ev=(min(rz)-z0) if rz else h
                    out.append((P.centroid.x,P.centroid.y,h,0 if roof in(None,'1000') else 1,max(2.2,ev)))
                except Exception:pass
            el.clear()
        return out
    from concurrent.futures import ThreadPoolExecutor
    def safe(t):
        try:return parse_tile(*t)
        except Exception as ex:log('lod2 tile failed',t,str(ex)[:100]);return []
    with ThreadPoolExecutor(6) as ex:
        for res in ex.map(safe,tiles):
            if res:got+=1
            for cx,cy,h,pt,ev in res:
                lo,la=Ti.transform(cx,cy)
                if dist_km(la,lo)>corr_km+0.2:continue
                nb+=1
                for i in tree.query(Point(cx,cy)):
                    if polys[i].contains(Point(cx,cy)) and 2<=h<=80:
                        k=keys[i]
                        if k not in BH or BH[k][0]<h:BH[k]=[round(h,1),pt,round(ev if pt else h,1)]   # [top, pitched, eave]
                        break
    json.dump({'source':'Bavarian LoD2 building models (CityGML), Bayerische Vermessungsverwaltung, CC BY 4.0','buildings':BH},open(os.path.join(OUT,'ndsm_buildings.json'),'w'),separators=(',',':'))
    done('bayern_lod2',tiles=len(tiles),tiles_read=got,lod2_buildings_near_course=nb,osm_buildings_with_height=len(BH))

# ---------------- 4f. United States: USGS 3DEP LiDAR point clouds (public domain), Entwine on AWS, read with PDAL ----------------
def do_us_lidar(corr_km=0.45,cell_m=800):
    """USGS 3DEP point clouds (EPT, public, no key) -> height above ground rasters per 800 m cell with PDAL -> building heights and tree tops.
    Cells with the most buildings go first, results are written every 10 cells, and the run stops taking new cells at LIDAR_BUDGET_MIN
    (default 115) so the course's 140 minute limit never throws away finished work. LIDAR_MAX_CELLS limits a test run."""
    import subprocess,tempfile,shutil
    from pyproj import Transformer
    from concurrent.futures import ThreadPoolExecutor,as_completed
    res=get('https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson',timeout=180).json()
    hits=[]
    from shapely.geometry import shape,box,Point
    for f in res['features']:
        nm=f['properties'].get('name','');url=f['properties'].get('url','')
        try:
            g=shape(f['geometry'])
            if not g.intersects(box(W,S,E,N)):continue
        except Exception:continue
        yrs=re.findall(r'(20\d\d)',nm);hits.append(dict(name=nm,url=url,year=max(map(int,yrs)) if yrs else 0,count=f['properties'].get('count',0),geom=g))
    hits.sort(key=lambda h:(-h['year'],-h['count']))
    log('usgs lidar resources over the box',[(h['name'],h['year'],h['count']) for h in hits])
    if not hits:return failed('us_lidar','no USGS 3DEP point cloud covers the box')
    utm=spec.get('lidar_utm','EPSG:32605');T=Transformer.from_crs('EPSG:4326',utm,always_xy=True);Tm=Transformer.from_crs(utm,'EPSG:3857',always_xy=True)
    x,y=T.transform(LON,LAT)
    X=[];Y=[]
    for i in range(len(x)-1):X+=list(np.linspace(x[i],x[i+1],6,endpoint=False));Y+=list(np.linspace(y[i],y[i+1],6,endpoint=False))
    from scipy.spatial import cKDTree
    tr=cKDTree(np.c_[X,Y]);x0,x1,y0,y1=min(X)-corr_km*1000,max(X)+corr_km*1000,min(Y)-corr_km*1000,max(Y)+corr_km*1000
    cells=[]
    for cx in np.arange(x0,x1,cell_m):
        for cy in np.arange(y0,y1,cell_m):
            if tr.query([cx+cell_m/2,cy+cell_m/2])[0]<corr_km*1000+cell_m*0.71:cells.append((float(cx),float(cy)))
    # town first: count OSM buildings per cell
    acc=NdsmAcc(corr_m=corr_km*1000)
    cnt={}
    for f in acc.blds:
        c=f['geometry']['coordinates']
        while isinstance(c[0][0],list):c=c[0]
        bx,by=T.transform(c[0][0],c[0][1]);k=(int((bx-x0)//cell_m),int((by-y0)//cell_m));cnt[k]=cnt.get(k,0)+1
    cells.sort(key=lambda c:-cnt.get((int(round((c[0]-x0)/cell_m)),int(round((c[1]-y0)/cell_m))),0))
    mx=int(os.environ.get('LIDAR_MAX_CELLS','0') or 0)
    if mx:cells=cells[:mx]
    budget=float(os.environ.get('LIDAR_BUDGET_MIN','115'))*60
    log('lidar cells',len(cells),'with buildings',sum(1 for c in cells if cnt.get((int(round((c[0]-x0)/cell_m)),int(round((c[1]-y0)/cell_m))),0)))
    tmp=tempfile.mkdtemp();ERRS=[];t0=time.time()
    Tl=Transformer.from_crs(utm,'EPSG:4326',always_xy=True)
    def one(k):
        if time.time()-t0>budget:return None
        cx,cy=cells[k];lo,la=Tl.transform(cx+cell_m/2,cy+cell_m/2)
        src=next((h for h in hits if h['geom'].contains(Point(lo,la))),None)
        if not src:return None
        mx0,my0=Tm.transform(cx-20,cy-20);mx1,my1=Tm.transform(cx+cell_m+20,cy+cell_m+20)
        out=os.path.join(tmp,'c%05d.tif'%k)
        pipe=[{'type':'readers.ept','filename':src['url'],'bounds':'([%f,%f],[%f,%f])'%(min(mx0,mx1),max(mx0,mx1),min(my0,my1),max(my0,my1)),'resolution':0.7,'threads':2},
              {'type':'filters.range','limits':'Classification![7:7],Classification![14:17],Classification![18:18]'},
              {'type':'filters.hag_nn'},
              {'type':'filters.reprojection','out_srs':utm},
              {'type':'writers.gdal','filename':out,'resolution':1.0,'dimension':'HeightAboveGround','output_type':'max','nodata':-9999,
               'origin_x':cx,'origin_y':cy,'width':int(cell_m),'height':int(cell_m),'data_type':'float32','gdalopts':'COMPRESS=DEFLATE'}]
        pj=os.path.join(tmp,'p%05d.json'%k);json.dump(pipe,open(pj,'w'))
        try:r=subprocess.run([os.environ.get('PDAL_BIN','pdal'),'pipeline',pj],capture_output=True,text=True,timeout=900)
        except subprocess.TimeoutExpired:return None
        if r.returncode!=0 or not os.path.exists(out):
            if len(ERRS)<3:log('pdal failed',r.stderr[-300:]);ERRS.append(r.stderr[-300:])
            return None
        return out
    nread=0
    with ThreadPoolExecutor(int(os.environ.get('LIDAR_THREADS','6'))) as ex:
        futs=[ex.submit(one,k) for k in range(len(cells))]
        for fu in as_completed(futs):
            t=fu.result()
            if not t:continue
            nread+=1;acc.add(t,None,1.0)
            try:os.remove(t)
            except OSError:pass
            if nread%10==0:
                nb,nt,tot=acc.write();done('us_lidar',partial=True,resources=[h['name'] for h in hits[:4]],cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,tile_stats=acc.stats,minutes=round((time.time()-t0)/60,1));save_report()
    log('lidar cells done',nread,'of',len(cells),'in',round((time.time()-t0)/60,1),'min')
    shutil.rmtree(tmp,ignore_errors=True)
    if not nread:return failed('us_lidar','PDAL produced no height rasters: '+' | '.join(ERRS[:2]))
    nb,nt,tot=acc.write()
    done('us_lidar',resources=[h['name'] for h in hits[:4]],cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,tile_stats=acc.stats,tile_failures=acc.fail,minutes=round((time.time()-t0)/60,1),stopped_on_budget=time.time()-t0>budget)

# ---------------- 4g. United States: USGS 3DEP bare-earth terrain and NAIP imagery (public domain) from The National Map image services ----------------
TNM_DEM='https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage'
TNM_NAIP='https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage'
def _tnm(url,bbox,epsg,w,h,fmt,pix=None):
    p=dict(bbox='%f,%f,%f,%f'%bbox,bboxSR=epsg,imageSR=epsg,size='%d,%d'%(w,h),format=fmt,f='image')
    if pix:p.update(pixelType=pix,noDataInterpretation='esriNoDataMatchAny',interpolation='RSP_BilinearInterpolation')
    r=get(url,params=p,timeout=240);ct=r.headers.get('content-type','')
    if 'image' not in ct and not r.content[:2] in(b'II',b'MM'):raise RuntimeError('image service answered '+ct+' '+r.text[:160])
    return r.content
def do_us():
    """NAIP box at 8 m and spots at 0.8 m; 3DEP bare earth along the course at 3 m (decimetres, int16), same files as the Bavarian recipe."""
    from pyproj import Transformer
    epsg=int(spec.get('lidar_utm','EPSG:32615').split(':')[1]);T=Transformer.from_crs('EPSG:4326','EPSG:%d'%epsg,always_xy=True)
    x0,y0=T.transform(W,S);x1,y1=T.transform(E,N);x0=min(x0,T.transform(W,N)[0]);x1=max(x1,T.transform(E,S)[0])
    # imagery
    try:
        res_m=8;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);dst=np.zeros((3,nrow,ncol),np.uint8);n=0;CH=2000
        for r0 in range(0,nrow,CH):
            for c0 in range(0,ncol,CH):
                h=min(CH,nrow-r0);w=min(CH,ncol-c0);xa=x0+c0*res_m;yb=y1-r0*res_m
                im=np.asarray(Image.open(io.BytesIO(_tnm(TNM_NAIP,(xa,yb-h*res_m,xa+w*res_m,yb),epsg,w,h,'jpg'))).convert('RGB'),np.uint8).transpose(2,0,1)
                dst[:,r0:r0+im.shape[1],c0:c0+im.shape[2]]=im[:,:h,:w];n+=1
        write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,from_origin(x0,y1,res_m,res_m),CRS.from_epsg(epsg))
        k=0
        for la,lo,sz,nm in spots():
            cx,cy=T.transform(lo,la);px=min(4000,int(sz*1000/0.8))
            im=np.asarray(Image.open(io.BytesIO(_tnm(TNM_NAIP,(cx-sz*500,cy-sz*500,cx+sz*500,cy+sz*500),epsg,px,px,'jpg'))).convert('RGB'),np.uint8).transpose(2,0,1)
            write_tif(os.path.join(OUT,'ortho_spot_%d.tif'%k),im,from_origin(cx-sz*500,cy+sz*500,sz*1000/im.shape[2],sz*1000/im.shape[1]),CRS.from_epsg(epsg));k+=1
        done('us_naip',chunks=n,spots=k,res_m=8)
    except Exception as e:failed('us_naip',e);traceback.print_exc()
    # bare earth: 2 km blocks within 1.5 km of the course, 3 m, mosaicked in decimetres
    try:
        res_m=3;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);acc=np.full((nrow,ncol),-32768,np.int16);B=2000;got=tried=0
        Ti=Transformer.from_crs('EPSG:%d'%epsg,'EPSG:4326',always_xy=True)
        for bx in np.arange(x0,x1,B):
            for by in np.arange(y0,y1,B):
                lo,la=Ti.transform(bx+B/2,by+B/2)
                if dist_km(la,lo)>1.5+1.42:continue
                tried+=1;w=int(min(B,x1-bx)/res_m);h=int(min(B,y1-by)/res_m)
                if w<2 or h<2:continue
                try:b=_tnm(TNM_DEM,(bx,by,bx+w*res_m,by+h*res_m),epsg,w,h,'tiff','F32')
                except Exception as e:log('dem block failed',e);continue
                with rasterio.open(io.BytesIO(b)) as d:a=d.read(1).astype(np.float32);nd=d.nodata
                m=np.isfinite(a)&(a>-500)&(a<6000)&((a!=nd) if nd is not None else True)
                c0=int(round((bx-x0)/res_m));r0=int(round((y1-(by+h*res_m))/res_m));sub=acc[r0:r0+h,c0:c0+w]
                sub[m[:sub.shape[0],:sub.shape[1]]]=np.round(a[:sub.shape[0],:sub.shape[1]][m[:sub.shape[0],:sub.shape[1]]]*10).astype(np.int16);got+=1
        log('3dep blocks',got,'of',tried,'valid share',round(float((acc>-32768).mean()),3))
        if got:write_tif(os.path.join(OUT,'dem_course_3m_dm.tif'),acc[None],from_origin(x0,y1,res_m,res_m),CRS.from_epsg(epsg),nodata=-32768);done('us_3dep_dem',blocks=got,tried=tried,file='dem_course_3m_dm.tif',units='decimetres')
        else:failed('us_3dep_dem','no 3DEP block answered (%d tried)'%tried)
    except Exception as e:failed('us_3dep_dem',e);traceback.print_exc()

# ---------------- 4h. Hessen (IRONMAN Frankfurt): HVBG open geodata, dl-de/zero-2-0 (free, commercial use allowed) ----------------
HE_WCS='https://inspire-hessen.de/raster/%s/ows'
HE_WMS='https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows'
def _he_cov(cov,e0,n0,e1,n1,svc):
    """GetCoverage in ETRS89 / UTM 32N (axes e, n); returns GeoTIFF bytes"""
    q=[('SERVICE','WCS'),('VERSION','2.0.1'),('REQUEST','GetCoverage'),('COVERAGEID',cov),('FORMAT','image/tiff'),('SUBSET','e(%d,%d)'%(e0,e1)),('SUBSET','n(%d,%d)'%(n0,n1))]
    for k in range(4):
        try:
            r=requests.get(HE_WCS%svc,params=q,headers=UA,timeout=240)
            if r.status_code==200 and r.content[:2] in(b'II',b'MM'):return r.content
            err='HTTP %s %s'%(r.status_code,r.text[:120])
        except Exception as e:err=repr(e)[:160]
        time.sleep(5*(k+1))
    raise RuntimeError(err)
def do_hessen():
    """DOP20 imagery (WMS, box at 8 m, spots at 0.8 m), DGM1 terrain along the course (2 m level, mosaicked at 3 m in decimetres),
    and building heights and tree crowns from DOM1 minus DGM1 at 1 m in 800 m cells within 450 m of the course."""
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326','EPSG:25832',always_xy=True);Ti=Transformer.from_crs('EPSG:25832','EPSG:4326',always_xy=True)
    x0,y0=T.transform(W,S);x1,y1=T.transform(E,N);x0=min(x0,T.transform(W,N)[0]);x1=max(x1,T.transform(E,S)[0])
    def getmap(xa,ya,xb,yb,px,py):
        r=get(HE_WMS,params=dict(SERVICE='WMS',VERSION='1.3.0',REQUEST='GetMap',LAYERS='he_dop20_rgb',STYLES='',CRS='EPSG:25832',BBOX='%f,%f,%f,%f'%(xa,ya,xb,yb),WIDTH=px,HEIGHT=py,FORMAT='image/jpeg'),timeout=240)
        if not r.headers.get('content-type','').startswith('image'):raise RuntimeError('WMS answered '+r.text[:160])
        return np.asarray(Image.open(io.BytesIO(r.content)).convert('RGB'),np.uint8).transpose(2,0,1)
    if 'imagery' not in ONLY_HE_SKIP:
        try:
            res_m=8;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);dst=np.zeros((3,nrow,ncol),np.uint8);n=0;CH=2000
            for r0 in range(0,nrow,CH):
                for c0 in range(0,ncol,CH):
                    h=min(CH,nrow-r0);w=min(CH,ncol-c0);xa=x0+c0*res_m;yb=y1-r0*res_m
                    dst[:,r0:r0+h,c0:c0+w]=getmap(xa,yb-h*res_m,xa+w*res_m,yb,w,h);n+=1
            if (dst.min(0)>=250).mean()<0.9:write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,from_origin(x0,y1,res_m,res_m),CRS.from_epsg(25832))
            else:do_hessen_box()
            k=0;covs=[]
            for la,lo,sz,nm in spots():
                cx,cy=T.transform(lo,la);px=min(3000,int(sz*1000/0.8));im=getmap(cx-sz*500,cy-sz*500,cx+sz*500,cy+sz*500,px,px)
                write_tif(os.path.join(OUT,'ortho_spot_%d.tif'%k),im,from_origin(cx-sz*500,cy+sz*500,sz*1000/im.shape[2],sz*1000/im.shape[1]),CRS.from_epsg(25832));k+=1
            done('hessen_dop20',chunks=n,spots=k,res_m=8)
        except Exception as e:failed('hessen_dop20',e);traceback.print_exc()
    # terrain: 2 km blocks within 2.9 km of the course from the 2 m DGM1 level
    try:
        res_m=3;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);acc=np.full((nrow,ncol),-32768,np.int16);B=2000;got=tried=0;tr=from_origin(x0,y1,res_m,res_m)
        for bx in np.arange(x0,x1,B):
            for by in np.arange(y0,y1,B):
                lo,la=Ti.transform(bx+B/2,by+B/2)
                if dist_km(la,lo)>1.5+1.42:continue
                tried+=1
                try:b=_he_cov('he_dgm1_2',int(bx),int(by),int(bx+B),int(by+B),'dgm1')
                except Exception as e:log('dgm1 block failed',e);continue
                with rasterio.open(io.BytesIO(b)) as d:
                    tmp=np.full((nrow,ncol),np.nan,np.float32)
                    reproject(rasterio.band(d,1),tmp,dst_transform=tr,dst_crs='EPSG:25832',resampling=Resampling.average,src_nodata=d.nodata,dst_nodata=np.nan)
                m=np.isfinite(tmp)&(tmp>-500)&(tmp<5000);acc[m]=np.round(tmp[m]*10).astype(np.int16);got+=1
        log('dgm1 blocks',got,'of',tried,'valid share',round(float((acc>-32768).mean()),3))
        if got:write_tif(os.path.join(OUT,'dem_course_3m_dm.tif'),acc[None],tr,CRS.from_epsg(25832),nodata=-32768);done('hessen_dgm1',blocks=got,tried=tried,file='dem_course_3m_dm.tif',units='decimetres',level='he_dgm1_2')
        else:failed('hessen_dgm1','no DGM1 block answered (%d tried)'%tried)
    except Exception as e:failed('hessen_dgm1',e);traceback.print_exc()
    if spec.get('skip_ndsm'):return
    # surface minus terrain at 1 m, 800 m cells within 450 m of the course, nearest the buildings first
    try:
        import tempfile,shutil
        from concurrent.futures import ThreadPoolExecutor,as_completed
        corr=450;cell=800;X=[];Y=[]
        xs,ys=T.transform(LON,LAT)
        for i in range(len(xs)-1):X+=list(np.linspace(xs[i],xs[i+1],6,endpoint=False));Y+=list(np.linspace(ys[i],ys[i+1],6,endpoint=False))
        from scipy.spatial import cKDTree
        tr_=cKDTree(np.c_[X,Y]);cells=[]
        for cx in np.arange(min(X)-corr,max(X)+corr,cell):
            for cy in np.arange(min(Y)-corr,max(Y)+corr,cell):
                if tr_.query([cx+cell/2,cy+cell/2])[0]<corr+cell*0.71:cells.append((int(cx),int(cy)))
        acc_=NdsmAcc(corr_m=corr)
        cnt={}
        for f in acc_.blds:
            c=f['geometry']['coordinates']
            while isinstance(c[0][0],list):c=c[0]
            bx,by=T.transform(c[0][0],c[0][1]);k=(int((bx-min(X)+corr)//cell),int((by-min(Y)+corr)//cell));cnt[k]=cnt.get(k,0)+1
        cells.sort(key=lambda c:-cnt.get((int((c[0]-min(X)+corr)//cell),int((c[1]-min(Y)+corr)//cell)),0))
        mx=int(os.environ.get('LIDAR_MAX_CELLS','0') or 0)
        if mx:cells=cells[:mx]
        log('hessen surface cells',len(cells))
        tmp=tempfile.mkdtemp();t0=time.time();budget=float(os.environ.get('LIDAR_BUDGET_MIN','115'))*60;ERR=[]
        def one(k):
            if time.time()-t0>budget:return None
            cx,cy=cells[k]
            try:
                a=_he_cov('dom1',cx-10,cy-10,cx+cell+10,cy+cell+10,'dom1');g=_he_cov('he_dgm1',cx-10,cy-10,cx+cell+10,cy+cell+10,'dgm1')
            except Exception as e:
                if len(ERR)<3:ERR.append(repr(e)[:200])
                return None
            pa=os.path.join(tmp,'s%05d.tif'%k);pg=os.path.join(tmp,'g%05d.tif'%k);open(pa,'wb').write(a);open(pg,'wb').write(g);return (pa,pg)
        nread=0
        with ThreadPoolExecutor(int(os.environ.get('LIDAR_THREADS','6'))) as ex:
            futs=[ex.submit(one,k) for k in range(len(cells))]
            for fu in as_completed(futs):
                r=fu.result()
                if not r:continue
                nread+=1;acc_.add(r[0],r[1],1.0)
                for q in r:
                    try:os.remove(q)
                    except OSError:pass
                if nread%10==0:
                    nb,nt,tot=acc_.write();done('hessen_ndsm',partial=True,cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,minutes=round((time.time()-t0)/60,1));save_report()
        shutil.rmtree(tmp,ignore_errors=True)
        if not nread:return failed('hessen_ndsm','no surface cells read: '+' | '.join(ERR))
        nb,nt,tot=acc_.write()
        done('hessen_ndsm',cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,tile_stats=acc_.stats,minutes=round((time.time()-t0)/60,1),stopped_on_budget=time.time()-t0>budget,errors=ERR)
    except Exception as e:failed('hessen_ndsm',e);traceback.print_exc()
def do_hessen_box():
    """The DOP20 WMS draws nothing at 8 m (it is scale limited), so the box comes from the WCS overview level he_dop20_32 in 4 km blocks."""
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326','EPSG:25832',always_xy=True)
    x0,y0=T.transform(W,S);x1,y1=T.transform(E,N);x0=min(x0,T.transform(W,N)[0]);x1=max(x1,T.transform(E,S)[0])
    try:
        res_m=8;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);dst=np.zeros((3,nrow,ncol),np.uint8);tr=from_origin(x0,y1,res_m,res_m);n=0;B=4000
        for bx in np.arange(x0,x1,B):
            for by in np.arange(y0,y1,B):
                b=_he_cov('he_dop20_32',int(bx),int(by),int(min(bx+B,x1)),int(min(by+B,y1)),'dop20')
                with rasterio.open(io.BytesIO(b)) as d:
                    if n==0:log('dop20_32 block',d.count,'bands',d.dtypes[0],d.res,d.width,d.height)
                    for k in range(3):
                        tmp=np.zeros((nrow,ncol),np.uint8)
                        reproject(rasterio.band(d,k+1),tmp,dst_transform=tr,dst_crs='EPSG:25832',resampling=Resampling.average,src_nodata=0,dst_nodata=0)
                        m=tmp>0;dst[k][m]=tmp[m]
                n+=1
        write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,tr,CRS.from_epsg(25832))
        done('hessen_dop20_box',blocks=n,res_m=8,source='WCS he_dop20_32',nonblack=round(float((dst.max(0)>0).mean()),3))
    except Exception as e:failed('hessen_dop20_box',e);traceback.print_exc()
# ---------------- Wales: Welsh Government LiDAR (OGL v3) and Sentinel-2 colour (Copernicus, free and open) ----------------
WALES_COG='/vsicurl/https://dmwproductionblob.blob.core.windows.net/cogs/lidar/wales_%s_32bit_cog.tif'
S2_STAC='https://earth-search.aws.element84.com/v1/search'
def _wales_read(kind,x0,y0,x1,y1,res):
    """A window of the whole-Wales 1 m DTM or DSM (EPSG:27700), averaged to res metres; NaN where there is no data (sea)."""
    from rasterio.windows import from_bounds as wfb
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_MAX_RETRY='4',GDAL_HTTP_RETRY_DELAY='2'):
        with rasterio.open(WALES_COG%kind) as d:
            w=wfb(x0,y0,x1,y1,d.transform);out=(max(1,int(round((y1-y0)/res))),max(1,int(round((x1-x0)/res))))
            a=d.read(1,window=w,out_shape=out,boundless=True,fill_value=-9999,resampling=Resampling.average).astype(np.float32)
    a[(a<-1000)|(a>2000)]=np.nan;return a
EA_WCS={'dtm':('https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs','13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m'),
        'dsm':('https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs','df4e3ec3-315e-48aa-aaaf-b5ae74d7b2bb__Lidar_Composite_Elevation_FZ_DSM_1m')}
def _ea_read(kind,x0,y0,x1,y1,res):
    """A window of the Environment Agency LIDAR composite (England, 1 m, Open Government Licence) through the Defra WCS, averaged to
    res metres; NaN where there is no data. Large windows are read in 1 km pieces."""
    url,cid=EA_WCS[kind];nx=max(1,int(round((x1-x0)/res)));ny=max(1,int(round((y1-y0)/res)));out=np.full((ny,nx),np.nan,np.float32);P=1000
    for px in np.arange(x0,x1,P):
        for py in np.arange(y0,y1,P):
            qx1,qy1=min(x1,px+P),min(y1,py+P)
            q=[('service','WCS'),('version','2.0.1'),('request','GetCoverage'),('coverageId',cid),('format','image/tiff'),('subset','E(%d,%d)'%(int(px),int(math.ceil(qx1)))),('subset','N(%d,%d)'%(int(py),int(math.ceil(qy1))))]
            r=None
            for k in range(4):
                try:
                    r=requests.get(url,params=q,headers=UA,timeout=180)
                    if r.status_code==200 and r.content[:2] in(b'II',b'MM'):break
                except requests.RequestException:r=None
                time.sleep(3*(k+1))
            if r is None or r.status_code!=200 or r.content[:2] not in(b'II',b'MM'):continue
            with rasterio.open(io.BytesIO(r.content)) as d:
                tmp=np.full((ny,nx),np.nan,np.float32)
                reproject(rasterio.band(d,1),tmp,src_nodata=d.nodata,dst_transform=from_origin(x0,y1,res,res),dst_crs=d.crs,resampling=Resampling.average if res>1 else Resampling.nearest,dst_nodata=np.nan)
            m=np.isfinite(tmp)&(tmp>-1000)&(tmp<2000);out[m]=tmp[m]
    return out
def _s2_scenes(bbox4326,months=(5,6,7,8,9),maxcc=10,n=8):
    """Least-cloudy Sentinel-2 L2A scenes (Earth Search STAC) over the box, summer months first."""
    q={'collections':['sentinel-2-l2a'],'bbox':[float(v) for v in bbox4326],'datetime':'2023-01-01T00:00:00Z/2026-12-31T23:59:59Z','query':{'eo:cloud_cover':{'lt':maxcc}},'limit':100}
    r=requests.post(S2_STAC,json=q,headers=UA,timeout=180);j=r.json() if r.headers.get('content-type','').startswith('application') else {}
    F=j.get('features',[])
    if not F:log('s2 search returned nothing',r.status_code,r.text[:300])
    F=[f for f in F if int(f['properties']['datetime'][5:7]) in months] or F
    F.sort(key=lambda f:f['properties'].get('eo:cloud_cover',99));return F[:n]
S2_ERRS=[]
def _s2_mosaic(scenes,dst_crs,tr,shape):
    """Reproject the true-colour (TCI) assets of the scenes onto one grid, filling gaps scene by scene (no-data is 0 in TCI)."""
    dst=np.zeros((3,)+shape,np.uint8);used=[]
    for f in scenes:
        href=(f['assets'].get('visual') or {}).get('href')
        if not href:S2_ERRS.append('%s: no visual asset'%f['id']);continue
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',AWS_NO_SIGN_REQUEST='YES',GDAL_HTTP_MAX_RETRY='4',GDAL_HTTP_RETRY_DELAY='2'),rasterio.open(href) as d:
                # read only the window that covers the target grid (a whole TCI tile is about 360 megapixels)
                from rasterio.windows import from_bounds as wfb
                db=rasterio.transform.array_bounds(shape[0],shape[1],tr);sb=transform_bounds(dst_crs,d.crs,db[0],db[1],db[2],db[3],densify_pts=21)
                w=wfb(*sb,d.transform).round_offsets().round_lengths();w=w.intersection(rasterio.windows.Window(0,0,d.width,d.height))
                if w.width<1 or w.height<1:continue
                src=d.read([1,2,3],window=w);stf=d.window_transform(w);scrs=d.crs
            tmp=np.zeros((3,)+shape,np.uint8)
            for b in range(3):reproject(src[b],tmp[b],src_transform=stf,src_crs=scrs,dst_transform=tr,dst_crs=dst_crs,resampling=Resampling.bilinear,src_nodata=0,dst_nodata=0)
        except Exception as e:log('s2 scene failed',f['id'],e);S2_ERRS.append('%s: %r'%(f['id'],e));continue
        if not (tmp.max(0)>0).any():S2_ERRS.append('%s: window read but nothing landed on the grid'%f['id'])
        m=(dst.max(0)==0)&(tmp.max(0)>0)
        if m.any():dst[:,m]=tmp[:,m];used.append(f['id'])
        if (dst.max(0)>0).mean()>0.995:break
    return dst,used
def _sharpen(rgb,dsm,dtm):
    """1 m imagery from 10 m Sentinel-2 colour and the 1 m LiDAR surface model: the hillshade of the surface model draws streets,
    roofs, hedges and crowns; tall rough surfaces (trees) and tall smooth ones (roofs) are tinted. Sea (no LiDAR) keeps the colour."""
    from scipy.ndimage import uniform_filter,binary_opening
    sea=~np.isfinite(dsm)|~np.isfinite(dtm);dsm=np.where(sea,0,dsm);dtm=np.where(sea,0,dtm)
    gy,gx=np.gradient(dsm);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gx,gy)
    hs=np.clip(np.sin(np.radians(45))*np.cos(sl)+np.cos(np.radians(45))*np.sin(sl)*np.cos(np.radians(315)-asp),0,1)
    nd=np.clip(dsm-dtm,0,40);rough=uniform_filter(np.abs(dsm-uniform_filter(dsm,3)),5);tall=nd>2.5
    tree=binary_opening(tall&(rough>0.35));bld=binary_opening(tall&(rough<=0.35))&~tree
    up=rgb.astype(np.float32);out=up*(0.6+0.6*hs)[None]
    g=np.array([46,70,36],np.float32)[:,None]
    out[:,tree]=out[:,tree]*0.5+g*(0.5+0.5*hs[tree][None])*0.5
    out[:,bld]=out[:,bld]*0.6+np.array([150,140,135],np.float32)[:,None]*(0.55+0.6*hs[bld][None])*0.4
    out[:,sea]=up[:,sea];return np.clip(out,0,255).astype(np.uint8)
def do_uk(read,tag):
    """Welsh recipe, every source free (including commercial use): terrain and surface from the Welsh Government LiDAR 2020-23
    (1 m DTM and DSM, Open Government Licence v3, whole-Wales Cloud Optimised GeoTIFFs on DataMapWales); colour from Sentinel-2
    (Copernicus, 10 m), sharpened to 1 m in the close-ups with the LiDAR surface model; buildings and trees from DSM minus DTM."""
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326','EPSG:27700',always_xy=True);Ti=Transformer.from_crs('EPSG:27700','EPSG:4326',always_xy=True)
    x0,y0=T.transform(W,S);x1,y1=T.transform(E,N);x0=min(x0,T.transform(W,N)[0]);x1=max(x1,T.transform(E,S)[0])
    crs=CRS.from_epsg(27700)
    # colour: Sentinel-2 box at 8 m
    scenes=[]
    try:
        scenes=_s2_scenes((W,S,E,N));res_m=8;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);tr=from_origin(x0,y1,res_m,res_m)
        dst,used=_s2_mosaic(scenes,crs,tr,(nrow,ncol))
        if not used:raise RuntimeError('no Sentinel-2 scene covered the box (%d found): %s'%(len(scenes),' | '.join(S2_ERRS[:4])[:900]))
        write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,tr,crs);done('s2_box',scenes=used,res_m=8,cover=round(float((dst.max(0)>0).mean()),3))
    except Exception as e:failed('s2_box',e);traceback.print_exc()
    # close-ups at 1 m: Sentinel-2 colour sharpened with the LiDAR surface
    if 'spots' not in ONLY_HE_SKIP and scenes:
        try:
            k=0;covs=[]
            for la,lo,sz,nm in spots():
                cx,cy=T.transform(lo,la);h=sz*500;n=int(sz*1000);tr=from_origin(cx-h,cy+h,1,1)
                rgb,_=_s2_mosaic(scenes,crs,tr,(n,n));dsm=read('dsm',cx-h,cy-h,cx+h,cy+h,1);dtm=read('dtm',cx-h,cy-h,cx+h,cy+h,1)
                cov=float((rgb.max(0)>0).mean());write_tif(os.path.join(OUT,'ortho_spot_%d.tif'%k),_sharpen(rgb,dsm,dtm),tr,crs);k+=1;log('spot',k,nm,'colour cover',cov);covs.append(round(cov,3))
            done(tag+'_spots',spots=k,res_m=1,method='Sentinel-2 colour sharpened with the LiDAR surface model',colour_cover=covs,errors=S2_ERRS[:4])
        except Exception as e:failed(tag+'_spots',e);traceback.print_exc()
    # terrain: 3 m in decimetres within 2.9 km of the course, in 2 km blocks
    try:
        res_m=3;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);acc=np.full((nrow,ncol),-32768,np.int16);B=2000;got=tried=0;tr=from_origin(x0,y1,res_m,res_m)
        for bx in np.arange(x0,x1,B):
            for by in np.arange(y0,y1,B):
                lo,la=Ti.transform(bx+B/2,by+B/2)
                if dist_km(la,lo)>1.5+1.42:continue
                tried+=1
                try:a=read('dtm',bx,by,bx+B,by+B,res_m)
                except Exception as e:log('dtm block failed',e);continue
                r0=int(round((y1-(by+B))/res_m));c0=int(round((bx-x0)/res_m));h,w=a.shape;r1=min(nrow,r0+h);c1=min(ncol,c0+w)
                if r1<=max(0,r0) or c1<=max(0,c0):continue
                sub=a[max(0,-r0):max(0,-r0)+r1-max(0,r0),max(0,-c0):max(0,-c0)+c1-max(0,c0)];m=np.isfinite(sub)
                blk=acc[max(0,r0):r1,max(0,c0):c1];blk[m]=np.round(sub[m]*10).astype(np.int16);got+=1
        if got:write_tif(os.path.join(OUT,'dem_course_3m_dm.tif'),acc[None],tr,crs,nodata=-32768);done(tag+'_dtm',blocks=got,tried=tried,file='dem_course_3m_dm.tif',units='decimetres',valid=round(float((acc>-32768).mean()),3))
        else:failed(tag+'_dtm','no DTM block read (%d tried)'%tried)
    except Exception as e:failed(tag+'_dtm',e);traceback.print_exc()
    if spec.get('skip_ndsm'):return
    # surface minus terrain at 1 m, 800 m cells within 450 m of the course, nearest the buildings first
    try:
        import tempfile,shutil
        from concurrent.futures import as_completed
        from scipy.spatial import cKDTree
        corr=450;cell=800;X=[];Y=[]
        xs,ys=T.transform(LON,LAT)
        for i in range(len(xs)-1):X+=list(np.linspace(xs[i],xs[i+1],6,endpoint=False));Y+=list(np.linspace(ys[i],ys[i+1],6,endpoint=False))
        tr_=cKDTree(np.c_[X,Y]);cells=[]
        for cx in np.arange(min(X)-corr,max(X)+corr,cell):
            for cy in np.arange(min(Y)-corr,max(Y)+corr,cell):
                if tr_.query([cx+cell/2,cy+cell/2])[0]<corr+cell*0.71:cells.append((int(cx),int(cy)))
        acc_=NdsmAcc(corr_m=corr);cnt={}
        for f in acc_.blds:
            c=f['geometry']['coordinates']
            while isinstance(c[0][0],list):c=c[0]
            bx,by=T.transform(c[0][0],c[0][1]);k=(int((bx-min(X)+corr)//cell),int((by-min(Y)+corr)//cell));cnt[k]=cnt.get(k,0)+1
        cells.sort(key=lambda c:-cnt.get((int((c[0]-min(X)+corr)//cell),int((c[1]-min(Y)+corr)//cell)),0))
        mx=int(os.environ.get('LIDAR_MAX_CELLS','0') or 0)
        if mx:cells=cells[:mx]
        log(tag,'surface cells',len(cells))
        tmp=tempfile.mkdtemp();t0=time.time();budget=float(os.environ.get('LIDAR_BUDGET_MIN','115'))*60;ERR=[]
        def one(k):
            if time.time()-t0>budget:return None
            cx,cy=cells[k];a0,b0,a1,b1=cx-10,cy-10,cx+cell+10,cy+cell+10
            try:
                ds=read('dsm',a0,b0,a1,b1,1);dt=read('dtm',a0,b0,a1,b1,1)
            except Exception as e:
                if len(ERR)<3:ERR.append(repr(e)[:200])
                return None
            if not np.isfinite(ds).any():return None
            trc=from_origin(a0,b1,1,1);pa=os.path.join(tmp,'s%05d.tif'%k);pg=os.path.join(tmp,'g%05d.tif'%k)
            write_tif(pa,np.nan_to_num(ds,nan=-9999)[None],trc,crs,nodata=-9999);write_tif(pg,np.nan_to_num(dt,nan=-9999)[None],trc,crs,nodata=-9999);return (pa,pg)
        nread=0
        with ThreadPoolExecutor(int(os.environ.get('LIDAR_THREADS','6'))) as ex:
            futs=[ex.submit(one,k) for k in range(len(cells))]
            for fu in as_completed(futs):
                r=fu.result()
                if not r:continue
                nread+=1;acc_.add(r[0],r[1],1.0)
                for q in r:
                    try:os.remove(q)
                    except OSError:pass
                if nread%10==0:
                    nb,nt,tot=acc_.write();done(tag+'_ndsm',partial=True,cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,minutes=round((time.time()-t0)/60,1));save_report()
        shutil.rmtree(tmp,ignore_errors=True)
        if not nread:return failed(tag+'_ndsm','no surface cells read: '+' | '.join(ERR))
        nb,nt,tot=acc_.write()
        done(tag+'_ndsm',cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,tile_stats=acc_.stats,minutes=round((time.time()-t0)/60,1),stopped_on_budget=time.time()-t0>budget,errors=ERR)
    except Exception as e:failed(tag+'_ndsm',e);traceback.print_exc()
def do_wales():do_uk(_wales_read,'wales')
def do_england():do_uk(_ea_read,'england')
ONLY_HE_SKIP=[x for x in os.environ.get('HE_SKIP','').split(',') if x]
def do_us_spots():
    """Only the 0.8 m NAIP spots whose files are missing (spots added to the course file after the full run)."""
    from pyproj import Transformer
    epsg=int(spec.get('lidar_utm','EPSG:32615').split(':')[1]);T=Transformer.from_crs('EPSG:4326','EPSG:%d'%epsg,always_xy=True);new=0
    for k,(la,lo,sz,nm) in enumerate(spots()):
        fn=os.path.join(OUT,'ortho_spot_%d.tif'%k)
        if os.path.exists(fn):continue
        cx,cy=T.transform(lo,la);px=min(4000,int(sz*1000/0.8))
        im=np.asarray(Image.open(io.BytesIO(_tnm(TNM_NAIP,(cx-sz*500,cy-sz*500,cx+sz*500,cy+sz*500),epsg,px,px,'jpg'))).convert('RGB'),np.uint8).transpose(2,0,1)
        write_tif(fn,im,from_origin(cx-sz*500,cy+sz*500,sz*1000/im.shape[2],sz*1000/im.shape[1]),CRS.from_epsg(epsg));new+=1;log('spot',k,nm)
    done('us_spots',new=new,spots=len(spots()))
def do_hessen_spots():
    """Only the 0.8 m imagery spots whose files are missing (spots added to the course file after the full run)."""
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326','EPSG:25832',always_xy=True);k=new=0
    for k,(la,lo,sz,nm) in enumerate(spots()):
        fn=os.path.join(OUT,'ortho_spot_%d.tif'%k)
        if os.path.exists(fn):continue
        cx,cy=T.transform(lo,la);px=min(3000,int(sz*1000/0.8))
        r=get(HE_WMS,params=dict(SERVICE='WMS',VERSION='1.3.0',REQUEST='GetMap',LAYERS='he_dop20_rgb',STYLES='',CRS='EPSG:25832',BBOX='%f,%f,%f,%f'%(cx-sz*500,cy-sz*500,cx+sz*500,cy+sz*500),WIDTH=px,HEIGHT=px,FORMAT='image/jpeg'),timeout=240)
        if not r.headers.get('content-type','').startswith('image'):raise RuntimeError('WMS answered '+r.text[:160])
        im=np.asarray(Image.open(io.BytesIO(r.content)).convert('RGB'),np.uint8).transpose(2,0,1)
        write_tif(fn,im,from_origin(cx-sz*500,cy+sz*500,sz*1000/im.shape[2],sz*1000/im.shape[1]),CRS.from_epsg(25832));new+=1;log('spot',k,nm)
    done('hessen_spots',new=new,spots=len(spots()))

# ---------------- France: IGN Géoplateforme, all Licence Ouverte Etalab 2.0 (free, commercial use allowed with attribution) ----------------
FR_WMS='https://data.geopf.fr/wms-r/wms'
FR_MNT='IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93';FR_MNH='IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93';FR_ORTHO='ORTHOIMAGERY.ORTHOPHOTOS'
def _fr_map(layer,x0,y0,x1,y1,px,py,fmt):
    r=get(FR_WMS,params=dict(SERVICE='WMS',VERSION='1.3.0',REQUEST='GetMap',LAYERS=layer,STYLES='',CRS='EPSG:2154',BBOX='%f,%f,%f,%f'%(x0,y0,x1,y1),WIDTH=px,HEIGHT=py,FORMAT=fmt),timeout=240)
    ct=r.headers.get('content-type','')
    if 'xml' in ct or 'text' in ct or 'html' in ct:raise RuntimeError('WMS answered '+r.text[:160])
    return r.content
def _fr_grid(layer,x0,y0,x1,y1,px,py):
    """raw values (metres) as float32, from the 32-bit BIL output; nodata (below -1000) as NaN"""
    b=_fr_map(layer,x0,y0,x1,y1,px,py,'image/x-bil;bits=32')
    if len(b)!=px*py*4:raise RuntimeError('BIL size %d, expected %d'%(len(b),px*py*4))
    a=np.frombuffer(b,dtype='<f4').reshape(py,px).astype(np.float32);a[~np.isfinite(a)|(a<-50)|(a>9000)]=np.nan;return a
def _fr_rgb(x0,y0,x1,y1,px,py):
    return np.asarray(Image.open(io.BytesIO(_fr_map(FR_ORTHO,x0,y0,x1,y1,px,py,'image/jpeg'))).convert('RGB'),np.uint8).transpose(2,0,1)
def do_france():
    """BD ORTHO aerial photography (box at 8 m, spots at 0.8 m), LiDAR HD terrain (MNT) at 3 m within 2.9 km of the course in 3 km blocks
    (metres stored as decimetres), and building heights and tree crowns from the LiDAR HD height model (MNH) at 1 m in 800 m cells
    within 450 m of the course. All in Lambert-93 (EPSG:2154) through the Géoplateforme WMS-R."""
    from pyproj import Transformer
    T=Transformer.from_crs('EPSG:4326','EPSG:2154',always_xy=True);Ti=Transformer.from_crs('EPSG:2154','EPSG:4326',always_xy=True)
    x0,y0=T.transform(W,S);x1,y1=T.transform(E,N);x0=min(x0,T.transform(W,N)[0]);x1=max(x1,T.transform(E,S)[0]);y1=max(y1,T.transform(E,N)[1],T.transform(W,N)[1]);y0=min(y0,T.transform(E,S)[1])
    L93=CRS.from_epsg(2154)
    if 'imagery' not in ONLY_HE_SKIP:
        try:
            res_m=8;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);dst=np.zeros((3,nrow,ncol),np.uint8);n=0;CH=4000
            for r0 in range(0,nrow,CH):
                for c0 in range(0,ncol,CH):
                    h=min(CH,nrow-r0);w=min(CH,ncol-c0);xa=x0+c0*res_m;yb=y1-r0*res_m
                    dst[:,r0:r0+h,c0:c0+w]=_fr_rgb(xa,yb-h*res_m,xa+w*res_m,yb,w,h);n+=1
            write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,from_origin(x0,y1,res_m,res_m),L93)
            k=0
            for la,lo,sz,nm in spots():
                cx,cy=T.transform(lo,la);px=min(5000,int(sz*1000/0.8));im=_fr_rgb(cx-sz*500,cy-sz*500,cx+sz*500,cy+sz*500,px,px)
                write_tif(os.path.join(OUT,'ortho_spot_%d.tif'%k),im,from_origin(cx-sz*500,cy+sz*500,sz*1000/im.shape[2],sz*1000/im.shape[1]),L93);k+=1;log('spot',k,nm)
            done('france_bdortho',chunks=n,spots=k,res_m=8,nonblack=round(float((dst.max(0)>0).mean()),3))
        except Exception as e:failed('france_bdortho',e);traceback.print_exc()
    save_report()
    if 'terrain' not in ONLY_HE_SKIP:
        try:
            res_m=3;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);acc=np.full((nrow,ncol),-32768,np.int16);B=3000;got=tried=0;miss=0
            for bx in np.arange(x0,x1,B):
                for by in np.arange(y0,y1,B):
                    lo,la=Ti.transform(bx+B/2,by+B/2)
                    if dist_km(la,lo)>1.5*1.42+2.9:continue
                    tried+=1
                    try:a=_fr_grid(FR_MNT,bx,by,bx+B,by+B,int(B/res_m),int(B/res_m))
                    except Exception as e:log('mnt block failed',e);continue
                    c0=int(round((bx-x0)/res_m));r0=int(round((y1-(by+B))/res_m));hh,ww=a.shape
                    rr0,cc0=max(0,r0),max(0,c0);rr1,cc1=min(nrow,r0+hh),min(ncol,c0+ww)
                    if rr1<=rr0 or cc1<=cc0:continue
                    sub=a[rr0-r0:rr1-r0,cc0-c0:cc1-c0];m=np.isfinite(sub);miss+=int((~m).sum())
                    tgt=acc[rr0:rr1,cc0:cc1];tgt[m]=np.round(sub[m]*10).astype(np.int16);got+=1
            log('mnt blocks',got,'of',tried,'valid share',round(float((acc>-32768).mean()),3))
            if got:write_tif(os.path.join(OUT,'dem_course_3m_dm.tif'),acc[None],from_origin(x0,y1,res_m,res_m),L93,nodata=-32768);done('france_mnt',blocks=got,tried=tried,file='dem_course_3m_dm.tif',units='decimetres',layer=FR_MNT,nodata_cells=miss)
            else:failed('france_mnt','no MNT block answered (%d tried)'%tried)
        except Exception as e:failed('france_mnt',e);traceback.print_exc()
        save_report()
    if spec.get('skip_ndsm'):return
    try:
        import tempfile,shutil
        from concurrent.futures import ThreadPoolExecutor,as_completed
        from scipy.spatial import cKDTree
        corr=450;cell=800;X=[];Y=[];xs,ys=T.transform(LON,LAT)
        for i in range(len(xs)-1):X+=list(np.linspace(xs[i],xs[i+1],6,endpoint=False));Y+=list(np.linspace(ys[i],ys[i+1],6,endpoint=False))
        tr_=cKDTree(np.c_[X,Y]);cells=[]
        for cx in np.arange(min(X)-corr,max(X)+corr,cell):
            for cy in np.arange(min(Y)-corr,max(Y)+corr,cell):
                if tr_.query([cx+cell/2,cy+cell/2])[0]<corr+cell*0.71:cells.append((int(cx),int(cy)))
        acc_=NdsmAcc(corr_m=corr);cnt={}
        for f in acc_.blds:
            c=f['geometry']['coordinates']
            while isinstance(c[0][0],list):c=c[0]
            bx,by=T.transform(c[0][0],c[0][1]);k=(int((bx-min(X)+corr)//cell),int((by-min(Y)+corr)//cell));cnt[k]=cnt.get(k,0)+1
        cells.sort(key=lambda c:-cnt.get((int((c[0]-min(X)+corr)//cell),int((c[1]-min(Y)+corr)//cell)),0))
        mx=int(os.environ.get('LIDAR_MAX_CELLS','0') or 0)
        if mx:cells=cells[:mx]
        log('france MNH cells',len(cells))
        tmp=tempfile.mkdtemp();t0=time.time();budget=float(os.environ.get('LIDAR_BUDGET_MIN','115'))*60;ERR=[]
        def one(k):
            if time.time()-t0>budget:return None
            cx,cy=cells[k];px=cell+20
            try:a=_fr_grid(FR_MNH,cx-10,cy-10,cx+cell+10,cy+cell+10,px,px)
            except Exception as e:
                if len(ERR)<3:ERR.append(repr(e)[:200])
                return None
            if not np.isfinite(a).any():return None
            pa=os.path.join(tmp,'h%05d.tif'%k);a2=np.where(np.isfinite(a),a,-9999).astype(np.float32)
            with rasterio.open(pa,'w',driver='GTiff',height=px,width=px,count=1,dtype='float32',crs=L93,transform=from_origin(cx-10,cy+cell+10,1.0,1.0),nodata=-9999) as d_:d_.write(a2[None])
            return pa
        nread=0
        with ThreadPoolExecutor(int(os.environ.get('LIDAR_THREADS','6'))) as ex:
            futs=[ex.submit(one,k) for k in range(len(cells))]
            for fu in as_completed(futs):
                r=fu.result()
                if not r:continue
                nread+=1;acc_.add(r,None,1.0)
                try:os.remove(r)
                except OSError:pass
                if nread%10==0:
                    nb,nt,tot=acc_.write();done('france_mnh',partial=True,cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,minutes=round((time.time()-t0)/60,1));save_report()
        shutil.rmtree(tmp,ignore_errors=True)
        if not nread:return failed('france_mnh','no height-model cells read: '+' | '.join(ERR))
        nb,nt,tot=acc_.write()
        done('france_mnh',cells=len(cells),cells_read=nread,buildings_with_height=nb,trees_kept=nt,trees_found=tot,tile_stats=acc_.stats,minutes=round((time.time()-t0)/60,1),stopped_on_budget=time.time()-t0>budget,errors=ERR)
    except Exception as e:failed('france_mnh',e);traceback.print_exc()

# ---------------- Europe: Copernicus dominant leaf type (10 m), so each measured tree gets broadleaf or conifer ----------------
def lonlat_3857(lon,lat):return lon*math.pi/180*R,math.log(math.tan(math.pi/4+math.radians(lat)/2))*R
DLT_URL='https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/GioLandPublic/HRL_DominanteLeafType_2018/ImageServer/exportImage'
def do_leaftype():
    """Copernicus HRL Dominant Leaf Type 2018, 10 m (free and open, credit the Copernicus Land Monitoring Service): 0 not tree,
    1 broadleaved, 2 coniferous, 254/255 outside. Fetched over the course box in Web Mercator chunks and written as one GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds
    t0=time.time();x0,y0=lonlat_3857(W,S);x1,y1=lonlat_3857(E,N);res=10.0;cw=2000*res
    nx=int(math.ceil((x1-x0)/res));ny=int(math.ceil((y1-y0)/res));A=np.full((ny,nx),255,np.uint8);nok=0;ERR=[]
    for cx in np.arange(x0,x1,cw):
        for cy in np.arange(y0,y1,cw):
            ex,ey=min(cx+cw,x1),min(cy+cw,y1);px,py=int(round((ex-cx)/res)),int(round((ey-cy)/res))
            if px<2 or py<2:continue
            try:
                r=get(DLT_URL,params=dict(bbox='%f,%f,%f,%f'%(cx,cy,ex,ey),bboxSR=3857,imageSR=3857,size='%d,%d'%(px,py),format='tiff',pixelType='U8',interpolation='RSP_NearestNeighbor',f='image'),timeout=240)
                with rasterio.MemoryFile(r.content) as mf:
                    with mf.open() as ds:a=ds.read(1)
                c0=int(round((cx-x0)/res));r0=int(round((y1-ey)/res));h,w=a.shape;A[r0:r0+h,c0:c0+w]=a[:ny-r0,:nx-c0];nok+=1
            except Exception as e:ERR.append(str(e)[:160])
    path=os.path.join(OUT,'leaftype_10m.tif')
    with rasterio.open(path,'w',driver='GTiff',width=nx,height=ny,count=1,dtype='uint8',crs='EPSG:3857',transform=from_bounds(x0,y0,x0+nx*res,y0+ny*res,nx,ny),compress='deflate') as ds:ds.write(A,1)
    v=A[A<3];cov={'not_tree':float((v==0).mean()) if v.size else 0,'broad':float((v==1).mean()) if v.size else 0,'conifer':float((v==2).mean()) if v.size else 0}
    if not nok:raise RuntimeError('no leaf-type chunks: '+'; '.join(ERR[:3]))
    done('leaftype',file='leaftype_10m.tif',chunks=nok,errors=ERR[:5],cover={k:round(x,3) for k,x in cov.items()},valid=round(float(v.size/A.size),3),minutes=round((time.time()-t0)/60,1),licence='Copernicus Land Monitoring Service, free and open with attribution')


# ---------------- OSM corridor detail ----------------
def do_osm_detail():
    """Everything a map builder places by hand within reach of the course, from Overpass in corridor queries ('around' the decimated course
    line, a few km at a time so no query is large): street lamps, benches, bins, bollards, signals, crossings, bus stops, shelters,
    fountains, barriers (fences, walls, hedges, guard rails), man_made, power lines and towers (400 m), trees with species, historic,
    artwork, playgrounds and pitches, and every building and building:part with all its tags (colour, material, roof shape, levels).
    Writes osm_detail.json (Overpass JSON, merged, de-duplicated)."""
    servers=['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter','https://overpass.private.coffee/api/interpreter']
    pts=[(float(a),float(b)) for a,b in spec['line']];out=[]
    for la,lo in pts:
        if not out or math.hypot((la-out[-1][0])*kz,(lo-out[-1][1])*kx)>=0.12:out.append((la,lo))
    if spec.get('swim'):out+=[(float(a),float(b)) for a,b in spec['swim'][::3]]
    chunks=[out[i:i+60] for i in range(0,len(out),55)]
    seen={};nq=0
    def q_for(ch,r=140):
        a='(around:%d,%s)'%(r,','.join('%.5f,%.5f'%p for p in ch));ap='(around:400,%s)'%','.join('%.5f,%.5f'%p for p in ch)
        return ('[out:json][timeout:180];(nwr[highway~"^(street_lamp|bus_stop|traffic_signals|crossing|stop|give_way|steps)$"]%s;'
                'nwr[amenity~"^(bench|waste_basket|shelter|fountain|bicycle_parking|drinking_water|toilets|clock|telephone|post_box|vending_machine)$"]%s;'
                'nwr[barrier]%s;nwr[man_made]%s;nwr[natural~"^(tree|tree_row|beach|cliff|rock|stone)$"]%s;nwr[historic]%s;'
                'nwr[tourism~"^(artwork|viewpoint|information|attraction)$"]%s;nwr[leisure~"^(playground|pitch|park|garden|marina|slipway|picnic_table|outdoor_seating|bandstand)$"]%s;'
                'nwr[building]%s;nwr["building:part"]%s;nwr[advertising]%s;nwr[power]%s;);out body;>;out skel qt;')%(a,a,a,a,a,a,a,a,a,a,a,ap)
    for ci,ch in enumerate(chunks):
        q=q_for(ch);ok=False
        for k in range(6):
            s_=servers[k%len(servers)]
            try:
                r=requests.post(s_,data={'data':q},headers=UA,timeout=240)
                if r.status_code!=200 or not r.text.lstrip().startswith('{'):time.sleep(8*(k+1));continue
                for e in r.json().get('elements',[]):
                    key=(e['type'],e['id'])
                    if key not in seen or ('tags' in e and 'tags' not in seen[key]):seen[key]=e
                ok=True;nq+=1;break
            except Exception as ex:log('osm_detail chunk',ci,'try',k,str(ex)[:120]);time.sleep(8*(k+1))
        if not ok:log('osm_detail chunk',ci,'gave up')
        time.sleep(2)
    els=list(seen.values());json.dump({'elements':els},open(os.path.join(OUT,'osm_detail.json'),'w'),separators=(',',':'))
    tagged=sum(1 for e in els if e.get('tags'))
    if nq==0:return failed('osm_detail','no Overpass server answered')
    done('osm_detail',chunks=len(chunks),answered=nq,elements=len(els),tagged=tagged,file='osm_detail.json')
# ---------------- run ----------------
def do_s2box():
    """Colour for any course outside the national recipes: a Sentinel-2 (Copernicus, free) mosaic of the box at 8 m in the local UTM zone."""
    from pyproj import Transformer
    lonc=(W+E)/2;latc=(S+N)/2;epsg=(32600 if latc>=0 else 32700)+int((lonc+180)//6)+1
    T=Transformer.from_crs('EPSG:4326','EPSG:%d'%epsg,always_xy=True)
    xs,ys=zip(*[T.transform(a,b) for a in(W,E) for b in(S,N)]);x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys);crs=CRS.from_epsg(epsg)
    try:
        months=(11,12,1,2,3) if latc<0 else (5,6,7,8,9)
        scenes=_s2_scenes((W,S,E,N),months=months);res_m=8;ncol=int((x1-x0)/res_m);nrow=int((y1-y0)/res_m);tr=from_origin(x0,y1,res_m,res_m)
        dst,used=_s2_mosaic(scenes,crs,tr,(nrow,ncol))
        if not used:raise RuntimeError('no Sentinel-2 scene covered the box (%d found): %s'%(len(scenes),' | '.join(S2_ERRS[:4])[:900]))
        write_tif(os.path.join(OUT,'ortho_box_8m.tif'),dst,tr,crs);done('s2_box',scenes=used,res_m=8,epsg=epsg,cover=round(float((dst.max(0)>0).mean()),3))
    except Exception as e:failed('s2_box',e);traceback.print_exc()
def do_zenodo_meta():
    """Research: the licence, files and column header of candidate marathon results datasets on Zenodo (record ids in the spec's
    'zenodo' list, default 6959864), without downloading the whole files. Writes zenodo_meta.json."""
    out={}
    for rid in spec.get('zenodo') or [6959864]:
        try:
            m=requests.get('https://zenodo.org/api/records/%s'%rid,headers=UA,timeout=60).json();md=m.get('metadata',{})
            rec={'title':md.get('title'),'license':md.get('license'),'creators':[c.get('name') for c in md.get('creators',[])],'description':(md.get('description') or '')[:3000],'files':[]}
            for f in m.get('files',[]):
                link=f.get('links',{}).get('self');head=''
                try:
                    r=requests.get(link,headers=dict(UA,Range='bytes=0-20000'),timeout=60);head=r.text[:20000]
                except Exception as ex:head='ERR '+str(ex)[:100]
                rec['files'].append({'key':f.get('key'),'size':f.get('size'),'link':link,'head':head})
            out[str(rid)]=rec
        except Exception as ex:out[str(rid)]={'error':str(ex)[:300]}
    json.dump(out,open(os.path.join(OUT,'zenodo_meta.json'),'w'),indent=1);done('zenodo_meta',records=len(out))
def do_wx_replay():
    """Hourly weather the race page asks for (Open-Meteo archive URLs listed in the course spec under 'wx_urls'), saved as replay files
    data/<id>/wx/<sha1 of the URL>.json so a page can be tested offline with the real history. Research and testing only."""
    import hashlib
    urls=spec.get('wx_urls') or [];d=os.path.join(OUT,'wx');os.makedirs(d,exist_ok=True);ok=0
    for u in urls:
        for k in range(4):
            try:
                r=requests.get(u,headers=UA,timeout=90)
                if r.status_code==200 and r.text.lstrip()[:1] in '[{':
                    open(os.path.join(d,hashlib.sha1(u.encode()).hexdigest()+'.json'),'w').write(r.text);ok+=1;break
                time.sleep(6*(k+1))
            except Exception as ex:log('wx_replay',str(ex)[:120]);time.sleep(6*(k+1))
        time.sleep(1)
    if not ok:return failed('wx_replay','no URL answered')
    done('wx_replay',urls=len(urls),saved=ok)
def do_wx_audit():
    """Weather source audit (research only): hourly weather from several free models at each course's weather points and at the
    nearest weather stations, on past race days, so the planner's weather source can be chosen on evidence. Spec key 'wx_audit':
    {venues:{id:{tz,pts:[[lat,lon]..],days:[..]}}, archive_models:[..], hf_models:[..], station_km, max_stations}.
    Model data: Open-Meteo archive and historical-forecast APIs (research and testing only). Stations: NOAA ISD global hourly (NCEI) and the
    Iowa Environmental Mesonet ASOS/METAR archive."""
    import csv,io,datetime as dt
    A=spec['wx_audit'];V=A['venues'];d=os.path.join(OUT,'om');ds=os.path.join(OUT,'st');os.makedirs(d,exist_ok=True);os.makedirs(ds,exist_ok=True)
    rep={'om':{},'isd':{},'iem':{}}
    VAR='temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,shortwave_radiation,surface_pressure,pressure_msl'
    # 1. stations near each course (NOAA ISD station list), nearest first
    st={}
    try:
        h=get('https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv',timeout=180).text;rows=list(csv.DictReader(io.StringIO(h)))
        for ven,v in V.items():
            yrs=sorted({x[:4] for x in v['days']});c=[]
            for r in rows:
                try:la=float(r['LAT']);lo=float(r['LON'])
                except Exception:continue
                if not r['END'] or r['END'][:4]<yrs[0] or r['BEGIN'][:4]>yrs[0]:continue   # the station list's END date lags, so only require it to cover the first race year
                dk=min(math.hypot((la-p[0])*110.574,(lo-p[1])*111.32*math.cos(math.radians(p[0]))) for p in v['pts'])
                if dk<=A.get('station_km',45):c.append(dict(usaf=r['USAF'],wban=r['WBAN'],name=r['STATION NAME'],icao=r['ICAO'],lat=la,lon=lo,elev=r['ELEV(M)'],km=round(dk,1)))
            c.sort(key=lambda x:x['km']);st[ven]=c[:A.get('max_stations',6)]
        json.dump(st,open(os.path.join(OUT,'stations.json'),'w'),indent=1)
    except Exception as ex:log('stations',str(ex)[:200]);rep['stations_error']=str(ex)[:300]
    # 2. model data at course points + station points, one request per venue, day and model
    def om(url,q,key):
        for k in range(4):
            try:
                r=requests.get(url,params=q,headers=UA,timeout=120)
                if r.status_code==200:open(os.path.join(d,key+'.json'),'w').write(r.text);rep['om'][key]='ok';return
                if r.status_code==400:rep['om'][key]='400 '+r.text[:150];return
                time.sleep(8*(k+1))
            except Exception as ex:time.sleep(8*(k+1));rep['om'][key]='err '+str(ex)[:100]
    for ven,v in (V.items() if not A.get('skip_om') else []):
        pts=([] if A.get('om_station_only') else v['pts'])+[[s['lat'],s['lon']] for s in st.get(ven,[])]
        if not pts:continue
        for day in v['days']:
            base=dict(latitude=','.join('%.4f'%p[0] for p in pts),longitude=','.join('%.4f'%p[1] for p in pts),start_date=day,end_date=day,hourly=VAR,wind_speed_unit='ms',timezone=v['tz'])
            for m in A.get('archive_models',[]):om('https://archive-api.open-meteo.com/v1/archive',dict(base,models=m),'%s_%s_ar_%s%s'%(ven,day,m,'_st' if A.get('om_station_only') else ''));time.sleep(0.7)
            for m in A.get('hf_models',[])+v.get('hf_extra',[]):om('https://historical-forecast-api.open-meteo.com/v1/forecast',dict(base,models=m),'%s_%s_hf_%s%s'%(ven,day,m,'_st' if A.get('om_station_only') else ''));time.sleep(0.7)
        save_report()
    if A.get('skip_obs'):
        json.dump(rep,open(os.path.join(OUT,'wx_audit_report_%s.json'%('st' if A.get('om_station_only') else 'om')),'w'),indent=1);return done('wx_audit',om=sum(1 for x in rep['om'].values() if x=='ok'))
    # 3. station observations on race days (and the day either side), raw rows only
    for ven,v in V.items():
        want=set()
        for day in v['days']:
            t=dt.date.fromisoformat(day);want|={(t+dt.timedelta(days=k)).isoformat() for k in (-1,0,1)}
        for s in st.get(ven,[]):
            sid=s['usaf']+s['wban']
            for y in sorted({x[:4] for x in want}):
                key='%s_%s_%s'%(ven,sid,y)
                try:
                    r=requests.get('https://noaa-global-hourly-pds.s3.amazonaws.com/%s/%s.csv'%(y,sid),headers=UA,timeout=180)   # NOAA Open Data on AWS (the NCEI page now answers with an HTML page)
                    if r.status_code!=200 or r.text.lstrip()[:1]=='<':r=requests.get('https://www.ncei.noaa.gov/data/global-hourly/access/%s/%s.csv'%(y,sid),headers=UA,timeout=180)
                    if r.text.lstrip()[:1]=='<':rep['isd'][key]='html';continue
                    if r.status_code!=200:rep['isd'][key]=r.status_code;continue
                    L=r.text.splitlines();keep=[L[0]]+[x for x in L[1:] if x.count(',')>1 and x.split(',',2)[1].strip('"')[:10] in want]
                    open(os.path.join(ds,'isd_%s.csv'%key),'w').write('\n'.join(keep));rep['isd'][key]=len(keep)-1
                except Exception as ex:rep['isd'][key]='err '+str(ex)[:100]
            if s.get('icao'):
                for day in v['days']:
                    t=dt.date.fromisoformat(day);a=t-dt.timedelta(days=1);b=t+dt.timedelta(days=2);key='%s_%s_%s'%(ven,s['icao'],day)
                    q='station=%s&data=tmpf&data=relh&data=sknt&data=drct&data=gust&data=alti&data=mslp&year1=%d&month1=%d&day1=%d&year2=%d&month2=%d&day2=%d&tz=Etc%%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=empty&trace=empty&report_type=3&report_type=4'%(s['icao'],a.year,a.month,a.day,b.year,b.month,b.day)
                    try:
                        r=requests.get('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'+q,headers=UA,timeout=120)
                        if r.status_code==200 and r.text.count('\n')>1:open(os.path.join(ds,'iem_%s.csv'%key),'w').write(r.text);rep['iem'][key]=r.text.count('\n')-1
                        else:rep['iem'][key]=r.status_code
                    except Exception as ex:rep['iem'][key]='err '+str(ex)[:100]
                    time.sleep(1)
    json.dump(rep,open(os.path.join(OUT,'wx_audit_report.json'),'w'),indent=1)
    done('wx_audit',om=sum(1 for x in rep['om'].values() if x=='ok'),om_total=len(rep['om']),isd=len(rep['isd']),iem=len(rep['iem']))
ONLY=[x for x in os.environ.get('ONLY','').split(',') if x]
def run():
    global ONLY
    if not ONLY and spec.get('jobs'):ONLY=list(spec['jobs'])
    if ONLY:   # re-run selected sources only, keeping the rest of the report
        try:REPORT['sources']=json.load(open(os.path.join(OUT,'report.json'))).get('sources',{})
        except Exception:pass
        F={'wx_audit':do_wx_audit,'zenodo_meta':do_zenodo_meta,'s2box':do_s2box,'wx_replay':do_wx_replay,'osm_detail':do_osm_detail,'leaftype':do_leaftype,'osm':do_osm,'terrain':do_terrain,'profile':do_profile,'nz':do_nz,'ndsm':do_nz_ndsm if spec.get('region')=='nz' else None,'bayern':do_bayern,'bayern_probe':do_bayern_probe if spec.get('region')=='bayern' else None,'lod2':do_bayern_lod2 if spec.get('region')=='bayern' else None,'us_lidar':do_us_lidar if spec.get('region')=='us' else None,'us':do_us if spec.get('region')=='us' else None,'hessen':do_hessen if spec.get('region')=='hessen' else None,'hessen_box':do_hessen_box if spec.get('region')=='hessen' else None,'hessen_spots':do_hessen_spots if spec.get('region')=='hessen' else None,'us_spots':do_us_spots if spec.get('region')=='us' else None,'wales':do_wales if spec.get('region')=='wales' else None,'england':do_england if spec.get('region')=='england' else None,'france':do_france if spec.get('region')=='france' else None}
        for k in ONLY:
            fn=F.get(k)
            if not fn:continue
            try:fn()
            except Exception as e:failed(k,e);traceback.print_exc()
            save_report()
        REPORT['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save_report();return
    for name,fn in(('terrain',do_terrain),('osm',do_osm),('osm_detail',do_osm_detail),('profile',do_profile)):
        try:fn()
        except Exception as e:failed(name,e);traceback.print_exc()
        save_report()
    try:
        if spec.get('region')=='nz':do_nz()
        elif spec.get('region')=='bayern':do_bayern()
        elif spec.get('region')=='hessen':do_hessen()
        elif spec.get('region')=='wales':do_wales()
        elif spec.get('region')=='england':do_england()
        elif spec.get('region')=='france':do_france()
        elif spec.get('region')=='us':
            do_us();save_report()
            if not spec.get('skip_lidar'):do_us_lidar()
    except Exception as e:failed(spec.get('region','region'),e);traceback.print_exc()
    if spec.get('region') in('bayern','hessen','wales','england','france') or spec.get('leaftype'):
        try:do_leaftype()
        except Exception as e:failed('leaftype',e);traceback.print_exc()
    REPORT['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save_report()
    tot=sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(OUT) for f in fs)
    log('total MB',round(tot/1e6,1));print(json.dumps(REPORT,indent=1))
if __name__=='__main__':run()
