import io, math, zipfile, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import requests, rasterio, ezdxf, contextily as ctx
from shapely.geometry import Polygon, LineString, box, mapping
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer
from scipy.ndimage import gaussian_filter, distance_transform_edt
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title='Curvas de Nível para Irrigação', page_icon='🌱', layout='wide')
st.title('🌱 Projeto de irrigação — Satélite + Curvas de nível + DXF')
st.write('Envie o polígono da propriedade em KML/KMZ e gere um DXF com imagem de satélite incorporada, perímetro, curvas de nível e cotas.')
st.info('As curvas de 1 m são interpoladas de um modelo digital de elevação de aproximadamente 30 m. Não substituem levantamento topográfico para dimensionamento executivo.')


def get_polygon(blob, name):
    if name.lower().endswith('.kmz'):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = [x for x in z.namelist() if x.lower().endswith('.kml')]
            if not names: raise ValueError('O KMZ não contém KML.')
            blob = z.read(names[0])
    root = ET.fromstring(blob)
    polygons = []
    for elem in root.iter():
        if elem.tag.split('}')[-1] != 'Polygon': continue
        outer = next((e for e in elem.iter() if e.tag.split('}')[-1] == 'outerBoundaryIs'), None)
        if outer is None: continue
        coords = next((e for e in outer.iter() if e.tag.split('}')[-1] == 'coordinates'), None)
        if coords is None or not coords.text: continue
        pts = []
        for item in coords.text.split():
            fields = item.split(',')
            if len(fields) >= 2: pts.append((float(fields[0]), float(fields[1])))
        if len(pts) >= 3:
            p = Polygon(pts)
            if not p.is_valid: p = p.buffer(0)
            if not p.is_empty: polygons.append(p)
    if not polygons: raise ValueError('Nenhum polígono válido encontrado no KML/KMZ.')
    return unary_union(polygons)


def is_placeholder(a):
    a = np.asarray(a)[..., :3].astype(np.float32)
    a = a[::max(1,a.shape[0]//250), ::max(1,a.shape[1]//250)]
    sd = np.std(a.reshape(-1,3),axis=0)
    gray = np.mean((np.max(a,axis=2)-np.min(a,axis=2)) < 5)
    return bool(np.max(sd)<18 or (gray>0.94 and np.max(sd)<42))


def run(kml_bytes, filename, api_key, resolution, max_image):
    geo = get_polygon(kml_bytes, filename)
    center = geo.centroid
    zone = max(1,min(60,int((center.x+180)/6)+1))
    epsg = (32700 if center.y<0 else 32600)+zone
    utm = Transformer.from_crs(4326,epsg,always_xy=True)
    wgs = Transformer.from_crs(epsg,4326,always_xy=True)
    poly = shp_transform(utm.transform,geo)
    polygons = [poly] if poly.geom_type=='Polygon' else list(poly.geoms)
    xmin,ymin,xmax,ymax = poly.bounds
    xmin-=40; ymin-=40; xmax+=40; ymax+=40
    width=xmax-xmin; height=ymax-ymin
    if max(width,height)>3000 or poly.area>3_000_000:
        raise ValueError('Esta versão gratuita está limitada a terrenos com até 300 ha e dimensão máxima de 3 km. Divida o polígono em áreas menores.')
    factor=min(2.0,max_image/max(width,height))
    iw=max(64,round(width*factor)); ih=max(64,round(height*factor))
    sat_transform=from_bounds(xmin,ymin,xmax,ymax,iw,ih)
    merc=Transformer.from_crs(epsg,3857,always_xy=True)
    corners=[merc.transform(x,y) for x,y in [(xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)]]
    mx0=min(x for x,y in corners); mx1=max(x for x,y in corners)
    my0=min(y for x,y in corners); my1=max(y for x,y in corners)
    satellite=None
    for zoom in [18,17,16,15]:
        try:
            src_img, extent=ctx.bounds2img(mx0,my0,mx1,my1,zoom=zoom,source=ctx.providers.Esri.WorldImagery,ll=False,n_connections=1,max_retries=1,wait=1)
            if is_placeholder(src_img): continue
            ex0,ex1,ey0,ey1=extent
            rgb=np.asarray(src_img)[...,:3].astype(np.uint8)
            source_transform=from_bounds(ex0,ey0,ex1,ey1,rgb.shape[1],rgb.shape[0])
            dest=np.zeros((ih,iw,3),dtype=np.uint8)
            for b in range(3):
                reproject(source=rgb[:,:,b],destination=dest[:,:,b],src_transform=source_transform,src_crs='EPSG:3857',dst_transform=sat_transform,dst_crs=f'EPSG:{epsg}',resampling=Resampling.bilinear)
            if not is_placeholder(dest): satellite=dest; break
        except Exception: continue
    if satellite is None:
        raise RuntimeError('O provedor de satélite não entregou uma imagem válida. Tente novamente mais tarde.')

    dem_area=shp_transform(wgs.transform,box(xmin-120,ymin-120,xmax+120,ymax+120))
    west,south,east,north=dem_area.bounds
    resp=requests.get('https://portal.opentopography.org/API/globaldem',params={'demtype':'COP30','south':south,'north':north,'west':west,'east':east,'outputFormat':'GTiff','API_Key':api_key},timeout=180)
    resp.raise_for_status()
    if resp.content[:4] not in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+'):
        raise RuntimeError('OpenTopography não retornou GeoTIFF. Confira a chave e a cota de uso da API.')
    nx=max(5,math.ceil(width/3)); ny=max(5,math.ceil(height/3))
    if nx*ny>2_000_000: raise ValueError('Terreno extenso para a grade do servidor gratuito.')
    grid_transform=from_bounds(xmin,ymin,xmax,ymax,nx,ny)
    z=np.full((ny,nx),-9999,dtype=np.float32)
    with rasterio.MemoryFile(resp.content) as mem:
        with mem.open() as src:
            reproject(source=rasterio.band(src,1),destination=z,src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,dst_transform=grid_transform,dst_crs=f'EPSG:{epsg}',dst_nodata=-9999,resampling=Resampling.bilinear)
    valid=np.isfinite(z)&(z!=-9999)
    if not valid.any(): raise RuntimeError('MDE sem altitudes válidas.')
    if not valid.all():
        idx=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        z=z[tuple(idx)]
    z=gaussian_filter(z,sigma=0.5)
    Z=z[::-1,:]
    xs=xmin+(np.arange(nx)+0.5)*width/nx
    ys=ymin+(np.arange(ny)+0.5)*height/ny
    XX,YY=np.meshgrid(xs,ys)
    mask=geometry_mask([mapping(poly)],out_shape=(ny,nx),transform=grid_transform,invert=True)[::-1,:]
    z_area=np.where(mask,Z,np.nan)
    if not np.isfinite(z_area).any(): raise RuntimeError('Sem altitudes dentro do polígono.')
    zmin=float(np.nanmin(z_area)); zmax=float(np.nanmax(z_area))
    i_min=np.unravel_index(np.nanargmin(z_area),z_area.shape)
    i_max=np.unravel_index(np.nanargmax(z_area),z_area.shape)
    pmin=(float(XX[i_min]),float(YY[i_min])); pmax=(float(XX[i_max]),float(YY[i_max]))
    levels=np.arange(math.ceil(zmin),zmax+1e-7,1.0)
    curves=[]
    fig,ax=plt.subplots()
    if len(levels):
        cs=ax.contour(XX,YY,Z,levels=levels)
        for level,segments in zip(cs.levels,cs.allsegs):
            for segment in segments:
                if len(segment)<2: continue
                clipped=LineString(segment).intersection(poly)
                if clipped.is_empty: continue
                parts=[clipped] if clipped.geom_type=='LineString' else [g for g in getattr(clipped,'geoms',[]) if g.geom_type=='LineString']
                for line in parts:
                    if line.length>1: curves.append((float(level),line))
    plt.close(fig)

    doc=ezdxf.new('R2010'); doc.header['$INSUNITS']=6
    msp=doc.modelspace()
    colors={'01_SATELITE':7,'02_PERIMETRO':1,'03_CURVAS_1M':8,'04_CURVAS_MESTRAS':3,'05_COTAS':2,'06_COTAS_EXTREMAS':5,'07_DESNIVEL':4,'08_DESENHAR_IRRIGACAO':3,'09_INFORMACOES':7}
    for name,color in colors.items(): doc.layers.new(name,dxfattribs={'color':color})
    # Imagem incorporada em SOLIDs, com desenho colorido no próprio DXF.
    im=Image.fromarray(satellite)
    ratio=resolution/max(im.size)
    pw=max(2,round(im.width*ratio)); ph=max(2,round(im.height*ratio))
    pixels=np.asarray(im.resize((pw,ph),Image.Resampling.LANCZOS))
    dx=width/pw; dy=height/ph
    for j in range(ph):
        y1=ymax-j*dy; y0=y1-dy
        for i in range(pw):
            x0=xmin+i*dx; x1=x0+dx
            r,g,b=map(int,pixels[j,i])
            rgb=(r<<16)|(g<<8)|b
            msp.add_solid([(x0,y0,0),(x1,y0,0),(x0,y1,0),(x1,y1,0)],dxfattribs={'layer':'01_SATELITE','true_color':rgb})
    for p in polygons:
        msp.add_lwpolyline(list(p.exterior.coords),close=True,dxfattribs={'layer':'02_PERIMETRO','lineweight':50})
    for elev,line in curves:
        master=abs(elev/5-round(elev/5))<1e-6
        layer='04_CURVAS_MESTRAS' if master else '03_CURVAS_1M'
        msp.add_polyline3d([(x,y,elev) for x,y in line.coords],dxfattribs={'layer':layer,'lineweight':35 if master else 13})
        if line.length<12: continue
        count=max(1,math.ceil(line.length/45))
        for k in range(count):
            d=(k+0.5)*line.length/count
            p=line.interpolate(d); a=line.interpolate(max(0,d-1)); b=line.interpolate(min(line.length,d+1))
            angle=math.degrees(math.atan2(b.y-a.y,b.x-a.x))
            if angle>90: angle-=180
            if angle<-90: angle+=180
            msp.add_text(f'{elev:.0f}',dxfattribs={'layer':'05_COTAS','height':1.5,'rotation':angle,'insert':(p.x,p.y,0)})
    for pt,elev,label in [(pmax,zmax,'MAXIMA'),(pmin,zmin,'MINIMA')]:
        msp.add_circle((*pt,0),2,dxfattribs={'layer':'06_COTAS_EXTREMAS'})
        msp.add_text(f'{label}: {elev:.2f} m',dxfattribs={'layer':'06_COTAS_EXTREMAS','height':1.8,'insert':(pt[0]+3,pt[1]+3,0)})
    if math.dist(pmax,pmin)>2:
        msp.add_line((*pmax,0),(*pmin,0),dxfattribs={'layer':'07_DESNIVEL'})
        vx=pmin[0]-pmax[0]; vy=pmin[1]-pmax[1]; length=math.hypot(vx,vy); vx/=length; vy/=length
        for sign in [-1,1]:
            ang=math.radians(25)
            axx=-vx*math.cos(ang)-sign*vy*math.sin(ang)
            ayy=-vy*math.cos(ang)+sign*vx*math.sin(ang)
            arrow=min(8,length*0.15)
            msp.add_line((*pmin,0),(pmin[0]+arrow*axx,pmin[1]+arrow*ayy,0),dxfattribs={'layer':'07_DESNIVEL'})
    notes=['BASE TOPOGRAFICA - IRRIGACAO',f'AREA: {poly.area/10000:.3f} ha',f'PERIMETRO: {poly.length:.2f} m',f'COTA MINIMA: {zmin:.2f} m',f'COTA MAXIMA: {zmax:.2f} m',f'DESNIVEL: {zmax-zmin:.2f} m','CURVAS: 1 m',f'UTM EPSG:{epsg}','MDE COPERNICUS GLO-30','PLANEJAMENTO PRELIMINAR']
    for k,note in enumerate(notes): msp.add_text(note,dxfattribs={'layer':'09_INFORMACOES','height':1.8,'insert':(xmax+12,ymax-k*5,0)})
    dxf_stream=io.StringIO(); doc.write(dxf_stream)
    dxf_bytes=dxf_stream.getvalue().encode('utf-8')
    # Prévia para verificar o produto antes do download.
    fig,ax=plt.subplots(figsize=(11,8))
    ax.imshow(satellite,extent=(xmin,xmax,ymin,ymax),origin='upper')
    for elev,line in curves:
        pts=np.asarray(line.coords)
        master=abs(elev/5-round(elev/5))<1e-6
        ax.plot(pts[:,0],pts[:,1],color='yellow' if master else 'white',linewidth=1.2 if master else 0.5)
    for p in polygons:
        pts=np.asarray(p.exterior.coords)
        ax.plot(pts[:,0],pts[:,1],color='red',linewidth=2)
    ax.set_aspect('equal'); ax.set_title('Satélite + curvas de nível (1 m)')
    fig.tight_layout()
    preview=io.BytesIO(); fig.savefig(preview,format='png',dpi=150); plt.close(fig)
    sat_png=io.BytesIO(); Image.fromarray(satellite).save(sat_png,format='PNG')
    report={'area_ha':round(poly.area/10000,4),'perimetro_m':round(poly.length,2),'epsg':epsg,'cota_minima_m':round(zmin,2),'cota_maxima_m':round(zmax,2),'desnivel_m':round(zmax-zmin,2),'curvas':len(curves),'pixels_imagem_incorporada':pw*ph,'aviso':'MDE GLO-30: curvas de 1 m interpoladas; validar topografia em campo.'}
    zip_io=io.BytesIO()
    with zipfile.ZipFile(zip_io,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('IRRIGACAO_IMAGEM_INCORPORADA.dxf',dxf_bytes)
        zf.writestr('PREVIA.png',preview.getvalue())
        zf.writestr('SATELITE.png',sat_png.getvalue())
        zf.writestr('RELATORIO.json',json.dumps(report,indent=2,ensure_ascii=False))
    return zip_io.getvalue(),preview.getvalue(),report

upload=st.file_uploader('Arquivo da propriedade (KML ou KMZ)',type=['kml','kmz'])
col1,col2=st.columns(2)
with col1:
    resolution=st.select_slider('Resolução da imagem incorporada no DXF',options=[180,250,350,450,600,800],value=450)
with col2:
    max_image=st.select_slider('Resolução da imagem de satélite para prévia',options=[1200,1800,2400,3200,4000],value=2400)
st.caption('600 ou 800 pixels podem gerar DXFs muito grandes e exigir mais memória no servidor gratuito e no AutoCAD.')
try:
    api_key=st.secrets['OPENTOPOGRAPHY_API_KEY']
except Exception:
    api_key=None
if not api_key:
    api_key=st.text_input('Chave OpenTopography (somente se não estiver configurada no servidor)',type='password')

if st.button('🌍 Gerar projeto DXF',type='primary',disabled=(upload is None or not api_key)):
    try:
        with st.spinner('Baixando satélite, calculando curvas e incorporando imagem ao DXF. Aguarde...'):
            archive,preview,report=run(upload.getvalue(),upload.name,api_key,resolution,max_image)
        st.success('Projeto gerado com sucesso!')
        st.image(preview,caption='Prévia do projeto')
        a,b,c=st.columns(3)
        a.metric('Área (ha)',f"{report['area_ha']:.3f}")
        b.metric('Desnível (m)',f"{report['desnivel_m']:.2f}")
        c.metric('Curvas (segmentos)',str(report['curvas']))
        st.download_button('⬇️ Baixar DXF + imagem + relatório (ZIP)',data=archive,file_name='PROJETO_IRRIGACAO_DXF.zip',mime='application/zip',type='primary')
    except Exception as exc:
        st.error(f'Não foi possível gerar o projeto: {exc}')

st.caption('Uso preliminar para planejamento. O provedor de imagens pode impor restrições de uso e redistribuição; verifique sua licença antes de publicar ou comercializar os arquivos.')
