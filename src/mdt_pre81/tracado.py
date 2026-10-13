# -*- coding: utf-8 -*-
"""Traçado de CURSOS D'ÁGUA em fotos/imagens históricas (p&b): linhas, não manchas escuras.

Ideia: rio/córrego/mata ciliar = faixa ESCURA, FINA e CONTÍNUA, em forma de rede (árvore) que desemboca no rio principal.
Talhões escuros, sombras e manchas = escuros GROSSOS ou isolados ou em malha. Por isso:
 1. top-hat negro (fecha - imagem) com janela ~ largura máxima: mantém só escuro mais estreito que `largura_max_m`
    (talhões/manchas grandes somem);
 2. limiar com histerese + esqueleto (linha central);
 3. poda de esporões; descarta componentes curtos, em malha (muitos laços por km) ou não alongados (cercas, talhões);
 4. mantém só o que se liga (ponte de `ponte_m`) ao rio principal (semente = áreas NA_TD).
Saída: só LINHAS centrais (sem polígonos de 'água': não há lagoas na região).
"""
import os, shutil, warnings
import numpy as np
import geopandas as gpd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.enums import Resampling
from scipy import ndimage as ndi
from shapely.geometry import LineString
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize, disk

warnings.filterwarnings("ignore")
CRS_ALVO = "EPSG:31982"


def _skan(sk):
    from skan import Skeleton, summarize
    skel = Skeleton(sk)
    df = summarize(skel, separator="-") if "separator" in summarize.__code__.co_varnames else summarize(skel)
    df.columns = [c.replace("_", "-") for c in df.columns]
    return skel, df


def _podar(sk, esporao_px, voltas=2):
    for _ in range(voltas):
        if not sk.any():
            break
        skel, df = _skan(sk)
        curtos = df.index[(df["branch-type"] == 1) & (df["branch-distance"] < esporao_px)]
        if len(curtos) == 0:
            break
        for i in curtos:
            rc = skel.path_coordinates(i)
            sk[rc[:, 0].astype(int), rc[:, 1].astype(int)] = False
        sk = skeletonize(sk)
    return sk


def extrair_tracado(entrada, pasta, rotulo, mascara=None, semente=None, largura_max_m=250.0, comp_min_m=1500.0,
                    laco_max_por_km=1.2, elong_min=0.22, ponte_m=250.0, esporao_m=150.0, q_baixo=90, q_alto=97,
                    px_alvo_m=10.0):
    """Retorna GeoDataFrame (EPSG:31982) com as linhas centrais e grava drenagem_<rotulo>.gpkg/shp."""
    with rasterio.open(entrada) as src:
        px0 = abs(src.res[0])
        f = max(1, int(px_alvo_m // px0)) if px0 < px_alvo_m else 1
        H, W = src.height // f, src.width // f
        g = src.read(1, out_shape=(H, W), resampling=Resampling.average).astype("float32")
        tr = src.transform * rasterio.Affine.scale(src.width / W, src.height / H)
        crs, nd = src.crs, src.nodata
    px = px0 * f
    valid = np.ones(g.shape, bool) if nd is None else (g != nd)
    if mascara is not None:
        mg = gpd.GeoSeries([mascara], crs=CRS_ALVO).to_crs(crs).iloc[0]
        valid &= rasterize([(mg, 1)], out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    print(f"  traçado: pixel {px:.1f} m | busca em {valid.mean()*100:.0f}% da imagem")
    core = ndi.binary_erosion(valid, iterations=3)
    lo_, hi_ = np.percentile(g[core], (1, 99))
    gn = np.clip((g - lo_) / (hi_ - lo_ + 1e-9), 0, 1)
    gn[~valid] = np.median(gn[core])
    gn = ndi.gaussian_filter(gn, 0.8)
    k = int(2 * round(largura_max_m / px / 2) + 1)
    fecha = ndi.grey_closing(gn, size=(k, k))
    bth = ndi.gaussian_filter(fecha - gn, 0.8)       # só escuro mais ESTREITO que largura_max_m
    bth[~valid] = 0
    lo, hi = np.percentile(bth[core], (q_baixo, q_alto))
    cand = apply_hysteresis_threshold(bth, lo, hi) & valid
    cand = ndi.binary_closing(cand, structure=disk(1)) & valid
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    if n:
        tam = np.bincount(lab.ravel()); ok = tam >= max(8, int(2000 / (px * px))); ok[0] = False
        cand = ok[lab]
    sk = skeletonize(cand)
    sk = _podar(sk, esporao_m / px)
    if not sk.any():
        print("  ! nada detectado"); return _salvar([], crs, pasta, rotulo)
    skel, df = _skan(sk)
    skid = df["skeleton-id"].values
    comp_len = df.groupby("skeleton-id")["branch-distance"].sum() * px
    nos = df.groupby("skeleton-id").apply(lambda d: len(set(d["node-id-src"]) | set(d["node-id-dst"])))
    laços = (df.groupby("skeleton-id").size() - nos + 1).clip(lower=0)
    # elongação: diagonal da caixa / comprimento
    ext = {}
    for i in range(skel.n_paths):
        rc = skel.path_coordinates(i)
        a = ext.setdefault(skid[i], [1e9, 1e9, -1, -1])
        a[0] = min(a[0], rc[:, 0].min()); a[1] = min(a[1], rc[:, 1].min())
        a[2] = max(a[2], rc[:, 0].max()); a[3] = max(a[3], rc[:, 1].max())
    manter = set()
    desc = dict(curto=0, malha=0, curvo=0)
    for cid, L in comp_len.items():
        a = ext[cid]; diag = np.hypot(a[2] - a[0], a[3] - a[1]) * px
        if L < comp_min_m:
            desc["curto"] += 1
        elif laços[cid] / (L / 1000) > laco_max_por_km:
            desc["malha"] += 1
        elif diag / L < elong_min:
            desc["curvo"] += 1
        else:
            manter.add(cid)
    print(f"  componentes: {len(comp_len)} | mantidos {len(manter)} | descartados: {desc['curto']} curtos, "
          f"{desc['malha']} em malha (talhões), {desc['curvo']} não alongados")
    caminhos = [i for i in range(skel.n_paths) if skid[i] in manter]
    keep = np.zeros_like(sk)
    for i in caminhos:
        rc = skel.path_coordinates(i); keep[rc[:, 0].astype(int), rc[:, 1].astype(int)] = True
    # só o que se liga ao rio principal (ponte de ponte_m)
    if semente is not None and keep.any():
        sg = gpd.GeoSeries([semente], crs=CRS_ALVO).to_crs(crs).iloc[0]
        sem = rasterize([(sg, 1)], out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
        grosso = ndi.binary_dilation(keep, structure=disk(max(1, int(ponte_m / px / 2))))
        lab2, _ = ndi.label(grosso, structure=np.ones((3, 3)))
        bons = np.unique(lab2[sem & grosso]); bons = bons[bons > 0]
        antes = keep.sum()
        keep &= np.isin(lab2, bons)
        print(f"  ligados ao rio principal: {keep.sum()/max(antes,1)*100:.0f}% do traçado")
        caminhos = [i for i in caminhos if keep[tuple(skel.path_coordinates(i)[0].astype(int))]]
    linhas = []
    for i in caminhos:
        rc = skel.path_coordinates(i)
        if len(rc) < 2:
            continue
        xs, ys = rasterio.transform.xy(tr, rc[:, 0], rc[:, 1], offset="center")
        linhas.append(LineString(zip(xs, ys)).simplify(px * 0.7))
    return _salvar(linhas, crs, pasta, rotulo)


def _salvar(linhas, crs, pasta, rotulo):
    gl = gpd.GeoDataFrame({"comp_m": [round(l.length, 1) for l in linhas]}, geometry=linhas, crs=crs)
    os.makedirs(pasta, exist_ok=True)
    gpkg = os.path.join(pasta, f"drenagem_{rotulo}.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    gl.to_file(gpkg, layer="drenagem", driver="GPKG")
    shp = os.path.join(pasta, "shp"); shutil.rmtree(shp, ignore_errors=True); os.makedirs(shp)
    gl.to_file(os.path.join(shp, f"drenagem_{rotulo}.shp"))
    shutil.make_archive(os.path.join(pasta, f"drenagem_{rotulo}_shp"), "zip", shp)
    print(f"Traçado {rotulo}: {len(gl)} linhas | {gl['comp_m'].sum()/1000:.1f} km | {gpkg}")
    return gl.to_crs(CRS_ALVO) if len(gl) else gl, gpd.GeoDataFrame(geometry=[], crs=CRS_ALVO)
