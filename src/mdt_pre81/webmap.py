# -*- coding: utf-8 -*-
"""Webmap (Leaflet) com os resultados do projeto sobre imagem de satélite Esri World Imagery.

Gera uma pasta pronta para GitHub Pages:
    index.html + assets/ (data.js, PNGs)          -> modo "site"
ou um único HTML autocontido (dados embutidos)     -> modo "standalone"
Camadas: Landsat 1979 melhorado, MDT pré-81 (cores + sombreado), curvas, drenagem 1979,
água 1979, NA_TD (por classe, com popup de atributos), AOI, WMS opcionais.
Ferramentas: opacidade, coordenadas, altitude do MDT, régua/área/marcador, tela cheia, 3D.
"""
import os, io, json, base64, shutil, warnings
import numpy as np
import geopandas as gpd
import shapely
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image

warnings.filterwarnings("ignore")
CRS_ALVO = "EPSG:31982"
WEB = "EPSG:3857"
PALETA = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
          "#bcf60c", "#fabebe", "#008080", "#9a6324"]


# ------------------------------------------------------------------ vetores
def _gj(gdf, simplificar=0.0, max_cols=14, crs_metrico=CRS_ALVO):
    """GeoDataFrame -> dict GeoJSON em EPSG:4326, enxuto (precisão ~1 m, atributos truncados)."""
    if gdf is None or len(gdf) == 0:
        return None
    g = gdf.copy()
    if g.crs is None:
        g = g.set_crs(crs_metrico)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    if simplificar:
        g["geometry"] = g.to_crs(crs_metrico).geometry.simplify(simplificar).to_crs(g.crs)
    g = g.to_crs("EPSG:4326")
    g["geometry"] = shapely.make_valid(g.geometry.values)
    try:
        g["geometry"] = shapely.set_precision(g.geometry.values, 1e-5)
    except Exception:       # geometria problemática: arredonda uma a uma e mantém a original se falhar
        def _prec(x):
            try:
                return shapely.make_valid(shapely.set_precision(x, 1e-5))
            except Exception:
                return x
        g["geometry"] = [_prec(x) for x in g.geometry.values]
    g = g[~g.geometry.is_empty]
    cols = [c for c in g.columns if c != "geometry"][:max_cols]
    g = g[cols + ["geometry"]]
    for c in cols:
        if g[c].dtype == object or str(g[c].dtype).startswith(("string", "datetime", "category")):
            g[c] = g[c].astype(str).str.slice(0, 120)
    return json.loads(g.to_json(drop_id=True))


def _campo_classe(natd):
    for c in ("CLASSE", "classe", "Classe"):
        if c in natd.columns:
            return c
    for c in natd.columns:
        if c != "geometry" and natd[c].dtype == object and 1 < natd[c].nunique() <= 12:
            return c
    return None


# ------------------------------------------------------------------ rasters
def _warp(arr, transform, crs, resampling=Resampling.bilinear, max_px=2400, nodata=0):
    """arr (bandas,h,w) float32 -> EPSG:3857. Retorna (out, bounds_latlng[[s,w],[n,e]], transform3857, bounds3857)."""
    nb, h, w = arr.shape
    f = max(1.0, max(h, w) / max_px)
    tr, W, H = calculate_default_transform(crs, WEB, w, h, *array_bounds(h, w, transform),
                                           dst_width=int(w / f) or 1, dst_height=int(h / f) or 1)
    out = np.full((nb, H, W), nodata, dtype="float32")
    for i in range(nb):
        reproject(arr[i].astype("float32"), out[i], src_transform=transform, src_crs=crs, dst_transform=tr,
                  dst_crs=WEB, resampling=resampling, src_nodata=nodata if not np.isnan(nodata) else None,
                  dst_nodata=nodata)
    b = array_bounds(H, W, tr)  # left, bottom, right, top (3857)
    ll = gpd.GeoSeries(shapely.box(*b), crs=WEB).to_crs("EPSG:4326").total_bounds  # w s e n
    return out, [[float(ll[1]), float(ll[0])], [float(ll[3]), float(ll[2])]], tr, b


def _png_data(rgba):
    im = Image.fromarray(np.clip(rgba, 0, 255).astype("uint8"), "RGBA")
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _raster_landsat(tif, max_px=2400):
    with rasterio.open(tif) as s:
        f = max(1, int(np.ceil(max(s.width, s.height) / (max_px * 1.5))))
        arr = s.read(out_shape=(s.count, s.height // f, s.width // f), resampling=Resampling.average).astype("float32")
        tr = s.transform * rasterio.Affine.scale(f, f)
        crs = s.crs
    if arr.shape[0] < 3:
        arr = np.repeat(arr[:1], 3, axis=0)
    rgb = arr[:3]
    alfa = (np.any(rgb > 0, axis=0)).astype("float32") * 255
    out, bounds, _, _ = _warp(np.concatenate([rgb, alfa[None]]), tr, crs, max_px=max_px)
    return _png_data(np.moveaxis(out, 0, -1)), bounds


def _hillshade(dem, res, az=315, alt=45):
    gy, gx = np.gradient(dem.astype("float64"), res)
    slope = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    azr, altr = np.radians(az), np.radians(alt)
    return np.clip(np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - asp), 0, 1)


def _raster_mdt(dem, transform, res, crs=CRS_ALVO, max_px=2000, grade_px=500):
    """Cores 'terrain' + sombreado -> PNG; e grade de altitudes (int16 dm) para clique/3D."""
    import matplotlib
    from matplotlib import cm
    ok = np.isfinite(dem)
    if not ok.any():
        return None
    lo, hi = np.nanpercentile(dem, 1), np.nanpercentile(dem, 99)
    hi = hi if hi > lo else lo + 1
    pre = np.where(ok, dem, np.nanmean(dem))
    hs = _hillshade(pre, res)
    cmap = matplotlib.colormaps["terrain"]
    rgb = cmap(np.clip((dem - lo) / (hi - lo), 0, 1))[..., :3]
    rgb = rgb * (0.45 + 0.55 * hs[..., None])
    rgba = np.concatenate([rgb * 255, ok[..., None] * 255.0], axis=-1)
    out, bounds, tr, b3857 = _warp(np.moveaxis(rgba, -1, 0), transform, crs, max_px=max_px)
    png = _png_data(np.moveaxis(out, 0, -1))
    # grade de altitudes em 3857
    dem_nan = np.where(ok, dem, np.nan).astype("float32")
    g, _, _, _ = _warp(dem_nan[None], transform, crs, max_px=grade_px, nodata=np.nan)
    g = g[0]
    q = np.where(np.isfinite(g), np.round(g * 10), -32768).astype("<i2")
    grade = dict(w=int(q.shape[1]), h=int(q.shape[0]), b3857=[float(v) for v in b3857],
                 dados=base64.b64encode(q.tobytes()).decode("ascii"))
    cores = [matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0, 1, 9)]
    legenda = dict(min=float(lo), max=float(hi), cores=cores)
    return png, bounds, grade, legenda


# ------------------------------------------------------------------ construção
def montar_dados(aoi=None, natd=None, drenagem=None, agua=None, curvas=None, mdt=None, landsat=None,
                 wms=None, titulo="Projeto Isabel — MDT pré-1981", resumo=None, cartas=None, historicas=None, hidro_ibge=None, ana=None):
    """Retorna (dados_dict, arquivos_png {nome: bytes}). Rasters ficam como 'arquivo' (resolvido depois)."""
    dados = dict(titulo=titulo, resumo=resumo or "", camadas=[], wms=wms or [])
    arquivos = {}
    centro = None

    if landsat and os.path.exists(landsat):
        png, b = _raster_landsat(landsat)
        arquivos["landsat_1979.png"] = png
        dados["camadas"].append(dict(id="landsat", rotulo="Landsat MSS 1979 (melhorado)", tipo="raster",
                                     arquivo="landsat_1979.png", bounds=b, ligado=True, opacidade=1.0))
    if mdt is not None:
        r = _raster_mdt(mdt["dem"], mdt["transform"], mdt["res"])
        if r:
            png, b, grade, leg = r
            arquivos["mdt_pre81.png"] = png
            dados["camadas"].append(dict(id="mdt", rotulo="MDT pré-1981 (cartas antigas)", tipo="raster",
                                         arquivo="mdt_pre81.png", bounds=b, ligado=True, opacidade=0.7,
                                         legenda=leg))
            dados["grade"] = grade
            dados["mdt_bounds"] = b
    if hidro_ibge is not None and hidro_ibge[0] is not None and len(hidro_ibge[0]):
        L, P = hidro_ibge
        dados["camadas"].append(dict(id="hidro_ibge", rotulo="Hidrografia das cartas IBGE (pré-1981)", tipo="geojson",
                                     geojson=_gj(L, 10), estilo=dict(color="#1d4ed8", weight=1.4), ligado=True))
        if P is not None and len(P):
            dados["camadas"].append(dict(id="hidro_ibge_pol", rotulo="Rio/lagoas das cartas IBGE (pré-1981)", tipo="geojson",
                                         geojson=_gj(P, 10), estilo=dict(color="#1d4ed8", weight=1, fillColor="#3b82f6",
                                                                      fillOpacity=0.35), ligado=True))
    if ana is not None and len(ana):
        dados["camadas"].append(dict(id="ana", rotulo="Hidrografia ANA (BHO)", tipo="geojson", geojson=_gj(ana, 10),
                                     estilo=dict(color="#a21caf", weight=1.6, dashArray="5 3"), ligado=True))
    for h in historicas or []:
        hid = h["id"]
        png, b = _raster_landsat(h["tif"])
        arquivos[f"imagem_{hid}.png"] = png
        dados["camadas"].append(dict(id=f"img{hid}", rotulo=f"Imagem {h['rotulo']}", tipo="raster",
                                     arquivo=f"imagem_{hid}.png", bounds=b, ligado=False, opacidade=1.0))
    for h in historicas or []:
        hid, cor = h["id"], h.get("cor", "#ffd60a")
        if h.get("agua") is not None and len(h["agua"]):
            dados["camadas"].append(dict(id=f"agua{hid}", rotulo=f"Rio/água {h['rotulo']}", tipo="geojson",
                                         geojson=_gj(h["agua"], 20), estilo=dict(color=cor, weight=1.2, fillColor=cor,
                                                                              fillOpacity=0.35), ligado=True))
        if h.get("drenagem") is not None and len(h["drenagem"]):
            dados["camadas"].append(dict(id=f"dren{hid}", rotulo=f"Drenagem {h['rotulo']}", tipo="geojson",
                                         geojson=_gj(h["drenagem"], 20), estilo=dict(color=cor, weight=1.8), ligado=True))
    for i, (rot, png, b) in enumerate(cartas or []):
        nome = f"carta_{i}.png"
        arquivos[nome] = png
        dados["camadas"].append(dict(id=f"carta{i}", rotulo="Carta (validação): " + rot, tipo="raster",
                                     arquivo=nome, bounds=b, ligado=False, opacidade=0.85))
    if curvas is not None and len(curvas):
        gj = _gj(curvas, simplificar=8)
        if gj:
            dados["camadas"].append(dict(id="curvas", rotulo="Curvas de nível (cartas)", tipo="geojson",
                                         geojson=gj, estilo=dict(color="#8b5a2b", weight=0.8), ligado=False))
    if agua is not None and len(agua):
        dados["camadas"].append(dict(id="agua", rotulo="Água 1979 (Landsat)", tipo="geojson",
                                     geojson=_gj(agua, 30), estilo=dict(color="#00b4d8", weight=1, fillColor="#00b4d8",
                                                                         fillOpacity=0.35), ligado=True))
    if drenagem is not None and len(drenagem):
        dados["camadas"].append(dict(id="drenagem", rotulo="Drenagem 1979 (Landsat)", tipo="geojson",
                                     geojson=_gj(drenagem, 30), estilo=dict(color="#00e5ff", weight=1.6), ligado=True))
    if natd is not None and len(natd):
        campo = _campo_classe(natd)
        classes = {}
        if campo:
            for i, v in enumerate(sorted(natd[campo].astype(str).unique())):
                classes[v] = PALETA[i % len(PALETA)]
        dados["camadas"].append(dict(id="natd", rotulo="NA_TD", tipo="geojson", geojson=_gj(natd, 5),
                                     estilo=dict(color="#ff2d2d", weight=1.2, fillOpacity=0.25), campo_classe=campo,
                                     classes=classes, ligado=True))
    if aoi is not None:
        a = gpd.GeoDataFrame({"nome": ["Região de estudo (AOI)"]}, geometry=[aoi], crs=CRS_ALVO)
        dados["camadas"].append(dict(id="aoi", rotulo="Região de estudo (AOI)", tipo="geojson",
                                     geojson=_gj(a, 50), estilo=dict(color="#ffffff", weight=2, fill=False,
                                                                     dashArray="6 4"), ligado=True))
        ll = a.to_crs("EPSG:4326").total_bounds
        centro = [[float(ll[1]), float(ll[0])], [float(ll[3]), float(ll[2])]]
    dados["extensao"] = centro or dados.get("mdt_bounds") or [[-21.4, -50.6], [-20.9, -50.0]]
    return dados, arquivos


def build_webmap(saida, modo="site", **kw):
    """Escreve o webmap em `saida`. modo='site' -> saida/index.html + assets/ ; 'standalone' -> um HTML só."""
    dados, arquivos = montar_dados(**kw)
    os.makedirs(saida, exist_ok=True)
    if modo == "standalone":
        for c in dados["camadas"]:
            if c["tipo"] == "raster":
                c["url"] = "data:image/png;base64," + base64.b64encode(arquivos[c["arquivo"]]).decode()
        html = _html().replace("<!--DADOS-->", "<script>window.DADOS=" + json.dumps(dados, separators=(",", ":"))
                               + ";</script>")
        arq = os.path.join(saida, "webmap_standalone.html")
        open(arq, "w", encoding="utf-8").write(html)
    else:
        ass = os.path.join(saida, "assets")
        shutil.rmtree(ass, ignore_errors=True); os.makedirs(ass)
        for nome, b in arquivos.items():
            open(os.path.join(ass, nome), "wb").write(b)
        for c in dados["camadas"]:
            if c["tipo"] == "raster":
                c["url"] = "assets/" + c["arquivo"]
        open(os.path.join(ass, "dados.js"), "w", encoding="utf-8").write(
            "window.DADOS=" + json.dumps(dados, separators=(",", ":")) + ";")
        open(os.path.join(saida, ".nojekyll"), "w").close()
        html = _html().replace("<!--DADOS-->", '<script src="assets/dados.js"></script>')
        arq = os.path.join(saida, "index.html")
        open(arq, "w", encoding="utf-8").write(html)
    print(f"Webmap ({modo}): {arq} ({os.path.getsize(arq)/1e6:.2f} MB)")
    return arq


def _html():
    return open(os.path.join(os.path.dirname(__file__), "webmap_template.html"), encoding="utf-8").read()
