# -*- coding: utf-8 -*-
"""Imagens históricas (ex.: 1967, 1973): recorte pela AOI, realce e extração do rio/drenagem ANTES do alagamento.

Aceita GeoTIFF ou TIFF + .tfw (sem CRS: informe `crs` na config ou será deduzido pela AOI).
Lê só a janela da AOI e reduz (px_max) para caber na memória, mesmo com TIFs de ~1 GB.
"""
import os, warnings
import shapely
import numpy as np
import geopandas as gpd
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling, ColorInterp
from . import core

warnings.filterwarnings("ignore")
CAND_CRS = ["EPSG:31982", "EPSG:22522", "EPSG:29192", "EPSG:31982", "EPSG:4674", "EPSG:4326", "EPSG:4618", "EPSG:32722"]


def adivinhar_crs(src, aoi):
    """CRS (entre os plausíveis) cujo envelope da imagem cai sobre a AOI."""
    b = shapely.box(*src.bounds)
    alvo = gpd.GeoSeries([aoi], crs=core.CRS_ALVO)
    for c in dict.fromkeys(CAND_CRS):
        try:
            g = gpd.GeoSeries([b], crs=c).to_crs(core.CRS_ALVO)
            if np.isfinite(g.total_bounds).all() and g.iloc[0].intersects(alvo.iloc[0]):
                return c
        except Exception:
            continue
    return None


def _a8(a, nd):
    if a.dtype == np.uint8:
        return a
    v = a[a != nd] if nd is not None else a.ravel()
    lo, hi = np.percentile(v[::max(1, v.size // 2_000_000)], (0.5, 99.5)) if v.size else (0, 1)
    return np.clip((a.astype("float32") - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype("uint8")


def recortar_imagem(arq, aoi, saida_tif, crs=None, px_max=7000):
    """Recorta pela AOI (janela), reduz para <= px_max, estica contraste, grava uint8. Retorna (caminho, info)."""
    with rasterio.open(arq) as src:
        crs_src = CRS.from_user_input(crs) if crs else src.crs
        origem = "config" if crs else "arquivo"
        if crs_src is None:
            c = adivinhar_crs(src, aoi)
            if not c:
                raise ValueError(f"{os.path.basename(arq)}: sem CRS e fora da AOI em todos os CRS testados; informe 'crs' na config.")
            crs_src, origem = CRS.from_user_input(c), "deduzido pela AOI (confira!)"
        print(f"  {os.path.basename(arq)}: {src.width}x{src.height}px, {src.count} banda(s), CRS {crs_src.to_string()} [{origem}]")
        tr0 = src.transform
        b = gpd.GeoSeries([aoi], crs=core.CRS_ALVO).to_crs(crs_src).total_bounds
        win = from_bounds(*b, transform=tr0).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width < 2 or win.height < 2:
            raise ValueError(f"{os.path.basename(arq)}: a imagem não cobre a AOI (confira o CRS).")
        f = max(1.0, max(win.width, win.height) / px_max)
        H, W = int(win.height / f), int(win.width / f)
        bandas = [i + 1 for i, ci in enumerate(src.colorinterp) if ci != ColorInterp.alpha][:4] or [1]
        dados = src.read(bandas, window=win, out_shape=(len(bandas), H, W), resampling=Resampling.average)
        tr = src.window_transform(win) * rasterio.Affine.scale(win.width / W, win.height / H)
        nd = src.nodata
        px = abs(src.res[0]) * f
    if nd is None:  # moldura preta/branca de mosaico?
        cantos = np.concatenate([dados[:, :3, :3].ravel(), dados[:, -3:, -3:].ravel()])
        for v in (0, 255):
            if (cantos == v).mean() > 0.9:
                nd = v
                break
    if dados.shape[0] >= 3 and np.abs(dados[0].astype("float32") - dados[1]).mean() < 2:
        dados = dados[:1]                      # RGB cinza (foto pancromática) -> 1 banda
    dados = np.stack([_a8(b, nd) for b in dados])
    if nd is not None and nd > 255:
        nd = None
    prof = dict(driver="GTiff", height=H, width=W, count=dados.shape[0], dtype="uint8", crs=crs_src, transform=tr,
                compress="lzw", nodata=nd if nd is not None else None)
    os.makedirs(os.path.dirname(saida_tif) or ".", exist_ok=True)
    with rasterio.open(saida_tif, "w", **prof) as d:
        d.write(dados)
    util = (~np.all(dados == (nd if nd is not None else -1), axis=0)).mean() * 100
    print(f"  recorte: {W}x{H}px (pixel ~{px:.1f} m no CRS da imagem), {dados.shape[0]} banda(s), {util:.0f}% com dados")
    return saida_tif, dict(px=px, util=util, crs=crs_src.to_string())


def processar_imagem(item, aoi, base, saida, natd=None):
    """item: {id, rotulo, arquivo, crs?, cor?, px_max?, modo?, ajuste_limiar?, area_min_px?, compr_min_px?}"""
    hid = str(item["id"])
    arq = os.path.normpath(os.path.join(base, item["arquivo"]))
    if not os.path.exists(arq):
        print(f"  Imagem {hid}: {arq} não encontrada; pulada.")
        return None
    pasta = os.path.join(saida, f"hist_{hid}")
    rec, info = recortar_imagem(arq, aoi, os.path.join(pasta, f"recorte_{hid}.tif"), item.get("crs"),
                                int(item.get("px_max", 7000)))
    mascara = semente = None
    km = float(item.get("corredor_km", 3))
    if natd is not None and len(natd) and km > 0:   # procura só ao longo das áreas NA_TD (+ corredor)
        g = shapely.union_all(natd.geometry.values)
        mascara = g.buffer(km * 1000).intersection(aoi)
        semente = g.buffer(float(item.get("semente_m", 300))).intersection(aoi)
        print(f"  corredor de busca: NA_TD + {km:g} km")
    if item.get("metodo", "tracado") == "tracado":
        from . import tracado
        gl, ga = tracado.extrair_tracado(
            rec, pasta, hid, mascara=mascara, semente=semente,
            largura_max_m=float(item.get("largura_max_m", 250)), comp_min_m=float(item.get("comp_min_m", 1500)),
            laco_max_por_km=float(item.get("laco_max_por_km", 1.2)), elong_min=float(item.get("elong_min", 0.22)),
            ponte_m=float(item.get("ponte_m", 250)), esporao_m=float(item.get("esporao_m", 150)),
            q_baixo=float(item.get("q_baixo", 90)), q_alto=float(item.get("q_alto", 97)))
    else:                                           # método antigo (manchas escuras -> água)
        gl, ga = core.extrair_drenagem(rec, pasta, modo=item.get("modo", "todas"),
                                       ajuste_limiar=item.get("ajuste_limiar", 1.0),
                                       area_min_px=item.get("area_min_px", 12), compr_min_px=item.get("compr_min_px", 15),
                                       rotulo=hid, poligono=item.get("poligono", "toda"), mascara=mascara,
                                       textura_px=int(item.get("textura_px", 7)),
                                       agua_max_pct=item.get("agua_max_pct", 15),
                                       area_min_km2=float(item.get("area_min_km2", 0.02)))
    return dict(id=hid, gl=gl, ga=ga, info=info)


def carregar_resultados(itens, saida):
    """Lê o que já foi gerado (para o webmap): lista de dicts id/rotulo/cor/tif/drenagem/agua."""
    res = []
    for it in itens or []:
        hid = str(it["id"])
        pasta = os.path.join(saida, f"hist_{hid}")
        tif, gp = os.path.join(pasta, f"recorte_{hid}.tif"), os.path.join(pasta, f"drenagem_{hid}.gpkg")
        if not (os.path.exists(tif) and os.path.exists(gp)):
            print(f"  (webmap) imagem {hid}: sem resultados em {pasta}")
            continue
        try:
            ag = gpd.read_file(gp, layer="agua")
        except Exception:
            ag = None
        res.append(dict(id=hid, rotulo=it.get("rotulo", hid), cor=it.get("cor", "#ffd60a"), tif=tif,
                        drenagem=gpd.read_file(gp, layer="drenagem"), agua=ag))
    return res
