# -*- coding: utf-8 -*-
"""Orquestra as etapas lendo config/projeto.yml. Cada etapa grava em `saida/` e pode rodar sozinha."""
import os, sys
import numpy as np
import geopandas as gpd
import rasterio
import yaml
from . import core, webmap
from .ibge import baixar_curvas_ibge

ETAPAS = ["aoi", "landsat", "drenagem", "curvas", "mdt", "webmap"]


def carregar_config(caminho="config/projeto.yml"):
    with open(caminho, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_base"] = os.path.dirname(os.path.abspath(caminho)) + "/.."
    return cfg


def _p(cfg, chave):
    """Caminho relativo à raiz do repositório (None se vazio)."""
    v = cfg["entradas"].get(chave)
    if not v:
        return None
    p = os.path.normpath(os.path.join(cfg["_base"], v))
    if not os.path.exists(p) and chave in ("landsat", "natd"):
        # tolera outro nome/extensão na mesma pasta (ex.: NA_TD_TOTAL.rar em vez de .zip)
        import glob
        pasta = os.path.dirname(p)
        exts = (".tif", ".tiff") if chave == "landsat" else (".shp", ".zip", ".rar")
        cands = sorted(f for f in glob.glob(os.path.join(pasta, "*")) if f.lower().endswith(exts)
                       and (chave != "natd" or "na_td" in os.path.basename(f).lower() or "natd" in os.path.basename(f).lower()))
        if cands:
            print(f"  ({chave}: usando {os.path.basename(cands[0])})")
            return cands[0]
    return p


def _saida(cfg, *partes):
    d = os.path.normpath(os.path.join(cfg["_base"], cfg["saidas"]["pasta"]))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *partes)


def _natd(cfg):
    p = _p(cfg, "natd")
    if not p or not os.path.exists(p):
        print("  (NA_TD não encontrado: segue sem ele)")
        return None
    shp = core.preparar_shapefile(p, _saida(cfg, "_natd"))
    return gpd.read_file(shp).to_crs(core.CRS_ALVO)


def etapa_aoi(cfg):
    a = cfg["area"]
    natd = _natd(cfg)
    aoi = core.montar_aoi(a["municipios"], natd, margem_km=a.get("margem_km", 3.0), fallback_km=a.get("fallback_km", 12.0),
                          aoi_arquivo=_p(cfg, "aoi"), usar_ibge=a.get("baixar_limites_ibge", True))
    gpd.GeoDataFrame({"nome": ["AOI"]}, geometry=[aoi], crs=core.CRS_ALVO).to_file(_saida(cfg, "aoi.gpkg"), driver="GPKG")
    return aoi


def _aoi(cfg):
    arq = _saida(cfg, "aoi.gpkg")
    if not os.path.exists(arq):
        return etapa_aoi(cfg)
    import shapely
    return shapely.union_all(gpd.read_file(arq).geometry.values)


def etapa_landsat(cfg):
    lan = _p(cfg, "landsat")
    if not lan or not os.path.exists(lan):
        print("  Landsat não encontrado em", lan, "- etapa pulada."); return None
    aoi = _aoi(cfg)
    rec = core.recortar_landsat(lan, aoi, _saida(cfg))
    if cfg["landsat"].get("melhorar", True):
        l = cfg["landsat"]
        return core.melhorar_landsat(rec, _saida(cfg, "landsat_1979_melhorado.tif"), upscale=l.get("upscale", 2),
                                     gama=l.get("gama", 0.95), saturacao=l.get("saturacao", 1.25),
                                     clahe_clip=l.get("clahe_clip", 0.004), nitidez=l.get("nitidez", 0.8))
    # sem nova melhoria: a imagem de entrada (já melhorada) é usada como está no webmap
    import shutil
    shutil.copy(rec, _saida(cfg, "landsat_1979_melhorado.tif"))
    return rec


def etapa_drenagem(cfg):
    rec = _saida(cfg, "landsat_1979_recorte.tif")
    if not os.path.exists(rec):
        etapa_landsat(cfg)
    if not os.path.exists(rec):
        print("  Sem Landsat: drenagem pulada."); return None
    d = cfg["drenagem"]
    return core.extrair_drenagem(rec, _saida(cfg), modo=d.get("modo", "todas"), ajuste_limiar=d.get("ajuste_limiar", 1.0),
                                 area_min_px=d.get("area_min_px", 12), compr_min_px=d.get("compr_min_px", 15))


def etapa_curvas(cfg):
    dest = _p(cfg, "curvas")
    if dest and os.path.exists(dest):
        print("Curvas já disponíveis:", dest); return dest
    if cfg["curvas"].get("baixar_ibge", True) and dest:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        baixar_curvas_ibge(dest=os.path.join(os.path.dirname(dest), "ibge_bruto"), saida_gpkg=dest,
                           crs_dgn=cfg["curvas"].get("crs_origem") or "EPSG:22522", rotular=cfg["curvas"].get("rotular", True))
        return dest if os.path.exists(dest) else None
    print("Sem curvas: coloque-as em", dest); return None


def etapa_mdt(cfg):
    cam = _p(cfg, "curvas")
    if not cam or not os.path.exists(cam):
        etapa_curvas(cfg)
    if not cam or not os.path.exists(cam):
        print("  Sem curvas de nível: MDT NÃO gerado (nada é inventado nem misturado com Topodata)."); return None
    c = cfg["curvas"]
    hid = None
    arq_hid = _saida(cfg, "drenagem_1979.gpkg")
    if os.path.exists(arq_hid) and c.get("usar_drenagem_no_talvegue", True):
        hid = gpd.read_file(arq_hid, layer="drenagem")
    r = core.gerar_mdt_pre81(cam, _aoi(cfg), _saida(cfg), crs_origem=c.get("crs_origem"), campo=c.get("campo_cota"),
                             pontos_path=_p(cfg, "pontos"), hidro=hid, res=c.get("resolucao_m", 30.0),
                             suavizar=c.get("suavizar", 1.0), max_dist=c.get("dist_max_curva_m", 1500.0))
    core.qa_mdt(r, _natd(cfg))
    return r


def _carregar_mdt(cfg):
    arq = _saida(cfg, "mdt_pre81.tif")
    if not os.path.exists(arq):
        return None
    with rasterio.open(arq) as s:
        dem = s.read(1).astype("float32"); nd = s.nodata; tr = s.transform; res = abs(s.transform.a)
    dem[(dem == nd) | ~np.isfinite(dem)] = np.nan
    return dict(dem=dem, transform=tr, res=res, arq=arq)


def etapa_webmap(cfg):
    w = cfg["webmap"]
    aoi = _aoi(cfg)
    natd = _natd(cfg)
    gp = _saida(cfg, "drenagem_1979.gpkg")
    drenagem = gpd.read_file(gp, layer="drenagem") if os.path.exists(gp) else None
    try:
        agua = gpd.read_file(gp, layer="agua") if os.path.exists(gp) else None
    except Exception:
        agua = None
    cp = _p(cfg, "curvas")
    curvas = gpd.read_file(cp) if cp and os.path.exists(cp) else None
    mdt = _carregar_mdt(cfg)
    lan = _saida(cfg, "landsat_1979_melhorado.tif")
    lan = lan if os.path.exists(lan) else None
    from . import cartas as _c
    cartas = _c.camadas_cartas(os.path.join(cfg["_base"], "data", "cartas_raster"), w.get("cartas") or [])
    kw = dict(cartas=cartas, aoi=aoi, natd=natd, drenagem=drenagem, agua=agua, curvas=curvas, mdt=mdt, landsat=lan,
              wms=w.get("wms") or [], titulo=w.get("titulo", "Projeto Isabel — MDT pré-1981"), resumo=w.get("resumo", ""))
    pasta = os.path.normpath(os.path.join(cfg["_base"], w.get("pasta_site", "docs")))
    out = [webmap.build_webmap(pasta, modo="site", **kw)]
    if w.get("standalone", True):
        out.append(webmap.build_webmap(_saida(cfg), modo="standalone", **kw))
    return out


def etapa_cartas(cfg):
    """Baixa/inspeciona as cartas escaneadas do IBGE (rodar no GitHub Actions; fora de 'all')."""
    from . import cartas as _c
    _c.inspecionar_cartas(os.path.join(cfg["_base"], "data", "cartas_raster"))


FUNCOES = dict(cartas=etapa_cartas, aoi=etapa_aoi, landsat=etapa_landsat, drenagem=etapa_drenagem, curvas=etapa_curvas,
               mdt=etapa_mdt, webmap=etapa_webmap)


def executar(etapas, config="config/projeto.yml"):
    """Roda as etapas; um erro numa etapa é registrado e as seguintes continuam (sai com código 1 no fim)."""
    import traceback
    cfg = carregar_config(config)
    falhas = []
    for e in (ETAPAS if etapas == ["all"] else etapas):
        print("\n" + "=" * 72 + f"\nETAPA: {e}\n" + "=" * 72, flush=True)
        try:
            FUNCOES[e](cfg)
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"\n!!! ERRO NA ETAPA '{e}': {type(ex).__name__}: {ex}\n{tb}", flush=True)
            falhas.append((e, f"{type(ex).__name__}: {ex}"))
    print("\n" + "=" * 72 + "\nRESUMO\n" + "=" * 72)
    if falhas:
        for e, m in falhas:
            print(f"  FALHOU  {e}: {m}")
        sys.exit(1)
    print("  todas as etapas concluídas")
