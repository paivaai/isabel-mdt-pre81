# -*- coding: utf-8 -*-
"""Hidrografia da ANA (Base Hidrográfica Ottocodificada - BHO, derivada do Mapeamento Sistemático) recortada na AOI.

Os endereços exatos dos serviços da ANA mudam; por isso a busca é por DESCOBERTA (WFS GeoServer e ArcGIS REST) e tudo
o que for encontrado/tentado vai para o relatório `data/ana_inventario.txt` (rodar no GitHub Actions, que tem internet).
Também registra o GetCapabilities dos WMS do BDGEx (diagnóstico de por que não exibem).
"""
import os, re, json, warnings
import requests
import geopandas as gpd
import shapely

warnings.filterwarnings("ignore")
UA = {"User-Agent": "Mozilla/5.0 (projeto-isabel)"}
WFS = ["https://metadados.snirh.gov.br/geoserver/ows", "https://www.snirh.gov.br/geoserver/ows"]
ARCGIS = ["https://portal1.snirh.gov.br/server/rest/services", "https://www.snirh.gov.br/arcgis/rest/services"]
FILTRO = re.compile(r"bho|trecho|drenagem|hidrograf|curso", re.I)
BDGEX = ["https://bdgex.eb.mil.br/teogc/50/terraogcmed.cgi", "https://bdgex.eb.mil.br/teogc/25/terraogcmed.cgi",
         "https://bdgex.eb.mil.br/teogc/100/terraogcmed.cgi", "http://bdgex.eb.mil.br/mapcache"]


def _get(url, log, **kw):
    try:
        r = requests.get(url, timeout=kw.pop("timeout", 60), headers=UA, **kw)
        log.append(f"  GET {url[:150]} -> {r.status_code} {r.headers.get('content-type','')[:40]} ({len(r.content)} B)")
        return r
    except Exception as e:
        log.append(f"  GET {url[:150]} -> ERRO {type(e).__name__}: {str(e)[:80]}")
        return None


def _bbox(aoi, crs="EPSG:31982", margem_m=2000):
    g = gpd.GeoSeries([aoi.buffer(margem_m)], crs=crs).to_crs("EPSG:4326")
    return g.total_bounds  # minx,miny,maxx,maxy


def _so_linhas(g):
    if g is None or not len(g):
        return None
    g = g[g.geometry.notna() & g.geom_type.isin(["LineString", "MultiLineString"])]
    return g if len(g) else None


def _wfs(base, bb, log):
    out = []
    r = _get(f"{base}?service=WFS&version=2.0.0&request=GetCapabilities", log)
    if r is None or r.status_code != 200:
        return out
    nomes = re.findall(r"<(?:\w+:)?Name>([^<>]+)</(?:\w+:)?Name>", r.text)
    nomes = [n for n in dict.fromkeys(nomes) if FILTRO.search(n)]
    log.append(f"  tipos de feição candidatos ({len(nomes)}): {nomes[:40]}")
    nomes.sort(key=lambda n: (0 if "trecho" in n.lower() else 1, n))
    for n in nomes[:8]:
        for bbox in (f"{bb[1]},{bb[0]},{bb[3]},{bb[2]},EPSG:4326", f"{bb[0]},{bb[1]},{bb[2]},{bb[3]},EPSG:4326"):
            u = (f"{base}?service=WFS&version=2.0.0&request=GetFeature&typeNames={n}&outputFormat=application/json"
                 f"&srsName=EPSG:4326&count=20000&bbox={bbox}")
            r = _get(u, log, timeout=120)
            if r is None or r.status_code != 200 or not r.text.lstrip().startswith("{"):
                continue
            try:
                g = _so_linhas(gpd.GeoDataFrame.from_features(json.loads(r.text)["features"], crs="EPSG:4326"))
            except Exception:
                continue
            if g is not None:
                tb = g.total_bounds
                if abs(tb[0]) < 90 and abs(tb[1]) > 90:      # eixos trocados
                    continue
                out.append((n, g)); log.append(f"  OK {n}: {len(g)} linhas"); break
    return out


def _arcgis(root, bb, log, prof=2):
    out = []
    def ls(url, p):
        r = _get(url + "?f=json", log)
        if r is None or r.status_code != 200:
            return []
        try:
            j = r.json()
        except Exception:
            return []
        achados = []
        for s in j.get("services", []):
            if s.get("type") in ("MapServer", "FeatureServer") and FILTRO.search(s["name"]):
                achados.append(f"{root.split('/rest')[0]}/rest/services/{s['name']}/{s['type']}")
        if p < prof:
            for f in j.get("folders", []):
                achados += ls(f"{url}/{f}", p + 1)
        return achados
    for svc in ls(root, 0)[:10]:
        r = _get(svc + "?f=json", log)
        try:
            camadas = r.json().get("layers", [])
        except Exception:
            continue
        for c in camadas:
            if not FILTRO.search(c["name"]) or "area" in c["name"].lower() or "ponto" in c["name"].lower():
                continue
            u = (f"{svc}/{c['id']}/query?where=1%3D1&geometry={bb[0]},{bb[1]},{bb[2]},{bb[3]}&geometryType=esriGeometryEnvelope"
                 f"&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&returnGeometry=true&outSR=4326&f=geojson&resultRecordCount=5000")
            r = _get(u, log, timeout=120)
            try:
                g = _so_linhas(gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326"))
            except Exception:
                continue
            if g is not None:
                out.append((f"{svc.split('/services/')[-1]}/{c['name']}", g)); log.append(f"  OK {c['name']}: {len(g)} linhas")
    return out


def diagnostico_bdgex(log):
    log.append("\n== DIAGNÓSTICO BDGEx (por que o WMS não exibe) ==")
    for u in BDGEX:
        sep = "&" if "?" in u else "?"
        r = _get(f"{u}{sep}SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.1.1", log)
        if r is not None and r.status_code == 200:
            nomes = re.findall(r"<Name>([^<]+)</Name>", r.text)
            log.append(f"    camadas: {nomes[:20]} | CORS: {r.headers.get('access-control-allow-origin')}")
            m = re.findall(r"<(?:Abstract|Title)>([^<]{3,200})</", r.text)
            log.append(f"    textos: {m[:6]}")


def baixar_bho(aoi, saida_gpkg, relatorio):
    log = ["== ANA: descoberta da hidrografia (BHO) =="]
    bb = _bbox(aoi)
    log.append(f"bbox (lon/lat): {[round(float(v),4) for v in bb]}")
    achados = []
    for b in WFS:
        achados += _wfs(b, bb, log)
        if achados:
            break
    if not achados:
        for rz in ARCGIS:
            achados += _arcgis(rz, bb, log)
            if achados:
                break
    try:
        diagnostico_bdgex(log)
    except Exception as e:
        log.append(f"diagnóstico BDGEx falhou: {e}")
    os.makedirs(os.path.dirname(relatorio) or ".", exist_ok=True)
    if not achados:
        log.append("\nNADA obtido da ANA nesta execução (veja as linhas acima). Alternativa: baixar o GPKG da BHO "
                   "(metadados.snirh.gov.br, 'Trechos de Drenagem') e colocar em data/ana_bho.gpkg.")
        open(relatorio, "w", encoding="utf-8").write("\n".join(log)); print("\n".join(log)); return None
    nome, g = max(achados, key=lambda t: len(t[1]))
    cols = [c for c in g.columns if c != "geometry"][:12]
    g = g[cols + ["geometry"]].copy()
    for c in cols:
        g[c] = g[c].astype(str).str.slice(0, 80)
    g["geometry"] = shapely.make_valid(g.geometry.values)
    g.to_crs("EPSG:31982").to_file(saida_gpkg, layer="ana_bho", driver="GPKG")
    log.append(f"\nSalvo: {saida_gpkg} (camada '{nome}', {len(g)} trechos, colunas {cols})")
    open(relatorio, "w", encoding="utf-8").write("\n".join(log)); print("\n".join(log))
    return g
