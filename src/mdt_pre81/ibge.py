# -*- coding: utf-8 -*-
"""
Baixa do IBGE as cartas topográficas VETORIAIS 1:50.000 (conversão das cartas analógicas antigas)
das folhas que cobrem Buritama/Zacarias, INSPECIONA o conteúdo e monta o curvas_cartas.gpkg
(curvas de nível com campo numérico 'cota') para o notebook do MDT pré-1981.

Fonte: https://geoftp.ibge.gov.br/cartas_e_mapas/folhas_topograficas/vetoriais/escala_50mil/projeto_conv_digital/
Cada folha tem vetor/<MI>hp.zip (hipsografia = curvas de nível) e <MI>hd.zip (hidrografia), entre outros.

USO NO COLAB:
    !pip -q install geopandas pyogrio requests
    %run /content/baixar_curvas_ibge.py
Depois COLE AQUI a saída da célula (o RELATÓRIO no final) se algo não sair como esperado.
"""
import os, re, sys, zipfile, glob, shutil, warnings
import numpy as np
import pandas as pd
import requests
import geopandas as gpd

warnings.filterwarnings("ignore")

# ============================ CONFIGURAÇÃO ============================
BASE = os.environ.get(
    "IBGE_BASE",
    "https://geoftp.ibge.gov.br/cartas_e_mapas/folhas_topograficas/vetoriais/escala_50mil/projeto_conv_digital/")
# folhas já conhecidas (pasta = nome + código MI). Outras são procuradas pelas palavras abaixo.
FOLHAS = ["buritama26322", "avanhandava26333", "birigui26323", "aracatuba26321"]
PROCURAR = ["zacarias", "brejo", "penapolis", "bilac", "gabriel"]
# folhas vizinhas pelo código MI (sufixo da pasta): leste de Buritama (2633-1/2) e norte (2631/2632 ao norte)
CODIGOS_EXTRA = ["26331", "26332", "26334", "26311", "26312", "26313", "26314"]
TEMAS = ["hp", "hd"]            # hp = hipsografia (curvas de nível) | hd = hidrografia
DEST = os.environ.get("DEST", "/content/ibge_curvas")
SAIDA_GPKG = os.environ.get("SAIDA_GPKG", "/content/curvas_cartas.gpkg")
CAMPOS_COTA = ["cota", "elevacao", "elevação", "altitude", "alt", "elev", "z", "valor", "value",
               "height", "altura", "nivel", "contour", "elevation", "vl_cota", "cota_m"]
# ======================================================================
S = requests.Session()
ACHADOS = {}   # relatório


def listar(url):
    r = S.get(url, timeout=60)
    r.raise_for_status()
    nomes = re.findall(r'href="([^"?#]+)"', r.text)
    return [n for n in nomes if not n.startswith(("/", "http", "..", "?"))]


def baixar(url, destino):
    r = S.get(url, timeout=300)
    r.raise_for_status()
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "wb") as f:
        f.write(r.content)
    return destino


def descobrir_folhas():
    folhas = list(FOLHAS)
    try:
        todas = [n.rstrip("/") for n in listar(BASE) if n.endswith("/")]
        print(f"Índice do IBGE: {len(todas)} folhas listadas.")
        for p in PROCURAR:
            for n in todas:
                if p in n.lower() and n not in folhas:
                    folhas.append(n)
                    print(f"  + folha encontrada pela palavra '{p}': {n}")
        for cod in CODIGOS_EXTRA:
            for n in todas:
                if n.endswith(cod) and n not in folhas:
                    folhas.append(n)
                    print(f"  + folha vizinha pelo código MI {cod}: {n}")
        faltam = [f for f in FOLHAS if f not in todas]
        if faltam:
            print("  ! não achei no índice:", faltam)
    except Exception as e:
        print(f"  ! não consegui listar o índice ({type(e).__name__}); uso só a lista fixa.")
    return folhas


def baixar_folha(folha):
    arquivos = []
    try:
        nomes = listar(f"{BASE}{folha}/vetor/")
    except Exception as e:
        print(f"  ! {folha}: sem pasta vetor/ ({type(e).__name__})")
        return arquivos
    for n in nomes:
        base = n.lower()
        if base.endswith(".zip") and any(base[:-4].endswith(t) for t in TEMAS):
            z = baixar(f"{BASE}{folha}/vetor/{n}", os.path.join(DEST, folha, n))
            pasta = os.path.join(DEST, folha, n[:-4])
            os.makedirs(pasta, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(pasta)
            arquivos.append(pasta)
            print(f"  {folha}: {n} ({os.path.getsize(z)/1e3:.0f} KB)")
    return arquivos


EXT_VETOR = (".shp", ".dgn", ".dxf", ".gml", ".json", ".geojson", ".gpkg", ".kml", ".mif", ".tab")


def ler_tudo(pasta):
    """Lê todos os vetores dentro de uma pasta (cada camada de cada arquivo)."""
    saida = []
    for arq in sorted(glob.glob(os.path.join(pasta, "**", "*"), recursive=True)):
        if not arq.lower().endswith(EXT_VETOR):
            continue
        try:
            import pyogrio
            camadas = [c[0] for c in pyogrio.list_layers(arq)]
        except Exception:
            camadas = [None]
        for cam in camadas:
            g, erro = None, None
            for enc in (None, "latin-1", "cp1252"):      # DGN antigo: acentos fora de UTF-8
                try:
                    kw = {"encoding": enc} if enc else {}
                    g = gpd.read_file(arq, layer=cam, **kw) if cam else gpd.read_file(arq, **kw)
                    break
                except Exception as e:
                    erro = e
            if g is not None:
                saida.append((arq, cam, g))
            else:
                print(f"    não consegui ler {os.path.basename(arq)} [{cam}]: {type(erro).__name__}: {str(erro)[:80]}")
    return saida


def tem_z(g):
    try:
        return bool(g.geometry.has_z.any())
    except Exception:
        return False


def relatorio_camada(arq, cam, g):
    tipos = g.geom_type.value_counts().to_dict()
    cols = [c for c in g.columns if c != "geometry"]
    print(f"    - {os.path.relpath(arq, DEST)}{f' [{cam}]' if cam else ''}: {len(g)} feições | {tipos} | "
          f"CRS: {g.crs.to_string() if g.crs else 'SEM CRS'} | Z na geometria: {tem_z(g)}")
    for c in cols[:20]:
        s = g[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                print(f"        campo {c!r} (numérico): {s.min()} .. {s.max()} | {s.nunique()} valores")
            else:
                t = s.dropna().astype(str)          # astype(str): colunas com dict/lista não quebram
                print(f"        campo {c!r} (texto): {t.nunique()} valores, ex.: {list(t.unique()[:5])}")
        except Exception as e:
            print(f"        campo {c!r}: não consegui resumir ({type(e).__name__})")


def achar_cota(g):
    """Procura o campo de altitude; senão usa o Z da geometria. Retorna (serie, origem) ou (None, motivo)."""
    low = {str(c).lower(): c for c in g.columns if c != "geometry"}
    for nome in CAMPOS_COTA:
        if nome in low:
            s = g[low[nome]]
            s = s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
            if s.notna().mean() > 0.5 and s.nunique() > 1:
                return s.astype(float), f"campo '{low[nome]}'"
    if tem_z(g):
        z = g.geometry.apply(lambda ge: float(np.nanmean([c[2] for c in ge.coords])) if ge.geom_type == "LineString"
                             and ge.has_z else np.nan)
        if z.notna().mean() > 0.5 and z.nunique() > 1:
            return z, "coordenada Z da geometria"
    return None, "nenhum campo de altitude e sem Z"


def baixar_curvas_ibge(dest=None, saida_gpkg=None, crs_dgn="EPSG:22522", rotular=True):
    """Baixa, inspeciona e consolida as curvas das folhas do IBGE em um GeoPackage.
    Se as curvas vierem sem altitude (caso das cartas DGN do IBGE), rotula automaticamente (rotular.py).
    crs_dgn: datum ORIGINAL das cartas (DGN em km UTM 22S): Córrego Alegre EPSG:22522 (padrão) ou SAD69 EPSG:29192."""
    global DEST, SAIDA_GPKG
    if dest:
        DEST = dest
    if saida_gpkg:
        SAIDA_GPKG = saida_gpkg
    print("=" * 72, "\n1) FOLHAS\n" + "=" * 72)
    folhas = descobrir_folhas()
    print("Folhas a baixar:", folhas)

    print("\n" + "=" * 72, "\n2) DOWNLOAD\n" + "=" * 72)
    pastas = {f: baixar_folha(f) for f in folhas}

    if not any(pastas.values()):
        print("\n  Não consegui baixar nenhuma folha (sem internet, ou o IBGE mudou a estrutura das pastas).")
        print("  Abra no navegador e confira: " + BASE)
        print("  Se baixar os .zip 'hp' (hipsografia) à mão, descompacte-os em " + DEST + " e rode de novo.")
        return
    print("\n" + "=" * 72, "\n3) INSPEÇÃO DO CONTEÚDO\n" + "=" * 72)
    linhas_ok, sem_cota, textos = [], [], 0
    for folha, plist in pastas.items():
        for p in plist:
            tema = "hp" if p.lower().endswith("hp") else ("hd" if p.lower().endswith("hd") else "?")
            print(f"\n  {folha} / {os.path.basename(p)}  (tema {tema})")
            for arq, cam, g in ler_tudo(p):
                relatorio_camada(arq, cam, g)
                if tema != "hp":
                    continue
                lin = g[g.geom_type.isin(["LineString", "MultiLineString"])]
                pts = g[g.geom_type == "Point"]
                if len(pts) and any(str(c).lower() == "text" for c in g.columns):
                    t = pts[[c for c in g.columns if str(c).lower() == "text"][0]]
                    textos += int(pd.to_numeric(t, errors="coerce").notna().sum())
                if len(lin):
                    cota, origem = achar_cota(lin)
                    if cota is not None:
                        lin = lin.assign(cota=cota.values)[["cota", "geometry"]]
                        lin["folha"] = folha
                        linhas_ok.append(lin)
                        print(f"        => CURVAS COM ALTITUDE ({origem}): {len(lin)} linhas, "
                              f"cota {cota.min():.0f} a {cota.max():.0f}")
                    else:
                        sem_cota.append((folha, len(lin), origem))

    print("\n" + "=" * 72, "\n4) RESULTADO\n" + "=" * 72)
    if linhas_ok:
        crs = {str(l.crs) for l in linhas_ok}
        if len(crs) > 1:
            print("  ! CRS diferentes entre as folhas:", crs, "-> reprojetando para o primeiro.")
        alvo = linhas_ok[0].crs
        todas = pd.concat([l.to_crs(alvo) if (alvo and l.crs) else l for l in linhas_ok], ignore_index=True)
        todas = gpd.GeoDataFrame(todas, geometry="geometry", crs=alvo)
        todas.to_file(SAIDA_GPKG, driver="GPKG")
        niv = np.unique(np.round(todas.cota))
        print(f"  OK: {len(todas)} curvas salvas em {SAIDA_GPKG}")
        print(f"      cotas {todas.cota.min():.0f} a {todas.cota.max():.0f} m | equidistância típica "
              f"{np.median(np.diff(niv)):.0f} m | CRS: {alvo.to_string() if alvo else 'SEM CRS'}")
        if alvo is None:
            print("      ATENÇÃO: sem CRS. No notebook preencha CRS_CURVAS com o datum ORIGINAL da carta "
                  "('EPSG:22522' Córrego Alegre UTM 22S ou 'EPSG:29192' SAD69 UTM 22S; confira o PDF/metadado da folha).")
        print("      Próximo passo: suba este arquivo no notebook (CURVAS = '/content/curvas_cartas.gpkg').")
    else:
        print("  As curvas foram baixadas, mas NÃO encontrei altitude (campo numérico ou Z).")
        for folha, n, motivo in sem_cota:
            print(f"    - {folha}: {n} linhas de hipsografia ({motivo})")
        if textos:
            print(f"    - Há {textos} textos numéricos (as cotas impressas nas curvas mestras).")
        if rotular and textos:
            rotular_dgns(crs_dgn)
        else:
            print("  Alternativa: as cartas matriciais (escaneadas) estão em .../folhas_topograficas/matriciais/ .")


def rotular_dgns(crs_dgn="EPSG:22522"):
    """Rotulagem automática das curvas (DGN 'hp' sem altitude) -> SAIDA_GPKG + PNG de conferência por folha."""
    from . import rotular as R
    print("\n" + "=" * 72, "\n5) ROTULAGEM AUTOMÁTICA DAS CURVAS\n" + "=" * 72)
    print(f"  DGN em km (UTM 22S), datum assumido: {crs_dgn}. Âncoras = textos 350/400/450 colados nas curvas mestras.")
    dgns = sorted(glob.glob(os.path.join(DEST, "**", "*hp", "*.dgn"), recursive=True))
    partes, resumo = [], []
    for dgn in dgns:
        folha = os.path.basename(os.path.dirname(os.path.dirname(dgn)))
        try:
            g, info = R.rotular_folha_m3(dgn, verbose=True)
        except Exception as e:
            print(f"  ! {os.path.basename(dgn)}: não rotulada ({type(e).__name__}: {str(e)[:100]})")
            continue
        g["folha"] = folha
        png = os.path.join(os.path.dirname(SAIDA_GPKG) or ".", f"rotulagem_{folha}.png")
        try:
            R.png_conferencia(g, png, f"{folha} | equidistância {info['delta']:.2f} m")
        except Exception:
            pass
        partes.append(g)
        resumo.append((folha, info["delta"], info["pct_firmes"], len(g)))
    if not partes:
        print("  Nenhuma folha pôde ser rotulada."); return None
    todas = pd.concat(partes, ignore_index=True)
    firmes = todas[todas.confianca >= 2].copy()
    firmes = gpd.GeoDataFrame(firmes[["cota", "confianca", "nivel", "folha", "geometry"]], geometry="geometry", crs=crs_dgn)
    firmes.to_file(SAIDA_GPKG, driver="GPKG")
    print("\n  RESUMO POR FOLHA:")
    for folha, d, pf, n in resumo:
        print(f"    {folha:<18} equidistância {d:5.2f} m | {pf:3.0f}% das {n} curvas com valor firme")
    print(f"\n  OK: {len(firmes)} curvas com cota salvas em {SAIDA_GPKG} (cotas {firmes.cota.min():.0f} a {firmes.cota.max():.0f} m)")
    print("  Curvas ambíguas (confiança baixa) ficam FORA do MDT; veja os PNG rotulagem_*.png (pontilhado magenta).")
    return firmes


def hidrografia_ibge(dest=None, saida_gpkg=None, crs_dgn="EPSG:22522"):
    """Hidrografia ORIGINAL das cartas (DGN 'hd', pré-1981) -> GPKG com camadas 'linhas' e 'poligonos' (CRS da carta)."""
    global DEST
    from shapely.geometry import LineString
    dest = dest or DEST
    DEST = dest
    dgns = sorted(glob.glob(os.path.join(dest, "**", "*hd", "*.dgn"), recursive=True))
    if not dgns:
        for f in descobrir_folhas():
            baixar_folha(f)
        dgns = sorted(glob.glob(os.path.join(dest, "**", "*hd", "*.dgn"), recursive=True))
    if not dgns:
        print("  Sem DGN de hidrografia (hd) das cartas IBGE: camada pulada."); return None
    ruido = {22, 53, 61, 63}                      # moldura, grade, textos
    lin, pol = [], []
    for dgn in dgns:
        g = None
        for enc in (None, "latin-1"):
            try:
                g = gpd.read_file(dgn, **({"encoding": enc} if enc else {})); break
            except Exception:
                continue
        if g is None:
            print("  ! não li", dgn); continue
        folha = os.path.basename(os.path.dirname(os.path.dirname(dgn)))
        import shapely
        g["geometry"] = shapely.transform(g.geometry.values, lambda a: a * 1000.0)     # DGN em km
        for geo, lv in zip(g.geometry, g["Level"]):
            if geo is None or int(lv) in ruido or geo.geom_type == "Point":
                continue
            if geo.geom_type == "Polygon":
                pol.append(dict(nivel=int(lv), folha=folha, geometry=geo))
                continue
            for p in ([geo] if geo.geom_type == "LineString" else list(getattr(geo, "geoms", []))):
                if p.geom_type == "LineString" and p.length > 20:
                    lin.append(dict(nivel=int(lv), folha=folha, geometry=LineString(p.coords)))
    L = gpd.GeoDataFrame(lin, geometry="geometry", crs=crs_dgn)
    P = gpd.GeoDataFrame(pol, geometry="geometry", crs=crs_dgn)
    os.makedirs(os.path.dirname(saida_gpkg) or ".", exist_ok=True)
    if os.path.exists(saida_gpkg):
        os.remove(saida_gpkg)
    L.to_file(saida_gpkg, layer="linhas", driver="GPKG")
    if len(P):
        P.to_file(saida_gpkg, layer="poligonos", driver="GPKG")
    print(f"Hidrografia IBGE (pré-1981): {len(L)} linhas ({L.length.sum()/1000:.0f} km), {len(P)} polígonos, {len(dgns)} folhas -> {saida_gpkg}")
    return L, P


if __name__ == "__main__":
    baixar_curvas_ibge()
