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


def baixar_curvas_ibge(dest=None, saida_gpkg=None):
    """Baixa, inspeciona e consolida as curvas das folhas do IBGE em um GeoPackage."""
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
            print(f"    - Há {textos} textos numéricos (provavelmente as cotas impressas nas curvas mestras).")
        print("  => COLE ESTE RELATÓRIO NA CONVERSA: com a estrutura real eu monto a rotulagem automática")
        print("     (cotas dos textos -> curvas mestras -> curvas intermediárias).")
        print("  Alternativa: as cartas matriciais (escaneadas) estão em .../folhas_topograficas/matriciais/ .")


if __name__ == "__main__":
    baixar_curvas_ibge()
