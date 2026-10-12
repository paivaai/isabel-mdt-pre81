# -*- coding: utf-8 -*-
"""Cartas topográficas ESCANEADAS/impressas (PDF/TIF) como camada de validação no webmap.

1) `inspecionar_cartas()` percorre a pasta `editoradas/escala_50mil` do IBGE (precisa de internet:
   rodar pelo workflow "Cartas escaneadas") e baixa os arquivos das folhas da região, gravando o
   inventário em data/cartas_raster/inventario.txt.
2) `camadas_cartas(cfg)` transforma em camadas do webmap:
   - raster JÁ georreferenciado (GeoTIFF / GeoPDF): usado direto;
   - PDF/imagem sem georreferência: precisa de `bounds` (E/N mín/máx em UTM, do CANTO do desenho da carta)
     e `crs` na config (webmap.cartas). Sem isso o arquivo é listado mas NÃO entra no mapa
     (não invento posição).
"""
import os, re, io, glob, shutil, subprocess, warnings
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from .ibge import listar, S, baixar

warnings.filterwarnings("ignore")
BASE_ED = os.environ.get(
    "IBGE_BASE_EDITORADAS",
    "https://geoftp.ibge.gov.br/cartas_e_mapas/folhas_topograficas/editoradas/escala_50mil/")
BASE_FOTOLITO = BASE_ED.replace("editoradas", "fotolitos")
CODIGOS = ["26322", "26333", "26323", "26321", "26331", "26332", "26334", "26311", "26312", "26313", "26314"]
PALAVRAS = ["buritama", "zacarias", "avanhandava", "birigui", "aracatuba", "penapolis", "brejo", "bilac", "gabriel"]
EXT = (".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".zip", ".jp2")


def _varrer(base, prof=3, log=None):
    """Lista recursivamente (url, tamanho desconhecido) até `prof` níveis."""
    achados = []
    def rec(url, n):
        try:
            nomes = listar(url)
        except Exception as e:
            log and log.append(f"! {url}: {type(e).__name__}")
            return
        for nome in nomes:
            if nome.endswith("/"):
                if n < prof:
                    rec(url + nome, n + 1)
            elif nome.lower().endswith(EXT):
                achados.append(url + nome)
    rec(base, 1)
    return achados


def _interessa(url):
    u = url.lower()
    return any(c in u for c in CODIGOS) or any(p in u for p in PALAVRAS)


def inspecionar_cartas(dest="data/cartas_raster"):
    os.makedirs(dest, exist_ok=True)
    log = []
    todos = []
    for base in (BASE_ED, BASE_FOTOLITO):
        arqs = _varrer(base, log=log)
        log.append(f"{base}: {len(arqs)} arquivos ({', '.join(sorted({os.path.splitext(a)[1].lower() for a in arqs}))})")
        todos += arqs
    usados = [u for u in todos if _interessa(u)]
    log.append(f"\nArquivos que casam com as folhas/municípios da região: {len(usados)}")
    for u in usados:
        log.append("  " + u)
    baixados = []
    for u in usados[:40]:
        alvo = os.path.join(dest, re.sub(r"[^\w.\-]", "_", u.split("escala_50mil/")[-1]))
        try:
            baixar(u, alvo)
            baixados.append(alvo)
            log.append(f"  baixado: {alvo} ({os.path.getsize(alvo)/1e6:.1f} MB)")
            if alvo.lower().endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(alvo) as z:
                    z.extractall(alvo[:-4])
        except Exception as e:
            log.append(f"  ! falhou {u}: {type(e).__name__}")
    if not todos:
        log.append("\nNada listado nas pastas editoradas/fotolitos (verifique o relatório; talvez a estrutura seja outra).")
    txt = "\n".join(log)
    open(os.path.join(dest, "inventario.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    return baixados


# ---------------------------------------------------------------- camadas
def _pdf_para_png(pdf, dpi=150):
    """Rasteriza a 1ª página (pdftoppm, se existir). Retorna caminho do PNG ou None."""
    if shutil.which("pdftoppm") is None:
        return None
    pref = pdf[:-4] + "_p"
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1", "-singlefile", pdf, pref], check=False)
    return pref + ".png" if os.path.exists(pref + ".png") else None


def _a_partir_de_raster(arq, crs=None, bounds=None, max_px=2400):
    """Retorna (png_bytes, bounds_latlng) ou None. Usa georreferência do arquivo, ou crs+bounds da config."""
    from .webmap import _warp, _png_data
    if arq.lower().endswith(".pdf"):
        # GeoPDF? o GDAL lê com georreferência; senão cai no PNG + bounds
        try:
            with rasterio.open(arq) as s:
                if s.crs is not None and not s.transform.is_identity:
                    return _ler_geo(s, max_px)
        except Exception:
            pass
        png = _pdf_para_png(arq)
        if not png:
            return None
        arq = png
    with rasterio.open(arq) as s:
        if s.crs is not None and not s.transform.is_identity:
            return _ler_geo(s, max_px)
        if not (crs and bounds):
            return None
        return _ler_geo(s, max_px, crs=crs, tr=from_bounds(*bounds, s.width, s.height))


def _ler_geo(s, max_px, crs=None, tr=None):
    from .webmap import _warp, _png_data
    f = max(1, int(np.ceil(max(s.width, s.height) / (max_px * 1.5))))
    arr = s.read(out_shape=(s.count, s.height // f, s.width // f), resampling=rasterio.enums.Resampling.average).astype("float32")
    tr = (tr or s.transform) * rasterio.Affine.scale(f, f)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    rgb = arr[:3]
    alfa = (np.any(rgb < 250, axis=0) | np.any(rgb > 0, axis=0)).astype("float32") * 255
    out, bounds, _, _ = _warp(np.concatenate([rgb, alfa[None]]), tr, crs or s.crs, max_px=max_px)
    return _png_data(np.moveaxis(out, 0, -1)), bounds


def camadas_cartas(pasta, itens=None):
    """Lista de (rotulo, png, bounds). `itens`: config webmap.cartas [{nome, arquivo, crs, bounds}]."""
    res, vistos = [], set()
    for it in itens or []:
        arq = it.get("arquivo")
        if arq and os.path.exists(arq):
            vistos.add(os.path.abspath(arq))
            try:
                r = _a_partir_de_raster(arq, it.get("crs"), it.get("bounds"))
            except Exception as e:
                print(f"  ! carta {arq}: {type(e).__name__}: {e}"); r = None
            if r:
                res.append((it.get("nome") or os.path.basename(arq), *r))
            else:
                print(f"  ! carta {arq}: sem georreferência (informe crs e bounds na config).")
    for arq in sorted(glob.glob(os.path.join(pasta, "**", "*"), recursive=True)):
        if arq.lower().endswith((".tif", ".tiff", ".pdf")) and os.path.abspath(arq) not in vistos:
            try:
                r = _a_partir_de_raster(arq)
            except Exception as e:
                print(f"  ! carta {arq}: {type(e).__name__}"); r = None
            if r:
                res.append((os.path.splitext(os.path.basename(arq))[0], *r))
            else:
                print(f"  - carta {os.path.basename(arq)}: sem georreferência; não entra no mapa.")
    return res
