# Núcleo do pipeline (AOI, Landsat 1979, drenagem, MDT pré-1981, conferência).
# MDT anterior a 1981 - Buritama / Zacarias (SP)
# Fontes usadas: curvas de nível de cartas antigas (você vetoriza) + Landsat MSS 1979 + NA_TD.
# NÃO usa Topodata/SRTM (2000) em nenhuma etapa.
import os, sys, glob, gc, shutil, subprocess, unicodedata, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio import features
from rasterio.transform import from_origin, Affine
from rasterio.windows import from_bounds, Window
from scipy import ndimage as ndi
from scipy.interpolate import griddata
from scipy.ndimage import median_filter, uniform_filter1d, gaussian_filter, binary_erosion
from skimage import exposure
from skimage.filters import unsharp_mask, threshold_otsu
from skimage.morphology import skeletonize, disk
from PIL import Image

warnings.filterwarnings("ignore")
CRS_ALVO = "EPSG:31982"      # SIRGAS 2000 / UTM 22S (cobre Buritama e Zacarias)
NODATA = -9999.0


# ------------------------------------------------------------------ utilidades
def ler_vetor(path):
    """Lê shp/gpkg/geojson. Aceita 'arquivo.gpkg:camada'."""
    if ":" in path and not os.path.exists(path):
        arq, camada = path.rsplit(":", 1)
        return gpd.read_file(arq, layer=camada)
    return gpd.read_file(path)


def preparar_shapefile(caminho, destino):
    """Aceita .shp, .zip ou .rar e devolve o caminho do .shp."""
    if caminho.lower().endswith(".shp"):
        return caminho
    os.makedirs(destino, exist_ok=True)
    if caminho.lower().endswith(".zip"):
        shutil.unpack_archive(caminho, destino)
    else:
        feito = False
        try:
            import libarchive
            with libarchive.file_reader(caminho) as arq:
                for e in arq:
                    if e.isfile:
                        with open(os.path.join(destino, os.path.basename(e.pathname)), "wb") as f:
                            for b in e.get_blocks():
                                f.write(b)
            feito = True
        except Exception:
            pass
        if not feito:
            for cmd in (["unrar", "x", "-o+", caminho, destino + "/"], ["7z", "x", "-y", caminho, "-o" + destino]):
                try:
                    if subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                        feito = True
                        break
                except FileNotFoundError:
                    pass
        if not feito:
            raise RuntimeError("Não consegui abrir o .rar. Descompacte no seu computador e suba o .shp (ou um .zip).")
    shps = glob.glob(os.path.join(destino, "**", "*.shp"), recursive=True)
    if not shps:
        raise RuntimeError("Nenhum .shp encontrado dentro de " + caminho)
    return shps[0]


def normalizar(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


# ------------------------------------------------------------------ ETAPA 1: AOI
def baixar_municipios(nomes, uf="35"):
    import requests
    base = "https://servicodados.ibge.gov.br/api"
    lista = requests.get(f"{base}/v1/localidades/estados/{uf}/municipios", timeout=60).json()
    geoms, achados = [], []
    for nome in nomes:
        m = [x for x in lista if normalizar(x["nome"]) == normalizar(nome)]
        if not m:
            raise ValueError(f"município '{nome}' não encontrado")
        r = requests.get(f"{base}/v3/malhas/municipios/{m[0]['id']}",
                         params={"formato": "application/vnd.geo+json", "qualidade": "maxima"}, timeout=180)
        r.raise_for_status()
        g = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4674").to_crs(CRS_ALVO)
        geoms.append(shapely.union_all(g.geometry.values))
        achados.append(f"{m[0]['nome']} ({m[0]['id']})")
    return shapely.union_all(geoms), achados


def montar_aoi(municipios, natd, margem_km=3.0, fallback_km=12.0, aoi_arquivo=None, usar_ibge=True):
    """AOI = municípios (IBGE) + NA_TD, com margem. Sem internet: NA_TD + fallback_km."""
    partes, fonte = [], []
    if aoi_arquivo:
        partes.append(shapely.union_all(gpd.read_file(aoi_arquivo).to_crs(CRS_ALVO).geometry.values))
        fonte.append(f"arquivo {aoi_arquivo}")
    elif usar_ibge:
        try:
            g, ach = baixar_municipios(municipios)
            partes.append(g); fonte.append("IBGE: " + ", ".join(ach))
        except Exception as e:
            print(f"  ! Limites do IBGE indisponíveis ({type(e).__name__}).")
    nat_uni = shapely.union_all(natd.geometry.values) if natd is not None else None
    if not partes:
        if nat_uni is None:
            raise RuntimeError("Sem IBGE e sem NA_TD: informe aoi_arquivo.")
        partes.append(nat_uni.buffer(fallback_km * 1000))
        fonte.append(f"AOI APROXIMADA = NA_TD + {fallback_km:g} km (sem IBGE)")
    elif nat_uni is not None:
        partes.append(nat_uni); fonte.append("NA_TD")
    aoi = shapely.union_all(partes).buffer(margem_km * 1000)
    b = gpd.GeoSeries([aoi], crs=CRS_ALVO).to_crs("EPSG:4674").total_bounds
    print("AOI =", " + ".join(fonte), f"+ margem {margem_km:g} km")
    print(f"   área {aoi.area/1e6:.0f} km² | lon {b[0]:.3f} a {b[2]:.3f} | lat {b[1]:.3f} a {b[3]:.3f}")
    return aoi


# ------------------------------------------------------------------ ETAPA 2: Landsat 1979
def recortar_landsat(landsat, aoi, saida):
    with rasterio.open(landsat) as src:
        b = gpd.GeoSeries([aoi], crs=CRS_ALVO).to_crs(src.crs).total_bounds
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        dados = src.read(window=win)
        tr = src.window_transform(win)
        nd = src.nodata if src.nodata is not None else 0
        prof = src.profile.copy()
    prof.update(height=dados.shape[1], width=dados.shape[2], transform=tr, compress="lzw", tiled=False)
    prof.pop("blockxsize", None); prof.pop("blockysize", None)
    util = (~np.all(dados == nd, axis=0)).mean() * 100
    print(f"Landsat recortado: {dados.shape[2]} x {dados.shape[1]} px | {util:.0f}% com dados")
    if util < 50:
        print("  ATENÇÃO: a cena cobre só parte da AOI. Confira path/row da cena.")
    os.makedirs(saida, exist_ok=True)
    arq = os.path.join(saida, "landsat_1979_recorte.tif")
    with rasterio.open(arq, "w", **prof) as d:
        d.write(dados)
    return arq


def _esticar_banda(b, core, p0, p1):
    lo, hi = np.percentile(b[core], (p0, p1))
    return np.clip((b - lo) / (hi - lo + 1e-9), 0, 1).astype("float32")


def melhorar_landsat(entrada, saida, bandas=(1, 2, 3), upscale=2, p_min=1, p_max=99, gama=0.95,
                     saturacao=1.25, clahe_clip=0.004, nitidez=0.8):
    """Destriping, ruído, DOS1, estiramento, balanço de cor, CLAHE, saturação, gama,
    nitidez e reamostragem cúbica. Baixo consumo de memória (float32, upscale no final)."""
    with rasterio.open(entrada) as src:
        arr = src.read(list(bandas)).astype("float32")
        nodata, transform, crs = src.nodata, src.transform, src.crs
    h, w = arr.shape[1:]
    nd = nodata if nodata is not None else 0
    validos = ~(np.all(arr == nd, axis=0) | np.any(~np.isfinite(arr), axis=0))
    core = binary_erosion(validos, iterations=6)
    antes = np.stack([_esticar_banda(arr[i], core, 2, 98) for i in range(3)]); antes[:, ~validos] = 0
    x = np.empty_like(arr)
    for i in range(3):
        b = arr[i].copy()
        m = np.where(validos, b, np.nan)
        med = np.nanmedian(m, axis=1)
        med = np.where(np.isnan(med), np.nanmean(med), med)
        b = b - (med - uniform_filter1d(med, size=61, mode="nearest"))[:, None]
        del m
        b[~validos] = b[core].mean()
        b = gaussian_filter(median_filter(b, size=3), 0.7)
        b = np.clip(b - np.percentile(b[core], 0.1), 0, None)          # DOS1
        x[i] = _esticar_banda(b, core, p_min, p_max)
    del arr; gc.collect()
    med_b = np.array([x[i][core].mean() for i in range(3)])
    for i in range(3):
        x[i] = np.clip(x[i] * (med_b.mean() / (med_b[i] + 1e-9)), 0, 1)
        x[i] = _esticar_banda(x[i], core, 0.5, 99.5)
    for i in range(3):
        x[i][~validos] = x[i][core].mean()
        x[i] = exposure.equalize_adapthist(x[i], clip_limit=clahe_clip, kernel_size=max(h, w) // 8).astype("float32")
    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    for i in range(3):
        x[i] = np.clip(lum + (x[i] - lum) * saturacao, 0, 1) ** (1.0 / gama)
    del lum
    for i in range(3):
        x[i] = np.clip(unsharp_mask(x[i], radius=2, amount=nitidez), 0, 1).astype("float32")
        x[i][~validos] = 0
    W, H = w * upscale, h * upscale
    if upscale > 1:
        transform = transform * Affine.scale(1.0 / upscale)
    mask_big = np.asarray(Image.fromarray(validos.astype("uint8") * 255).resize((W, H), Image.NEAREST)) > 0 \
        if upscale > 1 else validos
    perfil = dict(driver="GTiff", dtype="uint8", count=3, nodata=0, compress="lzw", width=W, height=H,
                  transform=transform, crs=crs, tiled=True, blockxsize=256, blockysize=256, photometric="RGB")
    with rasterio.open(saida, "w", **perfil) as dst:
        for i in range(3):
            b8 = (x[i] * 255).round().astype("uint8")
            if upscale > 1:
                b8 = np.asarray(Image.fromarray(b8).resize((W, H), Image.BICUBIC))
            dst.write(np.where(mask_big, np.maximum(b8, 1), 0).astype("uint8"), i + 1)
    print(f"Landsat melhorado: {saida} ({os.path.getsize(saida)/1e6:.1f} MB, {W}x{H}, pixel {abs(transform.a):.0f} m)")
    return saida


def extrair_drenagem(entrada, pasta, modo="todas", ajuste_limiar=1.0, area_min_px=12, fechar_px=1,
                     agua_larga_px=3, esporao_min_px=8, compr_min_px=15, rotulo="1979", poligono="larga", mascara=None, textura_px=0, agua_max_pct=None,
                     area_min_km2=0.0):
    """Água (pixels escuros em todas as bandas) -> esqueleto -> linhas de drenagem + polígonos de água.

    Opcionais (usados nas fotos históricas p&b): `mascara` = geometria em EPSG:31982 que limita a busca (ex.: corredor
    do rio); `textura_px` > 0 exige superfície LISA (água) e descarta escuro texturizado (mata/sombra);
    `agua_max_pct` aperta o limiar até a água cair abaixo dessa % da área analisada; `area_min_km2` remove manchas pequenas."""
    from skan import Skeleton, summarize
    from shapely.geometry import LineString, shape

    def limpar_pequenos(mask, min_px):
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            return mask
        tam = np.bincount(lab.ravel()); ok = tam >= min_px; ok[0] = False
        return ok[lab]

    with rasterio.open(entrada) as src:
        arr = src.read().astype("float32"); nodata, transform, crs = src.nodata, src.transform, src.crs
        px = abs(src.res[0]); geog = src.crs is not None and src.crs.is_geographic
        km2_px = (px * px / 1e6) if not geog else 0.0
    n, h, w = arr.shape
    nd = nodata if nodata is not None else 0
    validos = ~(np.all(arr == nd, axis=0) | np.any(~np.isfinite(arr), axis=0))
    if mascara is not None:
        from rasterio.features import rasterize
        import geopandas as _gpd
        mg = _gpd.GeoSeries([mascara], crs=CRS_ALVO).to_crs(crs).iloc[0]
        validos &= rasterize([(mg, 1)], out_shape=(h, w), transform=transform, fill=0, dtype="uint8").astype(bool)
        print(f"  máscara (corredor): {validos.mean()*100:.0f}% da imagem recortada")
    core = binary_erosion(validos, iterations=4)

    def norm(b):
        lo, hi = np.percentile(b[core], (1, 99))
        return np.clip((b - lo) / (hi - lo + 1e-9), 0, 1)

    idx = np.max(np.stack([norm(arr[i]) for i in range(n)]), axis=0) if modo == "todas" else norm(arr[int(modo) - 1])
    del arr
    liso = None
    if textura_px:
        from scipy.ndimage import uniform_filter
        m1 = uniform_filter(idx, textura_px); m2 = uniform_filter(idx * idx, textura_px)
        desvio = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    idx = gaussian_filter(idx, 0.8); idx[~validos] = 1.0
    t1 = threshold_otsu(idx[core]); cauda = idx[core][idx[core] < t1]
    t2 = threshold_otsu(cauda) if cauda.size > 100 else t1
    limiar = t2 * ajuste_limiar
    if textura_px:
        esc = (idx < limiar) & core
        if esc.sum() > 500:
            lim_sd = threshold_otsu(desvio[esc])
            liso = desvio <= lim_sd
            print(f"  textura: {100*(esc & liso).sum()/esc.sum():.0f}% dos pixels escuros são lisos (água); "
                  f"o resto (mata/sombra) é descartado")
    def _mascara_agua(lim):
        a = (idx < lim) & validos
        return a & liso if liso is not None else a
    agua = _mascara_agua(limiar)
    if agua_max_pct:
        for _ in range(10):
            if agua[core].mean() * 100 <= agua_max_pct:
                break
            limiar *= 0.9
            agua = _mascara_agua(limiar)
        print(f"  limiar final {limiar:.3f} (água limitada a {agua_max_pct}% da área analisada)")
    frac = agua[core].mean() * 100
    print(f"Água bruta: {frac:.1f}% da cena (limiar {limiar:.3f})")
    if frac > 15:
        print("  ATENÇÃO: muita água detectada; reduza ajuste_limiar (ex.: 0.8) ou use modo=<número da banda>.")
    if frac < 0.2:
        print("  ATENÇÃO: quase nada detectado; aumente ajuste_limiar (ex.: 1.3).")
    agua = limpar_pequenos(agua, max(area_min_px, int(area_min_km2 / km2_px) if km2_px else 0))
    if fechar_px:
        agua = ndi.binary_closing(agua, structure=disk(fechar_px)) & validos
    agua |= (~agua & ~limpar_pequenos(~agua, area_min_px)) & validos
    larga = limpar_pequenos(ndi.binary_opening(agua, structure=disk(agua_larga_px)), area_min_px * 4)
    print(f"Água: {agua.sum()*km2_px:.1f} km² | água larga (rio largo/reservatório): {larga.sum()*km2_px:.1f} km²")

    sk = skeletonize(agua)
    skel = Skeleton(sk)
    df = summarize(skel, separator="-") if "separator" in summarize.__code__.co_varnames else summarize(skel)
    df.columns = [c.replace("_", "-") for c in df.columns]
    tipo, comp = df["branch-type"].values, df["branch-distance"].values
    linhas, comprimentos, em_larga = [], [], []
    for i in range(skel.n_paths):
        if tipo[i] == 1 and comp[i] < esporao_min_px:
            continue
        if tipo[i] in (0, 3) and comp[i] < compr_min_px:
            continue
        rc = skel.path_coordinates(i)
        if len(rc) < 2:
            continue
        xs, ys = rasterio.transform.xy(transform, rc[:, 0], rc[:, 1], offset="center")
        linhas.append(LineString(zip(xs, ys)).simplify(px * 0.6))
        comprimentos.append(comp[i])
        rr = np.clip(rc[:, 0].astype(int), 0, h - 1); cc = np.clip(rc[:, 1].astype(int), 0, w - 1)
        em_larga.append(bool(larga[rr, cc].mean() > 0.5))
    gl = gpd.GeoDataFrame({"comp_m": np.round(np.array(comprimentos) * px, 1), "em_reservatorio": em_larga},
                          geometry=linhas, crs=crs)
    print(f"Drenagem: {len(gl)} linhas | {gl['comp_m'].sum()/1000:.1f} km "
          f"(fora de água larga: {gl.loc[~gl['em_reservatorio'], 'comp_m'].sum()/1000:.1f} km)")
    pm = agua if poligono == "toda" else larga       # "toda": todo o corpo d'água (rio antes do alagamento)
    polis = [shape(g) for g, v in features.shapes(pm.astype("uint8"), mask=pm, transform=transform) if v == 1]
    ga = gpd.GeoDataFrame(geometry=polis, crs=crs)
    if len(ga):
        ga = ga.dissolve().explode(index_parts=False).reset_index(drop=True)
        ga["area_km2"] = (ga.geometry.area / 1e6).round(3)
    os.makedirs(pasta, exist_ok=True)
    gpkg = os.path.join(pasta, f"drenagem_{rotulo}.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    gl.to_file(gpkg, layer="drenagem", driver="GPKG")
    if len(ga):
        ga.to_file(gpkg, layer="agua", driver="GPKG")
    shp = os.path.join(pasta, "shp"); shutil.rmtree(shp, ignore_errors=True); os.makedirs(shp)
    gl.to_file(os.path.join(shp, f"drenagem_{rotulo}.shp"))
    shutil.make_archive(os.path.join(pasta, f"drenagem_{rotulo}_shp"), "zip", shp)
    print(f"Drenagem {rotulo}:", gpkg)
    return gl.to_crs(CRS_ALVO), (ga.to_crs(CRS_ALVO) if len(ga) else ga)


# ------------------------------------------------------------------ ETAPA 3: MDT pré-1981 (só cartas)
def achar_campo(gdf, pedido=None):
    if pedido:
        return pedido
    low = {str(c).lower(): c for c in gdf.columns}
    for c in ["cota", "elevacao", "elevação", "elev", "altitude", "alt", "z", "height", "contour",
              "valor", "value", "nivel", "altura", "elevation"]:
        if c in low:
            return low[c]
    nums = [c for c in gdf.columns if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])]
    return nums[0] if nums else None


def _para_sirgas(gdf, crs_origem):
    if crs_origem:
        gdf = gdf.set_crs(crs_origem, allow_override=True)
    elif gdf.crs is None:
        raise ValueError("O arquivo não tem CRS: informe CRS_CURVAS (ex.: 'EPSG:22522' Córrego Alegre UTM 22S, "
                         "'EPSG:29192' SAD69 UTM 22S).")
    return gdf.to_crs(CRS_ALVO)


def _pontos_de_linhas(gdf, campo, passo):
    xs, ys, zs = [], [], []
    for geom, z in zip(gdf.geometry, gdf[campo] if campo else [None] * len(gdf)):
        if geom is None or geom.is_empty:
            continue
        for g in (list(geom.geoms) if hasattr(geom, "geoms") else [geom]):
            if g.length == 0:
                continue
            d = np.linspace(0, g.length, max(int(g.length // passo), 1) + 1)
            co = shapely.get_coordinates(shapely.line_interpolate_point(g, d), include_z=False)
            zz = np.full(len(co), z, dtype="float64") if z is not None else np.array([g.coords[0][2]] * len(co))
            xs.append(co[:, 0]); ys.append(co[:, 1]); zs.append(zz)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def _cotas_na_hidro(hid, curvas, campo, passo):
    """Cota do talvegue = interpolação das cotas onde a drenagem cruza as curvas (corrige fundos de vale planos)."""
    vazio = (np.array([]), np.array([]), np.array([]))
    if not campo:
        return vazio
    geoms, cot = curvas.geometry.values, curvas[campo].astype(float).values
    tree = shapely.STRtree(geoms)
    X, Y, Z = [], [], []
    for linha in hid.geometry:
        if linha is None or linha.is_empty:
            continue
        for ln in (list(linha.geoms) if hasattr(linha, "geoms") else [linha]):
            if ln.length == 0:
                continue
            ds, zs = [], []
            for i in tree.query(ln, predicate="intersects"):
                inter = ln.intersection(geoms[i])
                if inter.is_empty:
                    continue
                for pt in shapely.get_parts(inter):
                    if pt.geom_type == "Point":
                        ds.append(ln.project(pt)); zs.append(cot[i])
            if len(ds) < 2:
                continue
            o = np.argsort(ds)
            dd = np.linspace(0, ln.length, max(int(ln.length // passo), 1) + 1)
            zz = np.interp(dd, np.array(ds)[o], np.array(zs)[o])
            co = shapely.get_coordinates(shapely.line_interpolate_point(ln, dd))
            X.append(co[:, 0]); Y.append(co[:, 1]); Z.append(zz)
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(Z)) if X else vazio


def _preencher_nn(Z):
    bad = ~np.isfinite(Z)
    if not bad.any():
        return Z
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return Z[tuple(idx)]


def gerar_mdt_pre81(curvas_path, aoi, saida, crs_origem=None, campo=None, pontos_path=None, hidro=None,
                    res=30.0, suavizar=1.0, max_dist=1500.0):
    """MDT somente das curvas de nível das cartas antigas (TIN), em SIRGAS 2000 UTM 22S.
    Células a mais de max_dist metros de qualquer curva ficam SEM DADOS (nada é inventado/misturado)."""
    os.makedirs(saida, exist_ok=True)
    curvas = ler_vetor(curvas_path)
    curvas = curvas[curvas.geometry.notna() & ~curvas.geometry.is_empty]
    campo = achar_campo(curvas, campo)
    if campo is None and not curvas.geometry.has_z.any():
        raise ValueError("Não achei o campo de cota nas curvas. Crie um campo numérico 'cota'.")
    print(f"Curvas: {len(curvas)} feições | campo de cota: {campo or '(Z da geometria)'} | CRS: {crs_origem or curvas.crs}")
    curvas = _para_sirgas(curvas, crs_origem)
    x, y, z = _pontos_de_linhas(curvas, campo, res / 2)
    niveis = np.unique(np.round(z))
    eq = np.median(np.diff(niveis)) if len(niveis) > 1 else float("nan")
    print(f"Cotas: {z.min():.0f} a {z.max():.0f} m | equidistância típica: {eq:.0f} m")
    if eq > 10:
        print("  Obs.: equidistância > 10 m: em relevo suave o MDT sai grosseiro (limite das cartas).")

    ex, ey, ez = [], [], []
    if pontos_path:
        pts = _para_sirgas(ler_vetor(pontos_path), crs_origem)
        co = shapely.get_coordinates(pts.geometry)
        ex.append(co[:, 0]); ey.append(co[:, 1]); ez.append(pts[achar_campo(pts, campo)].astype(float).values)
        print(f"Pontos cotados: {len(pts)}")
    if hidro is not None and len(hidro):
        hx, hy, hz = _cotas_na_hidro(hidro.to_crs(CRS_ALVO), curvas, campo, res / 2)
        if len(hx):
            ex.append(hx); ey.append(hy); ez.append(hz)
            print(f"Talvegue (drenagem 1979) cotado: {len(hx)} pontos, {hz.min():.0f} a {hz.max():.0f} m")
        else:
            print("Drenagem 1979 sem cruzamentos suficientes com as curvas: talvegue não cotado.")
    if ex:
        x = np.r_[x, np.concatenate(ex)]; y = np.r_[y, np.concatenate(ey)]; z = np.r_[z, np.concatenate(ez)]

    r = res
    bx0, by0, bx1, by1 = aoi.bounds
    minx, maxx = np.floor(bx0 / r) * r, np.ceil(bx1 / r) * r
    miny, maxy = np.floor(by0 / r) * r, np.ceil(by1 / r) * r
    w, h = int((maxx - minx) / r) + 1, int((maxy - miny) / r) + 1
    transform = from_origin(minx, maxy + r, r, r)
    print(f"Grade: {w} x {h} px a {r:g} m ({(maxx-minx)/1000:.0f} x {(maxy-miny)/1000:.0f} km)")
    mx = 20 * r
    sel = (x >= minx - mx) & (x <= maxx + mx) & (y >= miny - mx) & (y <= maxy + mx)
    if not sel.any():
        raise ValueError("As curvas não caem dentro da região. Confira CRS_CURVAS (datum) e o georreferenciamento.")
    x, y, z = x[sel], y[sel], z[sel]

    conhec = features.rasterize([(g, 1) for g in curvas.geometry], out_shape=(h, w), transform=transform,
                                fill=0, dtype="uint8", all_touched=True).astype(bool)
    if ex:
        exx, eyy = np.concatenate(ex), np.concatenate(ey)
        ci = ((maxy + r - eyy) // r).astype(int); cj = ((exx - minx) // r).astype(int)
        ok = (ci >= 0) & (ci < h) & (cj >= 0) & (cj < w)
        conhec[ci[ok], cj[ok]] = True
    dist = ndi.distance_transform_edt(~conhec) * r
    sem_cob = dist > max_dist
    pct = 100 * (1 - sem_cob.mean())
    print(f"Cobertura das curvas na região (<= {max_dist:g} m de uma curva): {pct:.0f}%")
    if pct < 90:
        print("  ATENÇÃO: parte da região não tem curvas e ficará SEM DADOS. Vetorize as cartas dessa região.")

    cx = minx + (np.arange(w) + 0.5) * r
    cy = maxy + r - (np.arange(h) + 0.5) * r
    gx, gy = np.meshgrid(cx, cy)
    dem = griddata((x, y), z, (gx, gy), method="linear").astype("float32")
    dem = _preencher_nn(dem)
    if suavizar > 0:
        dem = ndi.gaussian_filter(dem, suavizar).astype("float32")
    dem[sem_cob] = np.nan

    arq = os.path.join(saida, "mdt_pre81.tif")
    with rasterio.open(arq, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=CRS_ALVO,
                       transform=transform, nodata=NODATA, compress="deflate", predictor=3, tiled=True) as d:
        d.write(np.where(np.isfinite(dem), dem, NODATA).astype("float32"), 1)
    print(f"\nMDT pré-1981: {arq}")
    print(f"  altitude {np.nanmin(dem):.1f} a {np.nanmax(dem):.1f} m | média {np.nanmean(dem):.1f} m | "
          f"datum vertical: Imbituba | horizontal: SIRGAS 2000 UTM 22S")
    return dict(dem=dem, transform=transform, arq=arq, res=r)


# ------------------------------------------------------------------ ETAPA 4: conferência
def sombreado(dem, res, az=315, alt=45):
    gy, gx = np.gradient(dem.astype("float64"), res)
    slope = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    azr, altr = np.radians(az), np.radians(alt)
    return np.clip(np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - asp), 0, 1)


def qa_mdt(mdt, natd):
    """Elevação do MDT pré-81 dentro de cada classe do NA_TD (indicador de consistência)."""
    if natd is None or mdt is None:
        return
    dem, tr, r = mdt["dem"], mdt["transform"], mdt["res"]
    h, w = dem.shape
    col = "CLASSE" if "CLASSE" in natd.columns else None
    print("\nElevação do MDT pré-81 dentro do NA_TD (m):")
    print(f"  {'classe':<26}{'células':>8}{'mín':>8}{'mediana':>9}{'máx':>8}")
    grupos = natd.groupby(col) if col else [("todos", natd)]
    for nome, g in grupos:
        m = features.rasterize([(geo, 1) for geo in g.geometry], out_shape=(h, w), transform=tr, fill=0,
                               dtype="uint8").astype(bool) & np.isfinite(dem)
        if m.sum() == 0:
            continue
        v = dem[m]
        print(f"  {str(nome)[:25]:<26}{m.sum():>8}{v.min():>8.1f}{np.median(v):>9.1f}{v.max():>8.1f}")
    print("  (Se a 'FAIXA DE DESAPROPRIAÇÃO' acompanha a cota do reservatório, a mediana deve ficar perto da cota\n"
          "   de operação e com pouca variação; desvios grandes sugerem erro de georreferenciamento/datum.)")


def mapa_conferencia(aoi, natd, mdt, drenagem, landsat_melhorado, pasta):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 10))
    if mdt is not None:
        dem, tr, r = mdt["dem"], mdt["transform"], mdt["res"]
        h, w = dem.shape
        ext = [tr.c, tr.c + w * r, tr.f - h * r, tr.f]
        hs = sombreado(np.nan_to_num(dem, nan=float(np.nanmean(dem))), r)
        ax.imshow(hs, cmap="gray", extent=ext)
        im = ax.imshow(dem, cmap="terrain", alpha=0.5, extent=ext)
        plt.colorbar(im, ax=ax, shrink=0.7, label="MDT pré-1981 (m)")
        titulo = "MDT pré-1981 (cartas antigas)"
    elif landsat_melhorado and os.path.exists(landsat_melhorado):
        with rasterio.open(landsat_melhorado) as s:
            f = max(1, max(s.width, s.height) // 1500)
            rgb = np.moveaxis(s.read(out_shape=(3, s.height // f, s.width // f)), 0, -1)
            b = s.bounds
        ax.imshow(rgb, extent=[b.left, b.right, b.bottom, b.top])
        titulo = "Landsat MSS 1979 melhorado (sem MDT: faltam as curvas das cartas)"
    else:
        titulo = "Região"
    gpd.GeoSeries([aoi], crs=CRS_ALVO).boundary.plot(ax=ax, color="k", linewidth=1.0)
    if natd is not None:
        natd.boundary.plot(ax=ax, color="red", linewidth=0.6)
    if drenagem is not None and len(drenagem):
        drenagem.plot(ax=ax, color="cyan", linewidth=0.6)
    ax.set_title(f"{titulo}\npreto = região | vermelho = NA_TD | ciano = drenagem Landsat 1979")
    ax.set_xlabel("E (m) SIRGAS 2000 / UTM 22S"); ax.set_ylabel("N (m)")
    png = os.path.join(pasta, "mapa_conferencia.png")
    plt.savefig(png, dpi=95, bbox_inches="tight")
    plt.show()
    print("Mapa:", png)
    return png


def zipar(pasta):
    z = shutil.make_archive(pasta.rstrip("/"), "zip", pasta)
    print(f"Pacote final: {z} ({os.path.getsize(z)/1e6:.1f} MB)")
    return z
