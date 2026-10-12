# -*- coding: utf-8 -*-
"""Rotulagem automática de curvas de nível das cartas IBGE (DGN sem altitude).

Entrada: DGN 'hp' com linhas (curvas) e textos numéricos (curvas mestras 350/400/450 + cotas de pontos).
Ideia:
  1. âncoras  = textos múltiplos de 50 colados numa curva (curva mestra);
  2. grafo de vizinhança entre curvas (raios perpendiculares até a próxima curva);
  3. propagação: curva sem valor recebe z_vizinho + {-D, 0, +D}, escolhendo o candidato mais coerente com uma
     superfície TIN feita só com o que já está rotulado + cotas de pontos;
  4. a equidistância D é escolhida pelo menor erro nas cotas de pontos NÃO usadas na construção (validação).
Nada é inventado: curvas ambíguas ficam com confiança 'baixa' e não entram no MDT por padrão.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

warnings.filterwarnings("ignore")
NIVEIS_CURVA = (1, 2, 55, 56)      # 1/56 = mestras (grossas); 2/55 = intermediárias (finas)
ESCALA_KM = 1000.0                 # DGN do IBGE vem em km (UTM)


# ------------------------------------------------------------------ leitura
def ler_hp(dgn, escala=ESCALA_KM):
    """DGN -> (linhas GeoDataFrame em metros, textos DataFrame x,y,val)."""
    g = None
    for enc in (None, "latin-1"):
        try:
            g = gpd.read_file(dgn, **({"encoding": enc} if enc else {}))
            break
        except Exception:
            continue
    if g is None:
        raise RuntimeError("não consegui ler " + dgn)
    g = g.drop(columns=["ULink"], errors="ignore")
    g["geometry"] = shapely.transform(g.geometry.values, lambda a: a * escala)
    linhas, nivel = [], []
    for geo, lv in zip(g.geometry, g["Level"]):
        if lv not in NIVEIS_CURVA or geo is None:
            continue
        partes = [geo] if geo.geom_type == "LineString" else (list(geo.geoms) if geo.geom_type == "MultiLineString"
                                                              else [geo.exterior] if geo.geom_type == "Polygon" else [])
        for p in partes:
            if p.length > 40:
                linhas.append(LineString(p.coords)); nivel.append(int(lv))
    lin = gpd.GeoDataFrame({"nivel": nivel}, geometry=linhas)
    pts = g[g.geom_type == "Point"]
    tx = []
    for geo, t in zip(pts.geometry, pts["Text"]):
        try:
            v = float(str(t).strip())
        except Exception:
            continue
        tx.append((geo.x, geo.y, v))
    return lin, pd.DataFrame(tx, columns=["x", "y", "val"])


# ------------------------------------------------------------------ vizinhança
def _vizinhos(geoms, passo=120.0, max_pts=10, alcance=1500.0, seed=0):
    """Pares (i, j) de curvas adjacentes: raio perpendicular a partir de i até a primeira curva j."""
    rng = np.random.default_rng(seed)
    tree = STRtree(geoms)
    pares = {}
    for i, ln in enumerate(geoms):
        n = int(min(max_pts, max(2, ln.length / passo)))
        for t in np.linspace(0.05, 0.95, n) * ln.length:
            p = ln.interpolate(t)
            a = ln.interpolate(max(0.0, t - 10)); b = ln.interpolate(min(ln.length, t + 10))
            d = np.array([b.x - a.x, b.y - a.y]); m = np.hypot(*d)
            if m == 0:
                continue
            nr = np.array([-d[1], d[0]]) / m
            for s in (1, -1):
                ray = LineString([(p.x + s * nr[0] * 3, p.y + s * nr[1] * 3),
                                  (p.x + s * nr[0] * alcance, p.y + s * nr[1] * alcance)])
                melhor, dj = None, 1e18
                for j in tree.query(ray, predicate="intersects"):
                    if j == i:
                        continue
                    x = ray.intersection(geoms[j])
                    if x.is_empty:
                        continue
                    dd = ray.project(x if x.geom_type == "Point" else x.geoms[0] if hasattr(x, "geoms") else x.centroid)
                    if dd < dj:
                        dj, melhor = dd, j
                if melhor is not None:
                    k = (i, melhor) if i < melhor else (melhor, i)
                    pares[k] = pares.get(k, 0) + 1
    return pares


def _amostras(geoms, passo=40.0, max_pts=60):
    out = []
    for ln in geoms:
        n = int(min(max_pts, max(3, ln.length / passo)))
        ts = np.linspace(0, ln.length, n)
        out.append(np.array([[ln.interpolate(t).x, ln.interpolate(t).y] for t in ts]))
    return out


# ------------------------------------------------------------------ superfície
class Superficie:
    def __init__(self, xy, z):
        xy = np.asarray(xy); z = np.asarray(z, float)
        self.lin = LinearNDInterpolator(xy, z) if len(z) >= 4 else None
        self.nn = NearestNDInterpolator(xy, z)

    def __call__(self, xy):
        xy = np.asarray(xy)
        v = self.nn(xy)
        if self.lin is not None:
            l = self.lin(xy)
            v = np.where(np.isfinite(l), l, v)
        return v


def _montar_sup(amostras, z, rotulado, spots_xy, spots_z):
    xs, zs = [], []
    for i, ok in enumerate(rotulado):
        if ok:
            xs.append(amostras[i]); zs.append(np.full(len(amostras[i]), z[i]))
    if len(spots_z):
        xs.append(np.asarray(spots_xy)); zs.append(np.asarray(spots_z, float))
    return Superficie(np.vstack(xs), np.concatenate(zs))


# ------------------------------------------------------------------ rotulagem para uma equidistância D
def _rotular(geoms, amostras, pares, ancoras, spots_xy, spots_z, D):
    n = len(geoms)
    z = np.full(n, np.nan)
    conf = np.zeros(n)              # 0 = sem valor; 3 = âncora; 2 = propagada firme; 1 = ambígua
    for i, v in ancoras.items():
        z[i] = v; conf[i] = 3
    viz = [[] for _ in range(n)]
    for (i, j), c in pares.items():
        viz[i].append((j, c)); viz[j].append((i, c))
    for rodada in range(60):
        rot = ~np.isnan(z)
        if rot.sum() < 2:
            break
        sup = _montar_sup(amostras, z, rot, spots_xy, spots_z)
        novos = {}
        for i in np.where(~rot)[0]:
            nb = [(j, c) for j, c in viz[i] if rot[j]]
            if not nb:
                continue
            m = float(np.median(sup(amostras[i])))
            cands = set()
            for j, _ in nb:
                for k in (-D, 0.0, D):
                    cands.add(z[j] + k)
            # candidatos precisam ser coerentes com TODOS os vizinhos já rotulados (|dz| <= D)
            cands = [c for c in cands if all(abs(c - z[j]) <= D + 1e-6 for j, _ in nb)]
            if not cands:
                continue
            cands.sort(key=lambda c: abs(c - m))
            melhor = cands[0]
            seg = abs(cands[1] - m) if len(cands) > 1 else 9 * D
            margem = (seg - abs(melhor - m)) / D
            flancos = len(nb)
            if flancos >= 2 and margem >= 0.25:
                novos[i] = (melhor, 2, margem)
            elif flancos >= 1 and margem >= 0.6:
                novos[i] = (melhor, 2, margem)
        if not novos:
            break
        # rotula só as mais confiáveis a cada rodada (evita erro em cascata)
        ordem = sorted(novos.items(), key=lambda kv: -kv[1][2])
        for i, (v, c, mg) in ordem[: max(1, int(len(ordem) * 0.4))]:
            z[i] = v; conf[i] = c
    # restantes: melhor palpite, marcado ambíguo
    rot = ~np.isnan(z)
    if rot.sum() >= 2:
        sup = _montar_sup(amostras, z, rot, spots_xy, spots_z)
        for i in np.where(~rot)[0]:
            m = float(np.median(sup(amostras[i])))
            z[i] = np.round(m / D) * D; conf[i] = 1
    return z, conf


def _erro_spots(z, conf, amostras, spots_xy, spots_z, min_conf=2):
    ok = conf >= min_conf
    if ok.sum() < 3 or len(spots_z) == 0:
        return np.inf
    sup = _montar_sup(amostras, z, ok, [], [])
    e = sup(np.asarray(spots_xy)) - np.asarray(spots_z)
    return float(np.sqrt(np.mean(e ** 2)))


# ------------------------------------------------------------------ API
def rotular_folha(dgn, deltas=(5.0, 10.0, 12.5, 25.0, 50.0), tol_ancora_m=60.0, verbose=True, delta=None):
    """Retorna (GeoDataFrame com 'cota','confianca','ancora', info dict). Coordenadas em metros (UTM, datum da carta)."""
    lin, tx = ler_hp(dgn)
    geoms = list(lin.geometry)
    nome = os.path.basename(dgn)
    if len(geoms) < 5 or len(tx) < 3:
        raise RuntimeError(f"{nome}: poucas curvas/textos para rotular")
    tree = STRtree(geoms)
    # âncoras: texto múltiplo de 50 colado numa curva
    ancoras, usados = {}, set()
    for k, r in tx.iterrows():
        if r.val % 50 != 0:
            continue
        p = Point(r.x, r.y)
        j = tree.nearest(p)
        if geoms[j].distance(p) <= tol_ancora_m:
            ancoras.setdefault(j, []).append(r.val); usados.add(k)
    amb = {j: v for j, v in ancoras.items() if len(set(v)) > 1}
    ancoras = {j: float(pd.Series(v).mode().iloc[0]) for j, v in ancoras.items()}
    spots = tx.drop(index=list(usados))
    # cotas de pontos: separa 25% para validar (independentes da propagação)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(spots))
    nval = max(3, len(spots) // 4) if len(spots) >= 12 else 0
    val_i, trn_i = idx[:nval], idx[nval:]
    sp_xy_tr = spots.iloc[trn_i][["x", "y"]].values; sp_z_tr = spots.iloc[trn_i]["val"].values
    sp_xy_va = spots.iloc[val_i][["x", "y"]].values; sp_z_va = spots.iloc[val_i]["val"].values
    if verbose:
        print(f"  {nome}: {len(geoms)} curvas | {len(ancoras)} âncoras ({len(amb)} ambíguas) | "
              f"{len(spots)} cotas de pontos ({len(sp_z_va)} reservadas p/ validar)")
    amostras = _amostras(geoms)
    pares = _vizinhos(geoms)
    melhor = None
    for D in ([delta] if delta else deltas):
        if any(abs(v / D - round(v / D)) > 1e-6 for v in ancoras.values()):
            continue
        z, conf = _rotular(geoms, amostras, pares, ancoras, sp_xy_tr, sp_z_tr, D)
        err = _erro_spots(z, conf, amostras, sp_xy_va, sp_z_va, min_conf=1) if len(sp_z_va) else np.inf
        # penaliza equidistâncias que deixam muitas curvas ambíguas
        firm = float((conf >= 2).mean())
        if verbose:
            print(f"     D={D:>5g} m: erro nas cotas de validação = {err:6.2f} m | curvas firmes {100*firm:4.0f}%")
        score = err if np.isfinite(err) else 1e6 - firm
        if melhor is None or score < melhor[0]:
            melhor = (score, D, z, conf, err, firm)
    if melhor is None:
        raise RuntimeError(f"{nome}: nenhuma equidistância compatível com as âncoras")
    _, D, z, conf, err, firm = melhor
    out = gpd.GeoDataFrame({"cota": z, "confianca": conf.astype(int), "nivel": lin["nivel"].values,
                            "ancora": [i in ancoras for i in range(len(geoms))]}, geometry=geoms)
    info = dict(folha=nome, equidistancia=D, erro_validacao_m=err, curvas=len(geoms), ancoras=len(ancoras),
                firmes_pct=100 * firm, spots=len(spots))
    if verbose:
        print(f"  => {nome}: equidistância escolhida {D:g} m | erro de validação {err:.1f} m | firmes {100*firm:.0f}%")
    return out, info


def png_conferencia(gdf, caminho, titulo=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 10))
    g = gdf[gdf.confianca >= 1]
    g.plot(ax=ax, column="cota", cmap="terrain", linewidth=0.8, legend=True, legend_kwds={"shrink": 0.6, "label": "cota (m)"})
    amb = gdf[gdf.confianca == 1]
    if len(amb):
        amb.plot(ax=ax, color="magenta", linewidth=1.0, linestyle=":")
    ax.set_title(titulo + "\nmagenta pontilhado = ambígua (fora do MDT)")
    ax.set_aspect("equal")
    fig.savefig(caminho, dpi=80, bbox_inches="tight"); plt.close(fig)


# =====================================================================================================
# Método 2 (usado por padrão): contagem de curvas ao longo de raios entre curvas âncora.
# Entre duas curvas de valores diferentes (350 e 400) com m curvas no meio, cada uma vale
# z_a + k * (z_b - z_a) / (m + 1)   — não exige conhecer a equidistância de antemão.
# =====================================================================================================
def _raios_entre_ancoras(geoms, tree, valor, passo=100.0, max_pts=14, alcance=2500.0, max_entre=14, seed=0):
    """Para cada curva com valor, dispara raios perpendiculares e devolve evidências:
       mono:  (i_origem, [j1..jm], j_destino)  (valores origem != destino)
       cume:  idem com valores iguais (cume/vale)"""
    mono, cume = [], []
    for i in [k for k, v in valor.items()]:
        ln = geoms[i]
        n = int(min(max_pts, max(3, ln.length / passo)))
        for t in np.linspace(0.04, 0.96, n) * ln.length:
            p = ln.interpolate(t)
            a = ln.interpolate(max(0.0, t - 10)); b = ln.interpolate(min(ln.length, t + 10))
            d = np.array([b.x - a.x, b.y - a.y]); m = np.hypot(*d)
            if m == 0:
                continue
            nr = np.array([-d[1], d[0]]) / m
            for s in (1, -1):
                ray = LineString([(p.x + s * nr[0] * 3, p.y + s * nr[1] * 3),
                                  (p.x + s * nr[0] * alcance, p.y + s * nr[1] * alcance)])
                hits = []
                for j in tree.query(ray, predicate="intersects"):
                    if j == i:
                        continue
                    x = ray.intersection(geoms[j])
                    if x.is_empty:
                        continue
                    pt = x if x.geom_type == "Point" else (x.geoms[0] if hasattr(x, "geoms") else x.centroid)
                    hits.append((ray.project(pt), j))
                hits.sort()
                seq = []
                for _, j in hits:
                    if j in valor:
                        if len(seq) <= max_entre:
                            (mono if valor[j] != valor[i] else cume).append((i, tuple(seq), j))
                        break
                    seq.append(j)
                    if len(seq) > max_entre:
                        break
    return mono, cume


def estimar_delta(mono, valor, candidatos=None):
    """Moda de |dz|/(m+1) entre pares de curvas âncora. Retorna (delta, histograma)."""
    from collections import Counter
    c = Counter()
    for i, entre, j in mono:
        if len(entre) == 0:                  # mestras vizinhas sem curva no meio (área plana): não informa a equidistância
            continue
        step = abs(valor[j] - valor[i]) / (len(entre) + 1)
        c[round(step * 2) / 2] += 1          # resolução 0,5 m
    if not c:
        return None, c
    bruto = max(c.items(), key=lambda kv: kv[1])[0]
    # as curvas mestras são múltiplas de 50 m: a equidistância é 50/k (50, 25, 16,67, 12,5, 10, 8,33 ...)
    cand = [50.0 / k for k in range(1, 11)]
    return min(cand, key=lambda d: abs(d - bruto)), c


def _rotular_raios(geoms, amostras, ancoras, spots_xy, spots_z, delta=None, rodadas=8, verbose=False, tree=None):
    tree = tree or STRtree(geoms)
    n = len(geoms)
    valor = dict(ancoras)
    conf = np.zeros(n); conf_val = np.full(n, np.nan)
    for i, v in ancoras.items():
        conf[i] = 3; conf_val[i] = v
    D = delta
    for r in range(rodadas):
        mono, cume = _raios_entre_ancoras(geoms, tree, valor, seed=r)
        if D is None:
            D, hist = estimar_delta(mono, ancoras)
            if verbose:
                print("     histograma do passo |dz|/(m+1):", sorted(hist.items(), key=lambda kv: -kv[1])[:6], "-> D =", D)
        votos = {}
        for i, entre, j in mono:
            za, zb = valor[i], valor[j]
            m = len(entre)
            step = (zb - za) / (m + 1)
            if D and abs(abs(step) - D) > 0.35 * D:      # rota que não bate com a equidistância (lacuna/erro): ignora
                continue
            for k, lj in enumerate(entre, start=1):
                votos.setdefault(lj, []).append(za + np.sign(zb - za) * D * k)
        novos = 0
        for lj, vs in votos.items():
            if lj in valor:
                continue
            vs = np.array(vs)
            moda = pd.Series(np.round(vs / (D / 2)) * (D / 2)).mode().iloc[0]
            acordo = float(np.mean(np.abs(vs - moda) < 0.25 * D))
            if len(vs) >= 2 and acordo >= 0.7:
                valor[lj] = float(moda); conf[lj] = 2; conf_val[lj] = float(moda); novos += 1
        if verbose:
            print(f"     rodada {r}: {len(mono)} rotas monótonas, {len(cume)} de cume/vale -> +{novos} curvas")
        if novos == 0 and r > 0:
            break
    # cumes/vales e sobras: sinal pela superfície TIN do que já foi rotulado
    rot = np.array([i in valor for i in range(n)])
    z = np.array([valor.get(i, np.nan) for i in range(n)], float)
    if rot.sum() >= 3 and D:
        for passo in range(4):
            sup = _montar_sup(amostras, z, ~np.isnan(z), spots_xy, spots_z)
            mudou = False
            for i in np.where(np.isnan(z))[0]:
                m = float(np.median(sup(amostras[i])))
                z[i] = np.round(m / D) * D; conf[i] = 1; mudou = True
            if not mudou:
                break
            break
    return z, conf, D


def rotular_folha_raios(dgn, tol_ancora_m=60.0, delta=None, verbose=True, ocultar=0.0, seed=0):
    lin, tx = ler_hp(dgn)
    geoms = list(lin.geometry); tree = STRtree(geoms)
    nome = os.path.basename(dgn)
    anc, usados = {}, set()
    for k, r in tx.iterrows():
        if r.val % 50 != 0:
            continue
        p = Point(r.x, r.y); j = tree.nearest(p)
        if geoms[j].distance(p) <= tol_ancora_m:
            anc.setdefault(j, []).append(r.val); usados.add(k)
    anc = {j: float(pd.Series(v).mode().iloc[0]) for j, v in anc.items()}
    spots = tx.drop(index=list(usados))
    oculto = {}
    if ocultar > 0:
        rng = np.random.default_rng(seed)
        ks = list(anc)
        for j in rng.choice(ks, size=max(2, int(len(ks) * ocultar)), replace=False):
            oculto[j] = anc.pop(j)
    if verbose:
        print(f"  {nome}: {len(geoms)} curvas | {len(anc)} âncoras | {len(spots)} cotas de pontos")
    z, conf, D = _rotular_raios(geoms, _amostras(geoms), anc, spots[["x", "y"]].values, spots["val"].values,
                                delta=delta, verbose=verbose, tree=tree)
    out = gpd.GeoDataFrame({"cota": z, "confianca": conf.astype(int), "nivel": lin["nivel"].values}, geometry=geoms)
    return out, dict(delta=D, ocultas=oculto, ancoras=anc, spots=spots, nome=nome)


# =====================================================================================================
# Método 3 (padrão): 1) curvas mestras (grossas, níveis 1 e 56) valem múltiplos de 50 -> decididas pela superfície
#                    2) as finas recebem valor por contagem de curvas entre mestras (monótono) e por simetria
#                       (cume/vale), com a equidistância estimada da própria folha.
# =====================================================================================================
MESTRAS = {1: 0.0, 56: 50.0}      # nível -> resto da cota módulo 100


def _classe_mestras(geoms, nivel, amostras, ancoras, spots_xy, spots_z):
    """Atribui às curvas mestras o múltiplo de 100 (nível 1) ou 100k+50 (nível 56) mais próximo da superfície."""
    n = len(geoms)
    valor = dict(ancoras)
    sup = _montar_sup(amostras, np.array([valor.get(i, np.nan) for i in range(n)], float),
                      [i in valor for i in range(n)], spots_xy, spots_z)
    for i in range(n):
        if i in valor or nivel[i] not in MESTRAS:
            continue
        m = float(np.median(sup(amostras[i])))
        resto = MESTRAS[nivel[i]]
        k = np.round((m - resto) / 100.0)
        valor[i] = float(k * 100.0 + resto)
    return valor


def _rotular_m3(geoms, nivel, amostras, ancoras, spots_xy, spots_z, delta=None, verbose=False, tree=None, rodadas=6):
    tree = tree or STRtree(geoms)
    n = len(geoms)
    valor = _classe_mestras(geoms, nivel, amostras, ancoras, spots_xy, spots_z)
    conf = np.zeros(n)
    for i in valor:
        conf[i] = 3 if i in ancoras else 2.5
    base = dict(valor)
    mono, cume = _raios_entre_ancoras(geoms, tree, valor, seed=0)
    D = delta or estimar_delta(mono, valor)[0]
    if verbose:
        _, h = estimar_delta(mono, valor)
        print("     histograma do passo |dz|/(m+1):", sorted(h.items(), key=lambda kv: -kv[1])[:6], "-> D =", D)
    if not D:
        raise RuntimeError("não consegui estimar a equidistância")
    for r in range(rodadas):
        mono, cume = _raios_entre_ancoras(geoms, tree, valor, seed=r)
        votos = {}
        for i, entre, j in mono:
            za, zb = valor[i], valor[j]
            m = len(entre)
            if abs(abs((zb - za) / (m + 1)) - D) > 0.3 * D:
                continue
            for k, lj in enumerate(entre, start=1):
                votos.setdefault(lj, []).append((za + np.sign(zb - za) * D * k, 1.0))
        # cume/vale: simétrico; sinal pela superfície com o que já está rotulado
        z_atual = np.array([valor.get(i, np.nan) for i in range(n)], float)
        sup = _montar_sup(amostras, z_atual, ~np.isnan(z_atual), spots_xy, spots_z)
        for i, entre, j in cume:
            m = len(entre)
            if m == 0 or m > 8:
                continue
            za = valor[i]
            med = np.array([np.median(sup(amostras[l])) for l in entre])
            sinal = 1.0 if np.mean(med - za) >= 0 else -1.0
            for k, lj in enumerate(entre, start=1):
                votos.setdefault(lj, []).append((za + sinal * D * min(k, m + 1 - k), 0.5))
        novos = 0
        for lj, vs in votos.items():
            if lj in valor:
                continue
            v = np.array([a for a, _ in vs]); w = np.array([b for _, b in vs])
            grade = np.round(np.round(v / D) * D, 3)
            tab = pd.Series(w).groupby(grade).sum()
            moda = tab.idxmax()
            acordo = tab.max() / tab.sum()
            if tab.sum() >= 1.5 and acordo >= 0.65:
                valor[lj] = float(moda); conf[lj] = 2; novos += 1
        if verbose:
            print(f"     rodada {r}: {len(mono)} rotas monótonas, {len(cume)} cume/vale -> +{novos} curvas")
        if novos == 0:
            break
    z = np.array([valor.get(i, np.nan) for i in range(n)], float)
    ref = np.concatenate([np.asarray(spots_z, float), np.array(list(ancoras.values()), float)])
    if len(ref):
        lo, hi = ref.min() - 2 * D, ref.max() + 2 * D
        fora = (~np.isnan(z)) & ((z < lo) | (z > hi)) & (conf < 3)
        z[fora] = np.nan; conf[fora] = 0
    # sobras (isoladas): palpite pela superfície, confiança baixa
    if np.isnan(z).any() and (~np.isnan(z)).sum() >= 3:
        sup = _montar_sup(amostras, z, ~np.isnan(z), spots_xy, spots_z)
        for i in np.where(np.isnan(z))[0]:
            z[i] = np.round(float(np.median(sup(amostras[i]))) / D) * D; conf[i] = 1
    return z, conf, D


def rotular_folha_m3(dgn, tol_ancora_m=60.0, delta=None, verbose=True, ocultar=0.0, seed=0):
    lin, tx = ler_hp(dgn)
    return rotular_dados(lin, tx, os.path.basename(dgn), tol_ancora_m, delta, verbose, ocultar, seed)


def rotular_dados(lin, tx, nome="folha", tol_ancora_m=60.0, delta=None, verbose=True, ocultar=0.0, seed=0):
    """lin: GeoDataFrame (metros) com coluna 'nivel'; tx: DataFrame x,y,val (textos numéricos)."""
    geoms = list(lin.geometry); tree = STRtree(geoms); nivel = lin["nivel"].values
    anc, usados = {}, set()
    for k, r in tx.iterrows():
        if r.val % 50 != 0:
            continue
        p = Point(r.x, r.y); j = tree.nearest(p)
        if geoms[j].distance(p) <= tol_ancora_m:
            anc.setdefault(j, []).append(r.val); usados.add(k)
    anc = {j: float(pd.Series(v).mode().iloc[0]) for j, v in anc.items()}
    spots = tx.drop(index=list(usados)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(spots)); nv = max(3, len(spots) // 4) if len(spots) >= 12 else 0
    va, tr = idx[:nv], idx[nv:]
    oculto = {}
    if ocultar > 0:
        ks = [j for j in anc if nivel[j] in MESTRAS]
        for j in rng.choice(ks, size=max(2, int(len(ks) * ocultar)), replace=False):
            oculto[j] = anc.pop(j)
    if verbose:
        print(f"  {nome}: {len(geoms)} curvas ({int(np.isin(nivel, list(MESTRAS)).sum())} mestras) | {len(anc)} âncoras | "
              f"{len(spots)} cotas de pontos")
    am = _amostras(geoms)
    z, conf, D = _rotular_m3(geoms, nivel, am, anc, spots.iloc[tr][["x", "y"]].values, spots.iloc[tr]["val"].values,
                             delta=delta, verbose=verbose, tree=tree)
    out = gpd.GeoDataFrame({"cota": z, "confianca": conf, "nivel": nivel}, geometry=geoms)
    # validação independente: cotas de pontos reservadas, só dentro da faixa de cotas das curvas
    val = {}
    if nv:
        ok = conf >= 2
        sup = _montar_sup(am, z, ok, [], [])
        xy = spots.iloc[va][["x", "y"]].values; zz = spots.iloc[va]["val"].values
        faixa = (zz >= np.nanmin(z[ok]) - 5) & (zz <= np.nanmax(z[ok]) + 5)
        e = sup(xy)[faixa] - zz[faixa]
        val = dict(n=int(faixa.sum()), rmse=float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
                   mae=float(np.mean(np.abs(e))) if len(e) else np.nan)
    info = dict(nome=nome, delta=D, ancoras=anc, ocultas=oculto, validacao=val,
                pct_firmes=float(100 * np.mean(conf >= 2)))
    return out, info
