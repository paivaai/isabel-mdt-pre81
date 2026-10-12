import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import LineString
from mdt_pre81 import rotular


def _sintetico(D=10.0):
    """Morro suave com curvas a cada D m; mestras (nível 1 = 400, 56 = 350/450) e finas (2/55) alternadas."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = np.meshgrid(np.linspace(0, 12000, 400), np.linspace(0, 12000, 400))
    z = 330 + 130 * np.exp(-(((x - 4500) / 3500) ** 2 + ((y - 6000) / 4500) ** 2)) \
        + 60 * np.exp(-(((x - 9000) / 2000) ** 2 + ((y - 3000) / 2500) ** 2))
    niveis = np.arange(340, 470, D)
    cs = plt.contour(x, y, z, levels=niveis)
    linhas, nivel, cota = [], [], []
    for v, segs in zip(cs.levels, cs.allsegs):
        for sg in segs:
            if len(sg) < 5:
                continue
            if v % 100 == 0: nv = 1
            elif v % 50 == 0: nv = 56
            else: nv = 2 if int(round((v - 340) / D)) % 2 == 0 else 55
            linhas.append(LineString(sg)); nivel.append(nv); cota.append(float(v))
    plt.close("all")
    lin = gpd.GeoDataFrame({"nivel": nivel}, geometry=linhas)
    tx = []
    for g, v, nv in zip(linhas, cota, nivel):      # texto de âncora colado em cada mestra
        if nv in (1, 56):
            p = g.interpolate(0.4 * g.length); tx.append((p.x + 5, p.y + 5, v))
    # algumas cotas de pontos
    for (px, py) in [(4500, 6000), (9000, 3000), (2000, 2500)]:
        zz = 330 + 130 * np.exp(-(((px - 4500) / 3500) ** 2 + ((py - 6000) / 4500) ** 2)) \
            + 60 * np.exp(-(((px - 9000) / 2000) ** 2 + ((py - 3000) / 2500) ** 2))
        tx.append((px, py, round(float(zz))))
    return lin, pd.DataFrame(tx, columns=["x", "y", "val"]), np.array(cota)


def test_rotula_curvas_sinteticas():
    lin, tx, cota = _sintetico(10.0)
    g, info = rotular.rotular_dados(lin, tx, "sintetico", verbose=False)
    assert abs(info["delta"] - 10.0) < 1e-6
    firmes = g.confianca.values >= 2
    assert firmes.mean() > 0.6
    erro = np.abs(g.cota.values[firmes] - cota[firmes])
    assert np.mean(erro == 0) > 0.9          # >90% das curvas firmes com cota exata
    assert erro.max() <= 20
