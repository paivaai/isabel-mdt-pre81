import numpy as np, rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, LineString
from conftest import X0, Y0
from mdt_pre81.tracado import extrair_tracado


def _imagem(arq, n=700, px=10):
    rng = np.random.default_rng(5)
    a = rng.normal(150, 14, (n, n)); a = np.clip(a, 40, 255)
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(a, 1.2) + 40 * (gaussian_filter(rng.normal(0, 1, (n, n)), 12) * 8)
    yy, xx = np.mgrid[0:n, 0:n]
    # rio sinuoso (50 m) + afluente (30 m)
    cx = 300 + 60 * np.sin(yy / 70.0)
    a[np.abs(xx - cx) < 2.5] = 45
    ty = 250 + (xx - 300) * 0.35
    afl = (np.abs(yy - ty) < 1.5) & (xx > 300) & (xx < 560)
    a[afl] = 55
    # talhões escuros grossos (500 m) e rede de linhas finas (malha)
    a[80:140, 480:560] = 40; a[500:560, 80:160] = 40
    for k in range(560, 700, 14):
        a[k:k + 2, 420:520] = 60; a[400:520, k - 140 + 560 - 560 + 420:k - 140 + 560 - 560 + 422] = 60
    # linha fina isolada, longe do rio
    a[620, 40:300] = 60
    with rasterio.open(arq, "w", driver="GTiff", height=n, width=n, count=1, dtype="uint8", crs="EPSG:31982",
                       transform=from_origin(X0, Y0 + 7000, px, px)) as d:
        d.write(np.clip(a, 1, 255).astype("uint8"), 1)
    return cx, px


def test_tracado_acha_rio_e_ignora_talhao_e_malha(tmp_path):
    arq = str(tmp_path / "f.tif")
    cx, px = _imagem(arq)
    semente = box(X0 + 250 * px, Y0 + 7000 - 60 * px, X0 + 350 * px, Y0 + 7000)   # topo do rio
    gl, _ = extrair_tracado(arq, str(tmp_path / "o"), "t", semente=semente, comp_min_m=800)
    assert len(gl) and gl["comp_m"].sum() > 5000
    # tudo que foi traçado deve estar perto do rio ou do afluente (não nos talhões/malha/linha isolada)
    rio = LineString([(X0 + c * px, Y0 + 7000 - r * px) for r, c in zip(range(700), 300 + 60 * np.sin(np.arange(700) / 70.0))])
    afl = LineString([(X0 + c * px, Y0 + 7000 - (250 + (c - 300) * 0.35) * px) for c in range(300, 560)])
    from shapely.ops import unary_union
    alvo = unary_union([rio, afl]).buffer(80)
    dentro = sum(l.intersection(alvo).length for l in gl.geometry) / gl.length.sum()
    assert dentro > 0.9, dentro
    assert gl.length.sum() > 0.6 * (rio.length + afl.length)   # e achou a maior parte
