import os, numpy as np, rasterio, yaml
from rasterio.transform import from_origin
from mdt_pre81.pipeline import executar
from conftest import X0, Y0


def _foto(raiz, nome, tr, w, h, crs=None, tfw=False):
    rng = np.random.default_rng(3)
    a = rng.normal(150, 12, (h, w)).clip(20, 255)
    xs = tr.c + (np.arange(w) + .5) * tr.a
    a[np.abs(xs[None, :] - (X0 + 8000 + 400 * np.sin(np.arange(h)[:, None] / 30))) < 100] = 25
    arq = raiz / "data" / nome
    kw = dict(driver="GTiff", height=h, width=w, count=1, dtype="uint8")
    if crs:
        kw.update(crs=crs, transform=tr)
    with rasterio.open(arq, "w", **kw) as d:
        d.write(a.astype("uint8"), 1)
    if tfw:  # TIFF + world file, sem CRS
        open(str(arq)[:-4] + ".tfw", "w").write(f"{tr.a}\n0\n0\n{tr.e}\n{tr.c + tr.a/2}\n{tr.f + tr.e/2}\n")


def test_imagens_historicas(projeto):
    _foto(projeto, "h73.tif", from_origin(X0, Y0 + 18000, 30, 30), 600, 600, crs="EPSG:31982")
    _foto(projeto, "h67.tif", from_origin(X0, Y0 + 18000, 30, 30), 600, 600, tfw=True)   # sem CRS
    cfg = yaml.safe_load(open(projeto / "config/projeto.yml", encoding="utf-8"))
    cfg["historicas"] = [dict(id="1973", rotulo="1973", arquivo="data/h73.tif", cor="#ffd60a", tracar=True),
                         dict(id="1967", rotulo="1967", arquivo="data/h67.tif", cor="#ff7b00", tracar=True)]
    yaml.safe_dump(cfg, open(projeto / "config/projeto.yml", "w", encoding="utf-8"), allow_unicode=True)
    executar(["aoi", "historicas", "webmap"], str(projeto / "config/projeto.yml"))
    for y in ("1973", "1967"):
        assert (projeto / f"outputs/hist_{y}/drenagem_{y}.gpkg").exists()
    html = (projeto / "outputs/webmap_standalone.html").read_text(encoding="utf-8")
    for t in ("Drenagem 1967", "Drenagem 1973", "Imagem 1967"):
        assert t in html


def test_gj_poligono_invalido():
    import geopandas as gpd
    from shapely.geometry import Polygon
    from mdt_pre81.webmap import _gj
    gravata = Polygon([(500000, 7600000), (500100, 7600100), (500100, 7600000), (500000, 7600100)])
    r = _gj(gpd.GeoDataFrame({"a": [1]}, geometry=[gravata], crs="EPSG:31982"), 20)
    assert r and len(r["features"]) == 1


def test_guia_hidrografia_ibge_e_camadas(projeto):
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon
    rio = LineString([(X0 + 8000 + 400 * np.sin(i / 30), Y0 + 18000 - i * 30) for i in range(0, 600, 5)])
    gpd.GeoDataFrame({"nivel": [2, 7], "folha": ["t", "t"]}, geometry=[rio, rio.parallel_offset(2500, "left")],
                     crs="EPSG:31982").to_file(projeto / "data/hidrografia_ibge.gpkg", layer="linhas", driver="GPKG")
    gpd.GeoDataFrame({"nivel": [2], "folha": ["t"]}, geometry=[rio.buffer(60)], crs="EPSG:31982").to_file(
        projeto / "data/hidrografia_ibge.gpkg", layer="poligonos", driver="GPKG")
    cfg = yaml.safe_load(open(projeto / "config/projeto.yml", encoding="utf-8"))
    cfg["historicas"] = [dict(id="1973", rotulo="1973", arquivo="data/h73.tif", cor="#ffd60a", tracar=True)]
    cfg["entradas"]["hidro_ibge"] = "data/hidrografia_ibge.gpkg"
    yaml.safe_dump(cfg, open(projeto / "config/projeto.yml", "w", encoding="utf-8"), allow_unicode=True)
    executar(["historicas", "webmap"], str(projeto / "config/projeto.yml"))
    html = (projeto / "outputs/webmap_standalone.html").read_text(encoding="utf-8")
    assert "Hidrografia das cartas IBGE" in html


def test_so_imagem_sem_drenagem(projeto):
    cfg = yaml.safe_load(open(projeto / "config/projeto.yml", encoding="utf-8"))
    cfg["historicas"] = [dict(id="1973", rotulo="1973", arquivo="data/h73.tif")]
    yaml.safe_dump(cfg, open(projeto / "config/projeto.yml", "w", encoding="utf-8"), allow_unicode=True)
    executar(["historicas", "webmap"], str(projeto / "config/projeto.yml"))
    html = (projeto / "outputs/webmap_standalone.html").read_text(encoding="utf-8")
    assert "Imagem 1973" in html and "Drenagem 1973" not in html and "Drenagem 1979" not in html
