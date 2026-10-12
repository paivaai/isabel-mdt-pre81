import os, numpy as np, pytest, yaml
import geopandas as gpd, rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, LineString
from pyproj import Transformer

X0, Y0 = Transformer.from_crs("EPSG:4326", "EPSG:31982", always_xy=True).transform(-50.15, -21.07)
X0, Y0 = round(X0, -3), round(Y0, -3)


def superficie(x, y):
    """Vale em V descendo para o norte, com um rio no meio."""
    return 380 + 0.012 * (y - Y0) + 0.00004 * (x - X0 - 8000) ** 2 / 100 * 0 + 60 * np.abs(x - X0 - 8000) / 8000


@pytest.fixture(scope="session")
def projeto(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("proj")
    (raiz / "config").mkdir(); (raiz / "data").mkdir()
    # Landsat sintético 60 m, 3 bandas, rio escuro
    w = h = 300
    tr = from_origin(X0, Y0 + 18000, 60, 60)
    xs = X0 + (np.arange(w) + .5) * 60
    rng = np.random.default_rng(1)
    base = rng.normal(90, 6, (3, h, w)).clip(5, 255)
    rio = np.abs(xs[None, :] - (X0 + 8000 + 600 * np.sin(np.arange(h)[:, None] / 25))) < 120
    base[:, rio] = 12
    with rasterio.open(raiz / "data/landsat_1979.tif", "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                       crs="EPSG:31982", transform=tr, nodata=0) as d:
        d.write(base.astype("uint8").clip(1, 255))
    # curvas de nível de 20 m a partir da superfície
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gx, gy = np.meshgrid(np.linspace(X0 - 2000, X0 + 20000, 300), np.linspace(Y0 - 2000, Y0 + 20000, 300))
    cs = plt.contour(gx, gy, superficie(gx, gy), levels=np.arange(380, 700, 20))
    linhas, cotas = [], []
    for nivel, segs in zip(cs.levels, cs.allsegs):
        for s in segs:
            if len(s) > 3:
                linhas.append(LineString(s)); cotas.append(float(nivel))
    plt.close("all")
    gpd.GeoDataFrame({"cota": cotas}, geometry=linhas, crs="EPSG:31982").to_file(raiz / "data/curvas_cartas.gpkg", driver="GPKG")
    # NA_TD sintético
    nat = gpd.GeoDataFrame({"CLASSE": ["FAIXA DE DESAPROPRIACAO", "ÁREA ALAGADA"], "ID": [1, 2]},
                           geometry=[box(X0 + 7000, Y0 + 3000, X0 + 9000, Y0 + 9000),
                                     box(X0 + 7500, Y0 + 9000, X0 + 8500, Y0 + 12000)], crs="EPSG:31982")
    (raiz / "data/natd").mkdir()
    nat.to_file(raiz / "data/natd/NA_TD.shp")
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "projeto.yml"), encoding="utf-8"))
    cfg["entradas"]["natd"] = "data/natd/NA_TD.shp"
    cfg["area"]["baixar_limites_ibge"] = False
    cfg["area"]["fallback_km"] = 6
    cfg["curvas"]["crs_origem"] = "EPSG:31982"
    cfg["curvas"]["baixar_ibge"] = False
    cfg["curvas"]["resolucao_m"] = 60
    cfg["landsat"]["upscale"] = 1
    cfg["webmap"]["pasta_site"] = "site"
    yaml.safe_dump(cfg, open(raiz / "config/projeto.yml", "w", encoding="utf-8"), allow_unicode=True)
    return raiz
