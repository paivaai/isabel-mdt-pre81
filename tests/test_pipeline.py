import os, json, re
import numpy as np, rasterio
from mdt_pre81.pipeline import executar
from mdt_pre81 import core


def test_pipeline_completo(projeto):
    cfg = str(projeto / "config/projeto.yml")
    executar(["all"], cfg)
    out = projeto / "outputs"
    for f in ["aoi.gpkg", "landsat_1979_melhorado.tif", "drenagem_1979.gpkg", "mdt_pre81.tif", "webmap_standalone.html"]:
        assert (out / f).exists(), f
    with rasterio.open(out / "mdt_pre81.tif") as s:
        dem = s.read(1); assert s.crs.to_epsg() == 31982
    ok = dem != -9999.0
    assert ok.mean() > 0.5
    assert 380 <= dem[ok].min() and dem[ok].max() <= 700
    assert (projeto / "site/index.html").exists() and (projeto / "site/assets/mdt_pre81.png").exists()


def test_mdt_sem_curvas_nao_inventa(projeto, capsys):
    """Sem curvas o MDT não é gerado (nada de Topodata/SRTM)."""
    import yaml
    cfg = yaml.safe_load(open(projeto / "config/projeto.yml", encoding="utf-8"))
    cfg["entradas"]["curvas"] = "data/nao_existe.gpkg"
    cfg["saidas"]["pasta"] = "outputs_vazio"
    yaml.safe_dump(cfg, open(projeto / "config/sem_curvas.yml", "w", encoding="utf-8"))
    executar(["mdt"], str(projeto / "config/sem_curvas.yml"))
    assert "NÃO gerado" in capsys.readouterr().out
    assert not (projeto / "outputs_vazio/mdt_pre81.tif").exists()


def test_webmap_site_dados(projeto):
    executar(["webmap"], str(projeto / "config/projeto.yml"))
    html = (projeto / "site/index.html").read_text(encoding="utf-8")
    assert "World_Imagery" in html and "assets/dados.js" in html
    js = (projeto / "site/assets/dados.js").read_text(encoding="utf-8")
    d = json.loads(js[len("window.DADOS="):-1])
    ids = [c["id"] for c in d["camadas"]]
    assert {"landsat", "mdt", "natd", "aoi"} <= set(ids) and "drenagem" not in ids
    assert d["grade"]["w"] > 10
    for c in d["camadas"]:
        if c["tipo"] == "raster":
            assert (projeto / "site" / c["url"]).exists()
    natd = [c for c in d["camadas"] if c["id"] == "natd"][0]
    assert natd["campo_classe"] == "CLASSE" and len(natd["classes"]) == 2


def test_standalone_autocontido(projeto):
    html = (projeto / "outputs/webmap_standalone.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html and "assets/" not in html.replace("assets/dados.js", "")


def test_codigo_nao_le_topodata_srtm():
    import ast
    codigo = open(core.__file__, encoding="utf-8").read()
    arvore = ast.parse(codigo)
    textos = [n.value.lower() for n in ast.walk(arvore) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any("topodata" in t and "http" in t for t in textos)
