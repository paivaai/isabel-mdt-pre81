# Projeto Isabel — MDT pré-1981 (Buritama / Zacarias, SP)

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paivaai/isabel-mdt-pre81/blob/main/notebooks/colab_executar.ipynb)

Pipeline reprodutível que, a partir de **cartas topográficas antigas** (curvas de nível) e de uma **imagem Landsat 2 MSS de 1979**, gera:

| saída | arquivo |
|---|---|
| Landsat 1979 melhorado (GeoTIFF) | `outputs/landsat_1979_melhorado.tif` |
| Drenagem e água 1979 (GPKG + shapefile) | `outputs/drenagem_1979.gpkg`, `outputs/drenagem_1979_shp.zip` |
| **MDT pré-1981** (SIRGAS 2000 / UTM 22S, 30 m) | `outputs/mdt_pre81.tif` |
| **Webmap** sobre imagem de satélite | `docs/index.html` (GitHub Pages) e `outputs/webmap_standalone.html` |

> **Regra do projeto:** o MDT usa **somente** curvas de cartas anteriores a 1981. **Nada** de Topodata/SRTM (2000) é lido ou misturado. Onde não há curva a ≤ 1500 m, o MDT fica **sem dados** (não se inventa relevo).
> O `NA_TD` é só **referência** (áreas ao longo do rio alagado); a região de estudo é Buritama + Zacarias.

## Webmap
Base **Esri World Imagery** (+ Topográfico Esri e OSM), camadas com transparência: Landsat 1979, MDT (cores + sombreado, legenda em m), curvas, drenagem, água, NA_TD por classe (popup com atributos) e região de estudo. Clique no mapa = **altitude do MDT**; régua, área, pin, tela cheia, "desmarcar todos" e **visão 3D** do MDT.

Publicação: *Settings → Pages → Source: GitHub Actions*. (Em repositório privado o Pages exige plano pago; sem isso use `outputs/webmap_standalone.html`, um arquivo único que abre em qualquer navegador.)

## Como rodar
**Colab (sem instalar nada):** clique no badge acima, suba `landsat_1979.tif` e `NA_TD_TOTAL.zip`, execute.

**Local:**
```bash
pip install -e ".[dev]"
# coloque os dados em data/ (veja data/README.md) e ajuste config/projeto.yml
python -m mdt_pre81 all          # ou: aoi | landsat | drenagem | curvas | mdt | webmap
pytest
```

**GitHub Actions** (o runner tem internet aberta):
- `Baixar e inspecionar curvas do IBGE` — baixa as cartas vetoriais 1:50.000 e gera o relatório (têm altitude? qual datum?) + `curvas_cartas.gpkg` como artefato.
- `Processar tudo e atualizar o webmap` — pega os dados do Release `dados`, roda `all` e commita `docs/`.
- `Publicar webmap` — publica `docs/` no GitHub Pages a cada push.

## Estrutura
```
config/projeto.yml     parâmetros (datum das curvas, resolução, distância máxima, wms...)
src/mdt_pre81/         core.py (AOI, Landsat, drenagem, MDT) · ibge.py · webmap.py · pipeline.py · cli.py
tests/                 dados sintéticos + pytest (+ smoke_browser.py opcional)
notebooks/             launcher do Colab
docs/                  site do webmap (GitHub Pages)
data/  outputs/        entradas e saídas (fora do git)
```

## Datum
Curvas das cartas antigas: Córrego Alegre (`EPSG:22522`) ou SAD69 (`EPSG:29192`); ajuste `curvas.crs_origem`. Saída em SIRGAS 2000 UTM 22S (`EPSG:31982`), vertical = Imbituba.
