# data/

Entradas do projeto (NÃO versionadas; ficam no `.gitignore`):

| arquivo | o que é |
|---|---|
| `landsat_1979.tif` | cena Landsat 2 MSS (ago/1979) em GeoTIFF |
| `NA_TD_TOTAL.zip` | shapefile das áreas de referência (pode ser .shp, .zip ou .rar) |
| `curvas_cartas.gpkg` | (opcional) curvas das cartas antigas com campo numérico `cota`; se faltar, o pipeline tenta baixar do IBGE |

Para dados pequenos que possam ir ao repositório, use `git add -f`.
