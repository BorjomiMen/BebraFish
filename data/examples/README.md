# Примеры NAOMi

6 полных пар 400x485 из исходного экспорта ZEBRAFISH.

Первая пара:

- `naomi/images/naomi_0000_seed20263000.tif`
- соответствующий файл в `naomi/masks`

Имена, пути и seed-группы перечислены в `naomi/pairs.csv`.
Для импорта: `python pipeline.py prepare-unet --source data/examples/naomi --output data/prepared_examples`.
