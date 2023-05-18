# Experimenty so Superpixelmi a Watershed algoritmom

**Autor: Bc. Ivan Vykopal**

Tento adresár obsahuje epxerimenty do diplomovej práce a to predovšetkým vizualizáciu do finálnej verzie práce a zároveň adresár obsahuje script pre iterovanú dilatáciu a zhlukovanie.

Obsah:

- `args.py` - Súbor pre argumenty pre finálny script pre iterovanú dilatáciu a zhlukovanie.
- `script.py` - Script pre iterovanú dialtáciu a zhlukovanie.
- `Superpixels.ipynb` - Notebook obsahujúci experiment so superpixelmi na snímku z IKEM
- `Watershed.ipynb` - Notebook obsahujúci experiment s Watershed algortimom

Príklad ako spustiť script pre iterovanú dilatáciu a zhlukovanie:

```bash
python script.py -i "D:\Master Thesis\Data\EMB-IKEM-2022-03-09\1230_21_HE.vsi" -g "D:\Master Thesis\Data\EMB-IKEM-2022-03-09\QuPath project EMB - anotations\annotations\1230_21_HE.vsi - 20x.geojson" -o "C:\Users\ivanv\Desktop\Final tests" -a dbscan --only_immune True
```

