# Finálny cyklus pre identifikáciu vyšších morfologických štruktúr

**Autor: Bc. Ivan Vykopal**

Obsah:

- `configs` - Adresár obsahujúci konfiguračné súbory pre natrénované modely.
- `models` - Adresár obsahujúci zdrojové kódy k modelom.
- `utils` -  Adresár obsahujúci podporné funkcie pre konverziu výstupu na GeoJSON, post-processing a iné potrebné funkcie.
- `final_cycle_run.py` - script pre spúšťanie inferencie modelov.
- `final_cycle.py` - script pre inferenciu modelov.
- `Final_cycle.ipynb` - Notebook obsahujúci funkcionalitu pre spustenie predikcie nad celým adresárom

Príklad príkazu pre spustenie scriptu nad jedným snímkom:

```bash
python final_cycle_run.py --model_config_path "deeplabv3plus1_final.yaml" "deeplabv3plus2_final.yaml" "deeplabv3plus3_final.yaml" --image_path "D:\Master Thesis\Data\EMB-IKEM-2022-03-09\1225_21_HE.vsi" --cell_path "D:\Master Thesis\Data\Cell Annotations\1225_21_HE.vsi - 20x.geojson" --output_path "D:/Test" --metadata_path "D:/Test\temp" --overlap True
```
