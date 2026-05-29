# Mask -> Bounding Box converters

Questa cartella contiene script per convertire le maschere generate dagli
script di inferenza in bounding box, rispettando la struttura di ogni dataset.

Script disponibili:

- `egoexo_masks_to_boxes.py` (esistente) : conversione per EgoExo4D
- `egotracks_masks_to_boxes.py` (esistente) : conversione per EgoTracks
- `it3dego_masks_to_boxes.py` : conversione per IT3DEgo (aggiunto)
- `trek150_masks_to_boxes.py` : conversione per TREK-150 (aggiunto)

Esempi d'uso:

Convertire IT3DEgo masks:

```bash
python it3dego_masks_to_boxes.py --input_dir /path/to/it3dego_masks \
    --output_dir /path/to/converted_annotations
```

Convertire TREK-150 masks:

```bash
python trek150_masks_to_boxes.py --input_dir /path/to/trek150_masks \
    --output_dir /path/to/converted_annotations
```

Per `EgoExo4D` e `EgoTracks` usare gli script già presenti nella cartella.

Nota:
- Gli script scrivono file testuali con formati compatibili agli loader
  dei rispettivi dataset (es. `2d_bbox_annot/*.txt`, `frames.txt` + `boxes.txt`).
- I valori delle bounding box sono in formato `x y w h` con una cifra decimale.
