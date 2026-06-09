# datasets/

Data payloads live here but are **never committed** (see `.gitignore`). This
folder holds only download/prep instructions.

## Cityscapes (primary driving benchmark)

1. Register at https://www.cityscapes-dataset.com/ and download:
   - `leftImg8bit_trainvaltest.zip` (images)
   - `gtFine_trainvaltest.zip` (fine annotations, 19 eval classes)
2. Unzip into `datasets/cityscapes/` so the tree looks like:
   ```
   datasets/cityscapes/
     leftImg8bit/{train,val,test}/<city>/*.png
     gtFine/{train,val,test}/<city>/*.png
   ```
3. The loader in `src/fiedler/data/cityscapes.py` handles the 34->19 label remap.

For quick local smoke tests, a tiny synthetic driving-like set is produced by
`src/fiedler/data/toy.py` — no download required.
