
# Spectra Classifier — Binary/4 class classifier

**Defaults to binary** (classes 1 & 2):
- AGN → class 1 (internal 0)
- HM-STAR / LM-STAR → class 2 (internal 1)

Switch to **all 4 classes** with `--classes all4`:
- YSO / CV → class 3 (internal 2)
- NS / HMXB / LMXB / NS_bin → class 4 (internal 3)

Files:
- `Allsrc_classes.txt`: `<SRC_ID> <LABEL>` with blanks allowed
- `PN_spectra_rebinned.txt`: `(N_all, 380)` numeric, aligned by row to the labels file

## Run
```bash
pip install -r requirements.txt

python main.py \
  --labels data/Allsrc_classes.txt \
  --spectra data/PN_spectra_rebinned.txt \
  --classes 2        # default; for all 4 use: --classes 4

```
