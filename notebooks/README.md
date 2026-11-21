# Colab notebooks

Runtime for both notebooks: **T4 GPU** (Runtime → Change runtime type), Python 3. Run all cells from top to bottom.

## Drive layout

```text
MyDrive/Colab Notebooks/CCTV/
├── datasets/
│   ├── skyfinder/                      # <camera_id>/<file>.jpg (an extra amos/ level is detected)
│   └── cct20/
│       ├── eccv_18_all_images_sm/      # only used with IMAGE_SOURCE='drive'
│       ├── eccv_18_annotation_files/   # train, cis_val, cis_test, trans_val, trans_test JSON (downloaded from LILA if missing)
│       └── gtloc_f12/                  # earlier burst outputs; read-only
└── results/                            # each run writes a NEW <experiment>_<YYYYMMDD_HHMM>/ folder
```

The notebooks never overwrite existing files. They create each results folder with `exist_ok=False`.

## Run order

1. `GTLOC_PHASE1_COLAB.ipynb`: GT-Loc time-only reproduction on SkyFinder, using frozen CLIP ViT-L/14 with the GT-Loc projection, the time encoder, the `cvt.npy` gallery (16,656 rows) and TenCrop.
   - Paper: ToY 1.40 months, ToD 2.72 h, TPS 77.00.
   - Earlier fp32 reproduction: 1.49, 2.67 and 76.40.
   - Output: `results/gtloc_phase1_<stamp>/` with `skyfinder_predictions.csv`, `skyfinder_reproduction_results.json` and `environment.txt`.
2. `CCT20_PHASE2_EXPERIMENTS.ipynb`: experiments on the canonical CCT20 1 s triplets (frames 1-2-3, both gaps 1.0 s; expected 15,976 triplets / 47,928 frames / 18 locations), in the order A, C, B. It uses the same model, checkpoint, gallery and preprocessing as Phase 1. `scripts/cct20_phase2_experiments.py` contains the same code.
   - **A**, false conflict rate: share of authentic 1 s pairs with predicted dH > 0.5 h and > 1 h, overall, per transition and per camera.
   - **C**, latent cosine vs gallery hop: d_cos = 1 − cos(z1, z2) on the 512-d z. Pairs are classed as gallery hop, feature shift, other or no jump.
   - **B**, empty vs animal: median and p90/p95/p99 of d_burst = sqrt((dH/12)² + (dM/6)²), by camera and by day vs night IR.
   - Output: `results/cct20_phase2_<stamp>/` with `triplets.csv`, `frames.csv`, `frame_cache.npz`, `pairs.csv`, `burst_summary.csv`, `A_*.csv`, `C_*.csv/png`, `B_*.csv/png` and `summary.json`. The run ends by printing `RUN FINISHED`.
   - On a T4, the first run takes about 1–1.5 h. A rerun with `PREV_CACHE` set to an earlier results folder takes minutes.
   - **Images:** with `IMAGE_SOURCE = 'lila'` (the default), cell 3b downloads the official archive to `/content/cct20_lila/` with `wget -c`, or `aria2c` if installed, and checks its size and MD5 before extracting it. The archive is [eccv_18_all_images_sm.tar.gz](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz), 6,492,615,601 bytes. The download and extraction take about 5–15 min and are skipped if the images were already extracted in the same Colab session. `IMAGE_SOURCE = 'drive'` reads the Drive folder instead: it lists the folder once, with retries, then copies the needed frames to local disk. Do not rely on per-file checks on that folder, because Drive FUSE times out on 57k files.
   - **Missing images:** if more than 1% (`MAX_MISSING_FRAC`) of the triplet frames have no image file, cell 4 stops with the counts and example names, before any inference.
   - **Labels:** an image is `empty` only if *all* its annotations, combined across the five split JSONs, are `empty`. It is `animal` if any annotation is an animal, and `unlabeled` if it has none. The official JSONs give 53,850 animal and 4,014 empty images. In the eq3 triplets, 14,958 are animal and 1,018 are empty. The cell prints this rule and the MD5 of each annotation file.

Offline tests (no GPU or Drive needed): `python tests/test_cct20_phase2.py`. Set `CCT20_ANN_DIR` to the extracted `eccv_18_annotation_files/` to also run the check against the real annotation files.
After editing `scripts/cct20_phase2_experiments.py`, regenerate the notebook with `python scripts/build_phase2_notebook.py`.

## Caveats

- **The gallery has no night stamps.** `cvt.npy` only covers 04:57–22:43, so night IR frames cannot be predicted at their true hour and snap to dawn or dusk stamps. Keep this in mind when reading FCR and d_burst for IR frames.
- **The ~0.526 h gallery step is an average, not a grid.** It is 8,766 h / 16,656 stamps. The stamps sit on a month × hour plane, and the real median nearest-neighbour spacing is about 0.03 h. Experiment C therefore also reports hops at any dH, and jumps that land on the previous frame's second-best stamp.
