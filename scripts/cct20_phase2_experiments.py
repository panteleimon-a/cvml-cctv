# %% [markdown]
# # GT-Loc Phase 2: CCT20 1-second burst experiments (A, C, B)
#
# Built on `notebooks/GTLOC_PHASE1_COLAB.ipynb` and uses the same pieces: the official GT-Loc repo pinned at
# `6e7d9d2745d15a101b3b4dd05151807af77831ea`, the released checkpoint (Drive id `170wGyCYLSF3DWOvITIAbtN1OkmDlS4v6`),
# frozen CLIP ViT-L/14 with the GT-Loc projection, the GT-Loc time encoder (no GPS branch), the `cvt.npy` time gallery
# (16,656 rows), the official TenCrop eval transform and exact cosine retrieval with similarities averaged over crops.
#
# **Runtime:** *Runtime → Change runtime type → T4 GPU* (Python 3). Do not use TPU.
#
# **Run order:** run every cell from top to bottom (*Runtime → Run all*).
# 1. Config. Creates a new results folder `BASE/results/cct20_phase2_<YYYYMMDD_HHMM>/` and never overwrites anything.
# 2. Setup: installs packages, mounts Drive and checks the GPU.
# 3. Pure functions: circular distances, triplet filter, label rule, missing-frame guard, classifiers.
# 3b. Data source. With `IMAGE_SOURCE='lila'` (the default), downloads the official CCT20 small-images archive
#     (`eccv_18_all_images_sm.tar.gz`, 6,492,615,601 bytes) from LILA's public bucket to `/content` and extracts it
#     locally. It is skipped if already extracted in this session. The annotations come from Drive, or from LILA if
#     the Drive JSONs are missing. `IMAGE_SOURCE='drive'` lists the Drive folder once, with retries.
# 4. CCT20 annotations → canonical 1 s triplets. Checks against 15,976 triplets / 47,928 frames / 18 locations.
#    Stops before inference if more than 1% of the triplet frames have no image file.
# 5. GT-Loc encoders + gallery: the same loading as Phase 1, plus a gallery-resolution report.
# 6. Read-only inspection of `gtloc_f12/preds.npz` and `transition_distances.csv`.
# 7. Per-frame inference and cache: 512-d z, top-2 retrieval, predicted time, IR score. It resumes from shards.
#    It can also start from `PREV_CACHE`.
# 8. Pair table: f1→f2 and f2→f3 with dH, dM, d_burst and d_cos. Checks that the f1→f2 burst median is about 0.04.
# 9. **A**: false conflict rate. 10. **C**: latent cosine vs gallery hop. 11. **B**: empty vs animal bursts.
# 12. Final summary, ending with `RUN FINISHED`.
#
# **Expected time on a T4:** the LILA download (6.5 GB) and extraction take about 5–15 min. TenCrop fp16 inference on 47,928
# frames takes about 30–50 min. A rerun with `PREV_CACHE` set takes only a few minutes.
#
# **Outputs** (all in the new results folder):
#
# | file | content |
# |---|---|
# | `config.json`, `environment.txt` | settings, commits, checkpoint sha256, pip freeze |
# | `triplets.csv`, `frames.csv` | canonical triplets and their frames (file, location, datetime, label, split) |
# | `gallery_resolution.json` | gallery size, hour range, Δt_avg, nearest-neighbour spacing |
# | `preds_npz_inspection.txt` | keys, shapes and mapping result for the existing `preds.npz` |
# | `frame_cache.npz` (+ `cache_shards/`) | per-frame z (512, fp32), top-1/top-2 index and score, pred time, IR score |
# | `fp16_noise_floor.csv` | d_cos between the fp16 and fp32 encodings of the same frames (noise floor for C) |
# | `pairs.csv` | one row per authentic 1 s pair, with all distances |
# | `burst_summary.csv`, `f12_comparison.txt` | burst medians and IQR per transition, compared with `transition_distances.csv` |
# | `A_fcr_overall.csv`, `A_fcr_per_transition.csv`, `A_fcr_per_camera.csv`, `A_fcr_per_camera_transition.csv` | Exp A |
# | `C_class_counts_*.csv`, `C_dcos_hist.png`, `C_dcos_vs_dH.png` | Exp C |
# | `B_dburst_by_label*.csv`, `B_boxplot_*.png` | Exp B |
# | `summary.json` | headline numbers |
#
# Existing Drive files under `datasets/` (including `gtloc_f12/`) are only read.

# %%
# ===== 1. CONFIG =====
import os, json, time, datetime as _dt
from pathlib import Path

BASE = Path('/content/drive/MyDrive/Colab Notebooks/CCTV')
CCT_DIR = BASE / 'datasets' / 'cct20'
CCT_IMG_DIR = CCT_DIR / 'eccv_18_all_images_sm'
CCT_ANN_DIR = CCT_DIR / 'eccv_18_annotation_files'
ANN_SPLITS = ['train', 'cis_val', 'cis_test', 'trans_val', 'trans_test']
F12_DIR = CCT_DIR / 'gtloc_f12'          # read-only: existing outputs
SKYFINDER_DIR = BASE / 'datasets' / 'skyfinder'  # not used here; kept for reference

EXPERIMENT = 'cct20_phase2'
RUN_STAMP = _dt.datetime.now().strftime('%Y%m%d_%H%M')
RESULTS_DIR = BASE / 'results' / f'{EXPERIMENT}_{RUN_STAMP}'

# Inference settings (keep identical to Phase 1 for comparability)
TENCROP = True            # official GT-Loc eval protocol
USE_FP16 = True           # autocast fp16 for the image encoder on T4; retrieval runs in fp32
BATCH_FRAMES = 24         # frames per batch (x10 crops with TenCrop = 240 crops); lower to 12 on OOM
NUM_WORKERS = 2           # Colab standard has 2 vCPUs
SHARD_SIZE = 2048         # frames per resume shard
# Image source: 'lila' = download + extract the official CCT20 archive to local disk (recommended);
#               'drive' = read eccv_18_all_images_sm/ from Drive (slow, Drive FUSE can time out on the flat folder)
IMAGE_SOURCE = 'lila'
LILA_IMAGES_URL = 'https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz'
LILA_IMAGES_BYTES = 6_492_615_601          # verified with HTTP HEAD (Last-Modified 2023-06-29)
LILA_IMAGES_MD5_B64 = 'gUPBeqKhKHK2byhP8hFTHw=='   # x-goog-hash md5 from the same HEAD
LILA_ANN_URL = 'https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz'
LILA_ANN_BYTES = 2_997_071
VERIFY_MD5 = True         # ~1 min extra on Colab; set False to skip
LOCAL_DATA_DIR = Path('/content/cct20_lila')   # local disk only, never Drive
DELETE_TAR_AFTER_EXTRACT = True                # frees ~6.5 GB of local /content disk
MAX_MISSING_FRAC = 0.01   # stop before inference if more than 1% of the triplet frames have no image file
COPY_TO_LOCAL = True      # 'drive' mode only: copy needed frames to /content/cct20_local with retries
LOCAL_IMG_DIR = Path('/content/cct20_local')

# Reuse: a previous frame_cache.npz OR a previous results folder; None = compute
PREV_CACHE = None         # e.g. BASE/'results'/'cct20_phase2_20261121_0930'
REUSE_PREDS_NPZ = 'auto'  # 'auto' | False: use gtloc_f12/preds.npz if it maps to frames

# Triplet filter
EXPECTED_TRIPLETS, EXPECTED_FRAMES, EXPECTED_LOCATIONS = 15976, 47928, 18
EXPECTED_F12_BURST_MEDIAN = 0.04
REQUIRE_SEQ_LEN = 'auto'  # 'auto' (pick the variant matching 15,976) | 'eq3' | 'ge3'

# Experiments
FCR_THRESHOLDS_H = [0.5, 1.0]
DCOS_HOP_MAX = 1e-4
DCOS_SHIFT_MIN = 0.05
HOP_STEP_H = 'auto'       # 'auto' = deck's Δt_avg from cvt.npy (8766 h / N), or a number in hours
HOP_TOL_FRAC = 0.5        # |dH - step| <= HOP_TOL_FRAC * step  ->  "≈ one gallery step"
IR_THRESH = 2.0           # mean |channel difference| (0-255) on a 64x64 thumbnail below this => grayscale/IR
FP16_CALIB_N = 256        # frames for the fp16-vs-fp32 noise-floor check (0 = skip)

print('Results will be written to:', RESULTS_DIR)

# %%
# ===== 2. SETUP: packages, Drive, GPU =====
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.45,<5', 'gdown'], check=True)

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('Not in Colab: make sure BASE points to a local copy.')

assert IMAGE_SOURCE in ('lila', 'drive'), IMAGE_SOURCE
if IMAGE_SOURCE == 'drive':
    assert CCT_IMG_DIR.exists(), f'Missing {CCT_IMG_DIR}'
RESULTS_DIR.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=False)   # fails instead of reusing/overwriting a folder

def out(name):
    """Return a NEW path inside RESULTS_DIR; refuse to overwrite anything."""
    p = RESULTS_DIR / name
    if p.exists():
        raise FileExistsError(f'Refusing to overwrite {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

import torch
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU. Select Runtime -> Change runtime type -> T4 GPU.')
DEVICE = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0),
      f'{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB')
print('torch', torch.__version__)

# %%
# ===== 3. PURE FUNCTIONS (no GPU / no Drive; unit-testable) =====
import math
import os
import time
import numpy as np
import pandas as pd

def frac_month(month, day):
    """Same convention as the GT-Loc evaluator: month + (day-1)/31."""
    return np.asarray(month, float) + (np.asarray(day, float) - 1.0) / 31.0

def frac_hour(hour, minute, second):
    return np.asarray(hour, float) + np.asarray(minute, float) / 60.0 + np.asarray(second, float) / 3600.0

def circ_dist(a, b, period):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % period
    return np.minimum(d, period - d)

def d_burst(dH, dM):
    return np.sqrt((np.asarray(dH) / 12.0) ** 2 + (np.asarray(dM) / 6.0) ** 2)

def cos_dist(z1, z2):
    z1 = np.asarray(z1, np.float64); z2 = np.asarray(z2, np.float64)
    n = np.linalg.norm(z1, axis=-1) * np.linalg.norm(z2, axis=-1)
    return 1.0 - np.sum(z1 * z2, axis=-1) / np.maximum(n, 1e-12)

def parse_datetimes(series):
    try:
        return pd.to_datetime(series, errors='coerce', format='mixed')
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors='coerce')

def load_cct_images(ann_jsons):
    """ann_jsons: dict split -> loaded COCO-CT dict. Returns (images_df, annotations_df, categories_df, n_raw).
    Deterministic: images are de-duplicated by id keeping the first split in the given split order. Annotations are
    the UNION over all split JSONs, de-duplicated by (image_id, category_id). Category ids must map to the same
    name in every JSON; otherwise it raises instead of letting the last file win."""
    imgs, anns, cats = [], [], {}
    for split, d in ann_jsons.items():
        for im in d.get('images', []):
            r = dict(im); r['split'] = split; imgs.append(r)
        for a in d.get('annotations', []):
            anns.append({'image_id': a['image_id'], 'category_id': a['category_id']})
        for c in d.get('categories', []):
            if c['id'] in cats and cats[c['id']] != c['name']:
                raise ValueError(f"category id {c['id']} is {cats[c['id']]!r} in one JSON and {c['name']!r} in {split}")
            cats[c['id']] = c['name']
    im = pd.DataFrame(imgs)
    n_raw = len(im)
    if 'datetime' not in im.columns and 'date_captured' in im.columns:
        im['datetime'] = im['date_captured']
    im['id'] = im['id'].astype(str)
    im = im.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    an = pd.DataFrame(anns, columns=['image_id', 'category_id'])
    an['image_id'] = an['image_id'].astype(str)
    an = an.drop_duplicates().sort_values(['image_id', 'category_id']).reset_index(drop=True)
    cat = pd.DataFrame(sorted(cats.items()), columns=['category_id', 'name'])
    return im, an, cat, n_raw

LABEL_RULE = ("image = 'empty' iff ALL of its annotations (union over all split JSONs, de-duplicated by "
              "(image_id, category_id)) have category 'empty'; 'animal' if it has any non-empty annotation; "
              "'unlabeled' if it has none. Triplet = 'empty' iff all 3 frames are 'empty'; 'animal' if any frame is "
              "'animal'; otherwise 'unlabeled' (excluded from B).")

def image_labels(images_df, ann_df, cat_df):
    """Per image, by LABEL_RULE. Independent of file, annotation and split order."""
    empty_ids = set(cat_df.loc[cat_df['name'].str.strip().str.lower() == 'empty', 'category_id'])
    if not empty_ids:
        raise ValueError("No 'empty' category in annotations")
    is_empty = ann_df['category_id'].isin(empty_ids)
    all_empty = is_empty.groupby(ann_df['image_id']).all()
    lab = images_df['id'].astype(str).map(all_empty)          # True / False / NaN (no annotation)
    return lab.map({True: 'empty', False: 'animal'}).fillna('unlabeled').to_numpy(dtype=object)

def label_diagnostics(ann_df, cat_df):
    """Counts that explain the labels: images with >1 annotation and images mixing 'empty' with an animal."""
    empty_ids = set(cat_df.loc[cat_df['name'].str.strip().str.lower() == 'empty', 'category_id'])
    g = ann_df.assign(e=ann_df['category_id'].isin(empty_ids)).groupby('image_id')['e']
    n = g.size(); a = g.all(); o = g.any()
    return {'annotated_images': int(len(n)), 'multi_annotation_images': int((n > 1).sum()),
            'mixed_empty_and_animal': int((o & ~a).sum())}

def check_missing_frames(n_missing, n_total, examples, max_frac=0.01):
    """Fail fast (before inference) when too many triplet frames have no image file."""
    if n_total == 0:
        raise RuntimeError('No triplet frames were built. Check the annotation JSONs and the triplet filter.')
    frac = n_missing / n_total
    if frac > max_frac:
        raise RuntimeError(
            f'{n_missing:,} of {n_total:,} triplet frames ({100*frac:.2f}%) have no image file '
            f'(limit {100*max_frac:.1f}%). Examples: {list(examples)[:5]}. The image folder is incomplete or '
            f'could not be listed; use IMAGE_SOURCE="lila" or fix the folder.')
    return frac

def require_frames(frames_df):
    if frames_df is None or len(frames_df) == 0:
        raise RuntimeError('No frames to process: the triplet table is empty. Re-run cell 4 and check its counts '
                           '(triplets, missing image files) before inference.')

def build_triplets(images_df, require_seq_len='ge3', gap_s=1.0, frame_base=None):
    """Canonical frames 1-2-3 of each sequence with both gaps == gap_s seconds.
    images_df needs: id, file_name, location, datetime (parsed), seq_id, frame_num, seq_num_frames.
    Returns (triplets_df, stats)."""
    df = images_df.copy()
    stats = {'images_in': len(df)}
    if frame_base is None:
        frame_base = int(pd.to_numeric(df['frame_num'], errors='coerce').min())
    stats['frame_base'] = frame_base
    df['frame_num'] = pd.to_numeric(df['frame_num'], errors='coerce')
    df['seq_num_frames'] = pd.to_numeric(df['seq_num_frames'], errors='coerce')
    df = df[df['frame_num'].isin([frame_base, frame_base + 1, frame_base + 2])].copy()
    df['frame_num'] = df['frame_num'].astype(int)
    if require_seq_len == 'eq3':
        df = df[df['seq_num_frames'] == 3]
    elif require_seq_len == 'ge3':
        df = df[df['seq_num_frames'] >= 3]
    # sequences with a duplicated frame_num are ambiguous -> drop
    dup = df.duplicated(subset=['seq_id', 'frame_num'], keep=False)
    stats['seqs_dropped_dup_frame'] = int(df.loc[dup, 'seq_id'].nunique())
    df = df[~df['seq_id'].isin(df.loc[dup, 'seq_id'])]
    wide = df.pivot(index='seq_id', columns='frame_num')
    need = [frame_base, frame_base + 1, frame_base + 2]
    complete = wide['id'][need].notna().all(axis=1)
    wide = wide[complete]
    stats['seqs_with_frames_123'] = int(len(wide))
    t = [wide['datetime'][k] for k in need]
    g12 = (t[1] - t[0]).dt.total_seconds()
    g23 = (t[2] - t[1]).dt.total_seconds()
    same_loc = (wide['location'][need[0]].astype(str) == wide['location'][need[1]].astype(str)) & \
               (wide['location'][need[1]].astype(str) == wide['location'][need[2]].astype(str))
    ok = (g12 == gap_s) & (g23 == gap_s) & same_loc
    stats['seqs_bad_datetime'] = int((g12.isna() | g23.isna()).sum())
    stats['seqs_gap_not_1s'] = int((~ok & g12.notna() & g23.notna()).sum())
    w = wide[ok]
    trip = pd.DataFrame({'seq_id': w.index.astype(str),
                         'location': w['location'][need[0]].astype(str).values})
    for j, k in enumerate(need, start=1):
        trip[f'id{j}'] = w['id'][k].astype(str).values
        trip[f'file{j}'] = w['file_name'][k].astype(str).values
        trip[f'dt{j}'] = w['datetime'][k].values
        if 'split' in w:
            trip[f'split{j}'] = w['split'][k].values
        if 'label' in w:
            trip[f'label{j}'] = w['label'][k].values
    trip = trip.sort_values(['location', 'dt1', 'seq_id']).reset_index(drop=True)
    stats['triplets'] = int(len(trip))
    stats['frames'] = int(3 * len(trip))
    stats['locations'] = int(trip['location'].nunique())
    return trip, stats

def triplet_label(trip):
    """'empty' iff all three frames are 'empty'; 'animal' if any frame 'animal'; else 'unlabeled'."""
    L = trip[['label1', 'label2', 'label3']].values
    is_animal = (L == 'animal').any(axis=1)
    all_empty = (L == 'empty').all(axis=1)
    return np.where(is_animal, 'animal', np.where(all_empty, 'empty', 'unlabeled'))

def ir_group(ir_flags_3):
    """ir_flags_3: (N,3) bool -> 'night_ir' (all IR), 'day_color' (none), 'mixed'."""
    a = np.asarray(ir_flags_3, bool)
    return np.where(a.all(1), 'night_ir', np.where(~a.any(1), 'day_color', 'mixed'))

def classify_jump(idx_changed, dH, dcos, step_h, tol_frac=0.5, hop_max=1e-4, shift_min=0.05):
    """Exp C classes. Pairs whose retrieved stamp did not change -> 'no_jump'."""
    idx_changed = np.asarray(idx_changed, bool); dH = np.asarray(dH, float); dcos = np.asarray(dcos, float)
    near_step = np.abs(dH - step_h) <= tol_frac * step_h
    cls = np.full(dH.shape, 'other', dtype=object)
    cls[dcos > shift_min] = 'feature_shift'
    cls[(dcos < hop_max) & near_step] = 'gallery_hop'
    cls[~idx_changed] = 'no_jump'
    return cls

def rate_ci(k, n, z=1.96):
    """Wilson 95% interval, in %."""
    if n == 0:
        return (float('nan'),) * 3
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)

def fcr_table(pairs, thresholds, by=None):
    rows = []
    groups = [((), pairs)] if not by else pairs.groupby(by, observed=True)
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by or [], key)); r['n_pairs'] = len(g)
        for t in thresholds:
            k = int((g['dH'] > t).sum())
            p, lo, hi = rate_ci(k, len(g))
            r[f'n_conflict_{t}h'] = k; r[f'FCR_{t}h_%'] = round(p, 3)
            r[f'FCR_{t}h_ci95_lo'] = round(lo, 3); r[f'FCR_{t}h_ci95_hi'] = round(hi, 3)
        rows.append(r)
    return pd.DataFrame(rows)

def tail_stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(n=0, median=np.nan, iqr=np.nan, p90=np.nan, p95=np.nan, p99=np.nan, mean=np.nan)
    q = np.percentile(x, [25, 50, 75, 90, 95, 99])
    return dict(n=len(x), median=q[1], iqr=q[2] - q[0], p90=q[3], p95=q[4], p99=q[5], mean=x.mean())

def list_dir_retry(path, retries=6, backoff=5.0, sleep=time.sleep):
    """One os.scandir listing -> {file name: full path}, with exponential backoff on OSError/TimeoutError.
    Drive FUSE can raise 'A Google Drive timeout has occurred' on huge flat folders. os.walk would silently swallow
    that error and return nothing, which is how the earlier run got an empty listing."""
    last = None
    for k in range(retries):
        try:
            with os.scandir(path) as it:
                return {e.name: e.path for e in it if e.is_file()}
        except (OSError, TimeoutError) as exc:
            last = exc
            wait = backoff * (2 ** k)
            print(f'listing {path} failed ({exc}); retry {k+1}/{retries} in {wait:.0f}s')
            sleep(wait)
    raise RuntimeError(f'Could not list {path} after {retries} attempts: {last}')

def ir_score_from_pil(img, size=64):
    """Mean absolute channel difference (0-255) on a thumbnail; ~0 for grayscale/IR frames."""
    a = np.asarray(img.convert('RGB').resize((size, size)), np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0)

print('Pure functions ready.')

# %%
# ===== 3b. DATA SOURCE: annotations + images (local disk) =====
import base64, hashlib, shutil, tarfile, urllib.request

def http_size(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])

def download(url, dst, expected_bytes):
    """Resumable download to local disk (aria2c if present, else wget -c), then a size check."""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        live = http_size(url)
        if live != expected_bytes:
            print(f'NOTE: server size {live:,} differs from the recorded {expected_bytes:,}; using the server size.')
            expected_bytes = live
    except Exception as exc:
        print('HEAD failed, using the recorded size:', exc)
    if dst.exists() and dst.stat().st_size == expected_bytes:
        print('Already downloaded:', dst); return dst
    t0 = time.time()
    if shutil.which('aria2c'):
        cmd = ['aria2c', '-c', '-x', '16', '-s', '16', '--summary-interval=30', '--console-log-level=warn',
               '-d', str(dst.parent), '-o', dst.name, url]
    else:
        cmd = ['wget', '-c', '--tries=20', '--retry-connrefused', '--waitretry=10', '--progress=dot:giga',
               '-O', str(dst), url]
    for attempt in range(3):
        if subprocess.run(cmd).returncode == 0 and dst.stat().st_size == expected_bytes:
            break
        print(f'download attempt {attempt+1} incomplete ({dst.stat().st_size if dst.exists() else 0:,} bytes); resuming')
    size = dst.stat().st_size
    if size != expected_bytes:
        raise RuntimeError(f'{dst.name}: got {size:,} bytes, expected {expected_bytes:,}')
    print(f'Downloaded {size/2**30:.2f} GiB in {(time.time()-t0)/60:.1f} min')
    return dst

def md5_b64(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(16 << 20), b''):
            h.update(blk)
    return base64.b64encode(h.digest()).decode()

def extract(tar_path, dest, prefix):
    """Extract to local disk with progress; skip if a completion marker exists. Only members under prefix/."""
    marker = Path(dest) / f'.{prefix}.extracted'
    if marker.exists():
        print('Already extracted:', Path(dest) / prefix); return Path(dest) / prefix
    t0 = time.time(); n = 0
    with tarfile.open(tar_path, 'r:gz') as tf:
        for m in tf:
            name = m.name.lstrip('./')
            if not name.startswith(prefix + '/') and name != prefix:
                continue
            if m.issym() or m.islnk() or '..' in Path(name).parts:
                continue
            tf.extract(m, dest, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
            n += 1
            if n % 5000 == 0:
                print(f'  extracted {n:,} entries ({(time.time()-t0)/60:.1f} min)')
    marker.write_text(str(n))
    print(f'Extracted {n:,} entries in {(time.time()-t0)/60:.1f} min')
    return Path(dest) / prefix

# Annotations: Drive JSONs if all five are there, otherwise the LILA archive on local disk
def find_split_jsons(folder):
    found = {}
    for s in ANN_SPLITS:
        cands = sorted(Path(folder).glob(f'*{s}*.json'), key=lambda c: len(c.name)) if Path(folder).exists() else []
        if cands:
            found[s] = cands[0]
    return found
ann_files = find_split_jsons(CCT_ANN_DIR)
if len(ann_files) < len(ANN_SPLITS):
    print(f'Drive annotations incomplete ({sorted(ann_files)}); downloading the LILA annotation archive.')
    tgz = download(LILA_ANN_URL, LOCAL_DATA_DIR / 'eccv_18_annotations.tar.gz', LILA_ANN_BYTES)
    ann_files = find_split_jsons(extract(tgz, LOCAL_DATA_DIR, 'eccv_18_annotation_files'))
assert len(ann_files) == len(ANN_SPLITS), f'missing annotation JSONs: {set(ANN_SPLITS) - set(ann_files)}'

# Images
if IMAGE_SOURCE == 'lila':
    img_dir = LOCAL_DATA_DIR / 'eccv_18_all_images_sm'
    if not (LOCAL_DATA_DIR / '.eccv_18_all_images_sm.extracted').exists():
        tar = download(LILA_IMAGES_URL, LOCAL_DATA_DIR / 'eccv_18_all_images_sm.tar.gz', LILA_IMAGES_BYTES)
        if VERIFY_MD5:
            got = md5_b64(tar)
            if got != LILA_IMAGES_MD5_B64:
                raise RuntimeError(f'MD5 mismatch for {tar.name}: {got} != {LILA_IMAGES_MD5_B64}. Delete it and rerun.')
            print('MD5 OK')
        img_dir = extract(tar, LOCAL_DATA_DIR, 'eccv_18_all_images_sm')
        if DELETE_TAR_AFTER_EXTRACT:
            tar.unlink()   # local /content temp file only
    IMAGE_DIR = img_dir
else:
    IMAGE_DIR = CCT_IMG_DIR
t0 = time.time()
avail = list_dir_retry(IMAGE_DIR)
print(f'Image source: {IMAGE_SOURCE} -> {IMAGE_DIR}: {len(avail):,} files listed in {time.time()-t0:.0f}s')

# %%
# ===== 4. CCT20 ANNOTATIONS -> CANONICAL 1 s TRIPLETS =====
ann = {}
for s in ANN_SPLITS:
    with open(ann_files[s], 'rb') as fh:
        raw_bytes = fh.read()
    ann[s] = json.loads(raw_bytes)
    print(f'{s:11s} <- {ann_files[s]}: {len(ann[s].get("images", [])):,} images, '
          f'{len(ann[s].get("annotations", [])):,} annotations, md5 {hashlib.md5(raw_bytes).hexdigest()}')

images_df, ann_df, cat_df, n_raw = load_cct_images(ann)
print(f'Images: {n_raw:,} raw -> {len(images_df):,} after de-dup by id; annotations {len(ann_df):,}')
print('Image fields:', sorted(images_df.columns))
images_df['datetime'] = parse_datetimes(images_df['datetime'])
print('Unparsed datetimes:', int(images_df['datetime'].isna().sum()))
images_df['label'] = image_labels(images_df, ann_df, cat_df)
print('LABEL RULE:', LABEL_RULE)
print('Label diagnostics:', label_diagnostics(ann_df, cat_df))
print('Image labels:', images_df['label'].value_counts().sort_index().to_dict(),
      '(official eccv_18 JSONs give animal 53,850 / empty 4,014)')

variants = {}
for v in (['eq3', 'ge3'] if REQUIRE_SEQ_LEN == 'auto' else [REQUIRE_SEQ_LEN]):
    variants[v] = build_triplets(images_df, require_seq_len=v)
    print(v, variants[v][1])
pick = next((v for v in variants if variants[v][1]['triplets'] == EXPECTED_TRIPLETS), list(variants)[-1])
triplets, tstats = variants[pick]
print(f'Using variant {pick!r}')

# Missing image files: membership in the single directory listing from cell 3b (no per-file stat on Drive)
def resolve(fn):
    return avail.get(fn) or avail.get(os.path.basename(fn))
for j in (1, 2, 3):
    triplets[f'path{j}'] = triplets[f'file{j}'].map(resolve)
miss_frames = pd.concat([triplets.loc[triplets[f'path{j}'].isna(), f'file{j}'] for j in (1, 2, 3)])
frac = check_missing_frames(len(miss_frames), 3 * len(triplets), miss_frames.head(5).tolist(), MAX_MISSING_FRAC)
missing = triplets[['path1', 'path2', 'path3']].isna().any(axis=1)
print(f'Missing frames: {len(miss_frames):,} ({100*frac:.2f}%); triplets dropped: {int(missing.sum()):,}')
triplets = triplets[~missing].reset_index(drop=True)
triplets['tlabel'] = triplet_label(triplets)

n_t, n_f, n_l = len(triplets), 3 * len(triplets), triplets['location'].nunique()
print(f'\nTRIPLETS {n_t:,} (expected {EXPECTED_TRIPLETS:,}) | FRAMES {n_f:,} (expected {EXPECTED_FRAMES:,}) | '
      f'LOCATIONS {n_l} (expected {EXPECTED_LOCATIONS})')
print('Triplet labels:', triplets['tlabel'].value_counts().to_dict())
if (n_t, n_f, n_l) != (EXPECTED_TRIPLETS, EXPECTED_FRAMES, EXPECTED_LOCATIONS):
    print('WARNING: counts differ from the deck. The deck triplets were built from LILA '
          'caltech_images_20210113.json; continue, but report this difference.')

frames = pd.concat([
    triplets[['seq_id', 'location', f'id{j}', f'file{j}', f'path{j}', f'dt{j}', f'label{j}']]
    .set_axis(['seq_id', 'location', 'image_id', 'file_name', 'path', 'datetime', 'label'], axis=1)
    .assign(frame=j) for j in (1, 2, 3)], ignore_index=True)
require_frames(frames)
assert frames['image_id'].is_unique, 'an image appears in two triplets'
triplets.drop(columns=['path1', 'path2', 'path3']).to_csv(out('triplets.csv'), index=False)
frames.drop(columns=['path']).to_csv(out('frames.csv'), index=False)

# %%
# ===== 5. GT-LOC ENCODERS + GALLERY (same as Phase 1) =====
import hashlib, gc, gdown
import torch.nn.functional as F

REPO_URL = 'https://github.com/dshatwell23/gtloc.git'
REPO_COMMIT = '6e7d9d2745d15a101b3b4dd05151807af77831ea'
REPO_DIR = Path('/content/gtloc')
if not (REPO_DIR / '.git').exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', REPO_COMMIT], check=True)
subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', REPO_COMMIT], check=True)

CHECKPOINT_ID = '170wGyCYLSF3DWOvITIAbtN1OkmDlS4v6'
CKPT_PATH = REPO_DIR / 'ckpts' / 'gtloc.pt'
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CKPT_PATH.exists() or CKPT_PATH.stat().st_size == 0:
    gdown.download(id=CHECKPOINT_ID, output=str(CKPT_PATH), quiet=False)
sha = hashlib.sha256()
with CKPT_PATH.open('rb') as fh:
    for blk in iter(lambda: fh.read(8 << 20), b''):
        sha.update(blk)
CKPT_SHA256 = sha.hexdigest()
print('checkpoint sha256:', CKPT_SHA256)

sys.path.insert(0, str(REPO_DIR / 'src'))
from model.image_encoder import ImageEncoder
from model.time_encoder import TimeEncoder
from datasets.transforms import get_transforms

image_encoder = ImageEncoder(hidden_dim=768, embedding_dim=512, freeze_backbone=True)
time_encoder = TimeEncoder(sigma=[2**0, 2**4, 2**8], embedding_dim=512, dropout_prob=None)
try:
    raw = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
except Exception as exc:
    print(f'weights_only=True failed ({type(exc).__name__}); retrying with weights_only=False (official checkpoint)')
    raw = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
sd = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
sd = {k.removeprefix('module.'): v for k, v in sd.items()}
img_sd = {k.removeprefix('image_encoder.'): v for k, v in sd.items() if k.startswith('image_encoder.')}
tim_sd = {k.removeprefix('time_encoder.'): v for k, v in sd.items() if k.startswith('time_encoder.')}
li = image_encoder.load_state_dict(img_sd, strict=False)
lt = time_encoder.load_state_dict(tim_sd, strict=False)
train_i = {n for n, _ in image_encoder.named_parameters()}
train_t = {n for n, _ in time_encoder.named_parameters()}
bad = ([k for k in li.missing_keys if k in train_i and not k.startswith('backbone.')]
       + [k for k in li.unexpected_keys if k in train_i]
       + [k for k in lt.missing_keys if k in train_t] + [k for k in lt.unexpected_keys if k in train_t])
if bad:
    raise RuntimeError(f'Checkpoint key mismatch: {bad[:20]}')
image_encoder = image_encoder.eval().to(DEVICE)
time_encoder = time_encoder.eval().to(DEVICE)
del raw, sd, img_sd, tim_sd; gc.collect()

time_gallery = np.load(REPO_DIR / 'src' / 'model' / 'time_galleries' / 'cvt.npy')
assert time_gallery.ndim == 2 and time_gallery.shape[1] == 5
with torch.inference_mode():
    gallery_feats = F.normalize(time_encoder(torch.from_numpy(time_gallery).float().to(DEVICE)), dim=1).float()
G_MONTH = frac_month(time_gallery[:, 0], time_gallery[:, 1])
G_HOUR = frac_hour(time_gallery[:, 2], time_gallery[:, 3], time_gallery[:, 4])
print('Gallery:', time_gallery.shape, '| encoded', tuple(gallery_feats.shape))

# Gallery time resolution
N_G = len(time_gallery)
dt_avg_h = 365.25 * 24 / N_G
rng = np.random.default_rng(0)
samp = rng.choice(N_G, min(3000, N_G), replace=False)
nn_dH, nn_d = [], []
for i in samp:
    dh = circ_dist(G_HOUR, G_HOUR[i], 24); dm = circ_dist(G_MONTH, G_MONTH[i], 12)
    dd = d_burst(dh, dm); dd[i] = np.inf; j = int(np.argmin(dd))
    nn_dH.append(dh[j]); nn_d.append(dd[j])
uh = np.unique(np.round(G_HOUR, 6))
gal_res = {
    'n_rows': int(N_G), 'hour_min': float(G_HOUR.min()), 'hour_max': float(G_HOUR.max()),
    'dt_avg_h_deck_formula': dt_avg_h,
    'unique_hour_values': int(len(uh)), 'median_spacing_unique_hours_h': float(np.median(np.diff(uh))),
    'median_nn_dH_h_torus': float(np.median(nn_dH)), 'median_nn_dburst_torus': float(np.median(nn_d)),
}
HOP_STEP = dt_avg_h if HOP_STEP_H == 'auto' else float(HOP_STEP_H)
gal_res['hop_step_h_used'] = HOP_STEP
out('gallery_resolution.json').write_text(json.dumps(gal_res, indent=2))
print(json.dumps(gal_res, indent=2))
if G_HOUR.min() > 3 or G_HOUR.max() < 23:
    print(f'NOTE: cvt.npy covers only {G_HOUR.min():.2f}-{G_HOUR.max():.2f} h. Night frames can never '
          'be predicted correctly: they snap to dawn or dusk stamps.')

# %%
# ===== 6. INSPECT EXISTING gtloc_f12 OUTPUTS (read-only) =====
insp_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); insp_lines.append(s)

for name in ['status.txt', 'transition_summary.txt']:
    p = F12_DIR / name
    if p.exists():
        log(f'--- {name} ---'); log(p.read_text()[:3000])

f12_preds = None   # dict image_id -> {'pred_time':(5,), 'z':(512,) optional}
p = F12_DIR / 'preds.npz'
if p.exists():
    npz = np.load(p, allow_pickle=True)
    log('--- preds.npz keys ---')
    for k in npz.files:
        a = npz[k]
        ex = a.ravel()[:3].tolist() if a.size else []
        log(f'{k:24s} shape={a.shape} dtype={a.dtype} first={ex}')
    keys = {k.lower(): k for k in npz.files}
    def first_key(names):
        return next((keys[n] for n in names if n in keys), None)
    id_key = first_key(['image_id', 'image_ids', 'ids', 'id', 'file_name', 'file_names', 'files',
                        'paths', 'path', 'names'])
    time_key = first_key(['pred_time', 'pred_times', 'pred', 'preds', 'predicted_time', 'pred_t'])
    z_key = first_key(['z', 'emb', 'embedding', 'embeddings', 'feats', 'features', 'image_features'])
    mh_keys = (first_key(['pred_month', 'month_pred', 'pred_months']),
               first_key(['pred_hour', 'hour_pred', 'pred_hours']))
    log('mapping guess -> id:', id_key, '| time:', time_key, '| z:', z_key, '| month/hour:', mh_keys)
    try:
        if id_key is not None:
            ids = np.asarray(npz[id_key]).reshape(-1).astype(str)
            ids = np.array([os.path.splitext(os.path.basename(x))[0] for x in ids])
            n = len(ids)
            rec = {}
            if time_key is not None and npz[time_key].reshape(n, -1).shape[1] == 5:
                rec['pred_time'] = npz[time_key].reshape(n, 5)
            if z_key is not None and npz[z_key].reshape(n, -1).shape[1] == 512:
                rec['z'] = npz[z_key].reshape(n, 512).astype(np.float32)
            if rec:
                f12_preds = {'ids': ids, **rec}
                fid = frames['image_id'].astype(str).values
                fstem = frames['file_name'].map(lambda x: os.path.splitext(os.path.basename(x))[0]).values
                pos = {v: i for i, v in enumerate(ids)}
                hit = np.array([pos.get(a, pos.get(b, -1)) for a, b in zip(fid, fstem)])
                f12_preds['frame_pos'] = hit
                log(f'preds.npz maps to {int((hit >= 0).sum()):,}/{len(frames):,} frames; fields {list(rec)}')
        if f12_preds is None:
            log('preds.npz has no usable id + time/z keys. Will recompute everything.')
    except Exception as exc:
        log('preds.npz mapping failed:', repr(exc)); f12_preds = None
else:
    log('No preds.npz found.')

f12_td = None
p = F12_DIR / 'transition_distances.csv'
if p.exists():
    f12_td = pd.read_csv(p)
    log('--- transition_distances.csv ---', f12_td.shape, list(f12_td.columns))
    log(f12_td.head(3).to_string())
    log(f12_td.describe().T.to_string())
out('preds_npz_inspection.txt').write_text('\n'.join(insp_lines) + '\n')

# %%
# ===== 7. PER-FRAME INFERENCE + CACHE (z, top-2 retrieval, pred time, IR score) =====
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torch.utils.data import Dataset, DataLoader

CACHE_FIELDS = ['z', 'top1_idx', 'top1_score', 'top2_idx', 'top2_score', 'ir_score']
cache = {}   # image_id -> dict

def load_cache_file(path):
    path = Path(path)
    if path.is_dir() and not (path / 'frame_cache.npz').exists():
        shards = sorted((path / 'cache_shards').glob('shard_*.npz'))   # resume an interrupted run
        for sp in shards:
            d = np.load(sp, allow_pickle=False)
            for i, iid in enumerate(d['image_id'].astype(str)):
                cache[iid] = {k: d[k][i] for k in CACHE_FIELDS}
        print(f'Loaded {len(cache):,} frames from {len(shards)} shards in {path}')
        return
    if path.is_dir():
        path = path / 'frame_cache.npz'
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta'])) if 'meta' in d.files else {}
    if meta and (meta.get('tencrop') != TENCROP or meta.get('ckpt_sha256') != CKPT_SHA256):
        print('WARNING: PREV_CACHE settings differ:', meta)
    for i, iid in enumerate(d['image_id'].astype(str)):
        cache[iid] = {k: d[k][i] for k in CACHE_FIELDS}
    print(f'Loaded {len(d["image_id"]):,} cached frames from {path}')

if PREV_CACHE:
    load_cache_file(PREV_CACHE)

require_frames(frames)
todo = frames[~frames['image_id'].isin(cache)].reset_index(drop=True)
print(f'Frames to encode: {len(todo):,} / {len(frames):,}')

if len(todo) and IMAGE_SOURCE == 'drive' and COPY_TO_LOCAL:
    LOCAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    def _cp(src, retries=5):
        dst = LOCAL_IMG_DIR / os.path.basename(src)
        for k in range(retries):
            src_size = None
            try:
                src_size = os.path.getsize(src)
                # copyfile can leave a short dest when Drive times out; only a full-size file counts.
                if dst.exists() and dst.stat().st_size == src_size:
                    return str(dst)
                shutil.copyfile(src, dst)   # destination is local /content, never Drive
                if dst.stat().st_size != src_size:
                    raise OSError(f'truncated copy of {src}')
                return str(dst)
            except (OSError, TimeoutError) as exc:
                try:
                    if src_size is not None and dst.exists() and dst.stat().st_size != src_size:
                        dst.unlink()
                except OSError:
                    pass
                if k == retries - 1:
                    raise RuntimeError(f'copy failed after {retries} tries: {src}: {exc}')
                time.sleep(2 * 2 ** k)
    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        todo['local_path'] = list(ex.map(_cp, todo['path']))
    print(f'Copied to local disk in {(time.time()-t0)/60:.1f} min')
else:
    todo['local_path'] = todo['path']

_, eval_tf = get_transforms(tencrop=TENCROP)

class FrameDS(Dataset):
    def __init__(self, df): self.p = df['local_path'].tolist(); self.ids = df['image_id'].tolist()
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        try:
            img = Image.open(self.p[i]); img.load()
        except Exception as exc:
            raise RuntimeError(f'Cannot read {self.p[i]}: {exc}')
        irs = ir_score_from_pil(img)
        return eval_tf(img), irs, self.ids[i]

@torch.inference_mode()
def encode_batch(x, fp16=USE_FP16):
    """x: [B,10,3,224,224] (TenCrop) or [B,3,224,224]. Returns z (mean of L2-normalised crop embeddings, fp32),
    which gives exactly the Phase-1 retrieval: argmax_i mean_c cos(v_c, T_i) = argmax_i z·T_i."""
    x = x.to(DEVICE, non_blocking=True)
    if x.ndim == 5:
        b, c = x.shape[:2]; x = x.reshape(-1, *x.shape[2:])
    else:
        b, c = x.shape[0], 1
    with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
        v = image_encoder(x)
    v = F.normalize(v.float(), dim=1).reshape(b, c, -1)
    return v.mean(dim=1)

@torch.inference_mode()
def retrieve(z):
    s = z @ gallery_feats.T          # fp32 [B, N_G]
    sc, ix = s.topk(2, dim=1)
    return ix.cpu().numpy(), sc.cpu().numpy()

shard_dir = RESULTS_DIR / 'cache_shards'
shard_dir.mkdir(exist_ok=False)
def flush(buf, k):
    np.savez(out(f'cache_shards/shard_{k:05d}.npz'), image_id=np.array(buf['image_id']),
             **{f: np.stack(buf[f]) for f in CACHE_FIELDS})

if len(todo):
    loader = DataLoader(FrameDS(todo), batch_size=BATCH_FRAMES, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    buf = {f: [] for f in CACHE_FIELDS + ['image_id']}; k = 0; t0 = time.time(); done = 0
    for x, irs, ids in loader:
        z = encode_batch(x)
        ix, sc = retrieve(z)
        z = z.cpu().numpy().astype(np.float32)
        for b in range(len(ids)):
            r = dict(z=z[b], top1_idx=ix[b, 0], top1_score=sc[b, 0], top2_idx=ix[b, 1], top2_score=sc[b, 1],
                     ir_score=float(irs[b]))
            cache[ids[b]] = r
            buf['image_id'].append(ids[b])
            for f in CACHE_FIELDS: buf[f].append(r[f])
        done += len(ids)
        if len(buf['image_id']) >= SHARD_SIZE:
            flush(buf, k); k += 1; buf = {f: [] for f in CACHE_FIELDS + ['image_id']}
            el = time.time() - t0
            print(f'{done:,}/{len(todo):,} frames | {done/el:.1f} fr/s | ETA {(len(todo)-done)/(done/el)/60:.1f} min')
    if buf['image_id']:
        flush(buf, k)
    print(f'Encoded {done:,} frames in {(time.time()-t0)/60:.1f} min')

missing_cache = [i for i in frames['image_id'] if i not in cache]
if missing_cache:
    raise RuntimeError(f'{len(missing_cache):,} frames are missing from the cache, e.g. {missing_cache[:3]}')
order = frames['image_id'].tolist()
CZ = np.stack([cache[i]['z'] for i in order]).astype(np.float32)
C1 = np.array([int(cache[i]['top1_idx']) for i in order])
C2 = np.array([int(cache[i]['top2_idx']) for i in order])
S1 = np.array([float(cache[i]['top1_score']) for i in order])
S2 = np.array([float(cache[i]['top2_score']) for i in order])
IRS = np.array([float(cache[i]['ir_score']) for i in order])
meta = json.dumps({'tencrop': TENCROP, 'fp16': USE_FP16, 'ckpt_sha256': CKPT_SHA256, 'repo_commit': REPO_COMMIT,
                   'gallery_rows': int(N_G), 'created': RUN_STAMP})
np.savez(out('frame_cache.npz'), image_id=np.array(order), z=CZ, top1_idx=C1, top1_score=S1, top2_idx=C2,
         top2_score=S2, ir_score=IRS, meta=np.array(meta))
frames['pred_idx'] = C1
frames['pred_month'] = G_MONTH[C1]; frames['pred_hour'] = G_HOUR[C1]
frames['top1_score'] = S1; frames['top2_score'] = S2; frames['margin'] = S1 - S2
frames['ir_score'] = IRS; frames['is_ir'] = IRS < IR_THRESH
frames['true_hour'] = pd.to_datetime(frames['datetime']).dt.hour + pd.to_datetime(frames['datetime']).dt.minute / 60
print('IR fraction:', round(frames['is_ir'].mean(), 3), '| IR score quantiles:',
      np.round(np.percentile(IRS, [1, 10, 50, 90, 99]), 2))

# Agreement with preds.npz (if it mapped)
if f12_preds is not None and 'pred_time' in f12_preds:
    pos = f12_preds['frame_pos']; m = pos >= 0
    pt = f12_preds['pred_time'][pos[m]]
    ph = frac_hour(pt[:, 2], pt[:, 3], pt[:, 4]); pm = frac_month(pt[:, 0], pt[:, 1])
    agree = (np.isclose(ph, frames.loc[m, 'pred_hour']) & np.isclose(pm, frames.loc[m, 'pred_month'])).mean()
    print(f'Agreement of recomputed pred time with preds.npz: {100*agree:.2f}% of {int(m.sum()):,} frames')

# fp16 noise floor for Exp C thresholds
if FP16_CALIB_N and USE_FP16:
    sub = frames.sample(min(FP16_CALIB_N, len(frames)), random_state=0)
    sub = sub.assign(local_path=sub['path'])
    dl = DataLoader(FrameDS(sub), batch_size=8, num_workers=NUM_WORKERS)
    dc = []
    for x, _, _ in dl:
        dc.append(cos_dist(encode_batch(x, fp16=True).cpu().numpy(), encode_batch(x, fp16=False).cpu().numpy()))
    dc = np.concatenate(dc)
    pd.DataFrame({'d_cos_fp16_vs_fp32': dc}).to_csv(out('fp16_noise_floor.csv'), index=False)
    print(f'fp16 vs fp32 d_cos: median {np.median(dc):.2e}, p99 {np.percentile(dc, 99):.2e} '
          f'(must be << DCOS_HOP_MAX={DCOS_HOP_MAX:g}; otherwise set USE_FP16=False)')

# %%
# ===== 8. PAIR TABLE + BURST CHECK =====
fi = {iid: k for k, iid in enumerate(order)}
tl = triplets.set_index('seq_id')
rows = []
for tr, (a, b) in {'f1->f2': (1, 2), 'f2->f3': (2, 3)}.items():
    ia = np.array([fi[x] for x in triplets[f'id{a}']]); ib = np.array([fi[x] for x in triplets[f'id{b}']])
    dH = circ_dist(G_HOUR[C1[ia]], G_HOUR[C1[ib]], 24)
    dM = circ_dist(G_MONTH[C1[ia]], G_MONTH[C1[ib]], 12)
    rows.append(pd.DataFrame({
        'seq_id': triplets['seq_id'].values, 'location': triplets['location'].values, 'transition': tr,
        'tlabel': triplets['tlabel'].values,
        'idx_a': C1[ia], 'idx_b': C1[ib], 'idx_changed': C1[ia] != C1[ib],
        'pred_hour_a': G_HOUR[C1[ia]], 'pred_hour_b': G_HOUR[C1[ib]],
        'dH': dH, 'dM': dM, 'd_burst': d_burst(dH, dM), 'd_cos': cos_dist(CZ[ia], CZ[ib]),
        'margin_a': S1[ia] - S2[ia], 'margin_b': S1[ib] - S2[ib],
        'b_is_a_top2': C1[ib] == C2[ia],
        'ir_a': IRS[ia] < IR_THRESH, 'ir_b': IRS[ib] < IR_THRESH,
        'true_hour_a': frames['true_hour'].values[ia],
    }))
pairs = pd.concat(rows, ignore_index=True)
tir = ir_group(np.stack([IRS[[fi[x] for x in triplets[f'id{j}']]] < IR_THRESH for j in (1, 2, 3)], 1))
triplets['ir_group'] = tir
pairs['ir_group'] = pairs['seq_id'].map(dict(zip(triplets['seq_id'], tir)))
pairs.to_csv(out('pairs.csv'), index=False)

bs = []
for tr, g in pairs.groupby('transition'):
    for q in ['dH', 'dM', 'd_burst']:
        t = tail_stats(g[q]); bs.append({'transition': tr, 'quantity': q, **t})
burst_summary = pd.DataFrame(bs)
burst_summary.to_csv(out('burst_summary.csv'), index=False)
print(burst_summary.round(3).to_string(index=False))
med12 = float(pairs.loc[pairs.transition == 'f1->f2', 'd_burst'].median())
print(f'f1->f2 d_burst median = {med12:.3f} (deck {EXPECTED_F12_BURST_MEDIAN}); '
      f'{"OK" if abs(round(med12, 2) - EXPECTED_F12_BURST_MEDIAN) < 0.011 else "MISMATCH: check settings"}')

cmp_lines = [f'recomputed f1->f2 median d_burst {med12:.4f}']
if f12_td is not None:
    for c in f12_td.select_dtypes('number').columns:
        cmp_lines.append(f'transition_distances.csv {c}: median {f12_td[c].median():.4f}')
    if 'seq_id' in f12_td.columns:
        piv = pairs.pivot(index='seq_id', columns='transition', values='d_burst')
        mm = f12_td.assign(_sid=f12_td['seq_id'].astype(str)).merge(piv, left_on='_sid', right_index=True)
        cmp_lines.append(f'seq_id overlap with transition_distances.csv: {len(mm):,}/{len(piv):,}')
out('f12_comparison.txt').write_text('\n'.join(cmp_lines) + '\n')
print('\n'.join(cmp_lines))

# %%
# ===== 9. EXPERIMENT A: FALSE CONFLICT RATE on authentic 1 s pairs =====
A_overall = fcr_table(pairs, FCR_THRESHOLDS_H)
A_trans = fcr_table(pairs, FCR_THRESHOLDS_H, by=['transition'])
A_cam = fcr_table(pairs, FCR_THRESHOLDS_H, by=['location']).sort_values(f'FCR_{FCR_THRESHOLDS_H[0]}h_%', ascending=False)
A_cam_tr = fcr_table(pairs, FCR_THRESHOLDS_H, by=['location', 'transition'])
A_ir = fcr_table(pairs, FCR_THRESHOLDS_H, by=['ir_group'])
A_overall.to_csv(out('A_fcr_overall.csv'), index=False)
A_trans.to_csv(out('A_fcr_per_transition.csv'), index=False)
A_cam.to_csv(out('A_fcr_per_camera.csv'), index=False)
A_cam_tr.to_csv(out('A_fcr_per_camera_transition.csv'), index=False)
A_ir.to_csv(out('A_fcr_per_ir_group.csv'), index=False)
print('=== A: False Conflict Rate (pair flagged if predicted dH > threshold) ===')
print(A_overall.to_string(index=False)); print(A_trans.to_string(index=False))
print(A_cam.to_string(index=False)); print(A_ir.to_string(index=False))

# %%
# ===== 10. EXPERIMENT C: LATENT COSINE vs GALLERY HOP =====
import matplotlib
import matplotlib.pyplot as plt
pairs['C_class'] = classify_jump(pairs['idx_changed'], pairs['dH'], pairs['d_cos'], HOP_STEP,
                                 HOP_TOL_FRAC, DCOS_HOP_MAX, DCOS_SHIFT_MIN)
pairs['C_hop_any_dH'] = pairs['idx_changed'] & (pairs['d_cos'] < DCOS_HOP_MAX)   # diagnostic: hop at any dH
C_counts = pairs['C_class'].value_counts().rename_axis('class').reset_index(name='n')
C_counts['%'] = (100 * C_counts['n'] / len(pairs)).round(2)
C_tr = pd.crosstab(pairs['transition'], pairs['C_class'], margins=True)
C_cam = pd.crosstab(pairs['location'], pairs['C_class'], margins=True)
C_big = pairs[pairs['dH'] >= 0.5]['C_class'].value_counts().rename_axis('class').reset_index(name='n_dH_ge_0.5h')
C_counts.to_csv(out('C_class_counts_overall.csv'), index=False)
C_tr.to_csv(out('C_class_counts_per_transition.csv'))
C_cam.to_csv(out('C_class_counts_per_camera.csv'))
C_big.to_csv(out('C_class_counts_dH_ge_0p5h.csv'), index=False)
print(f'=== C: step={HOP_STEP:.3f} h ± {100*HOP_TOL_FRAC:.0f}%, hop d_cos<{DCOS_HOP_MAX:g}, shift d_cos>{DCOS_SHIFT_MIN:g} ===')
print(C_counts.to_string(index=False)); print(C_tr); print(C_big.to_string(index=False))
print('d_cos quantiles (p1,p10,p50,p90,p99):', np.percentile(pairs['d_cos'], [1, 10, 50, 90, 99]))
print('jumps that are hops at any dH (idx changed & d_cos<hop_max):', int(pairs['C_hop_any_dH'].sum()))
print('jumps landing on the previous frame\'s 2nd-best stamp:', int((pairs['idx_changed'] & pairs['b_is_a_top2']).sum()))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.clip(pairs['d_cos'].values, 1e-8, None)
ax.hist(np.log10(x), bins=120, color='steelblue')
for v, c in [(DCOS_HOP_MAX, 'green'), (DCOS_SHIFT_MIN, 'red')]:
    ax.axvline(np.log10(v), color=c, ls='--')
ax.set_xlabel('log10 d_cos(z1, z2)'); ax.set_ylabel('pairs'); ax.set_title('Exp C: latent cosine distance, 1 s pairs')
fig.tight_layout(); fig.savefig(out('C_dcos_hist.png'), dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(np.log10(x), pairs['dH'], s=2, alpha=0.3)
ax.axhline(HOP_STEP, color='gray', ls=':'); ax.axvline(np.log10(DCOS_HOP_MAX), color='green', ls='--')
ax.axvline(np.log10(DCOS_SHIFT_MIN), color='red', ls='--')
ax.set_xlabel('log10 d_cos'); ax.set_ylabel('predicted dH (h)'); ax.set_title('Exp C: d_cos vs dH')
fig.tight_layout(); fig.savefig(out('C_dcos_vs_dH.png'), dpi=150); plt.close(fig)

# %%
# ===== 11. EXPERIMENT B: EMPTY vs ANIMAL BURSTS =====
def strat(df, by):
    r = []
    for key, g in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        r.append({**dict(zip(by, key)), **tail_stats(g['d_burst'])})
    return pd.DataFrame(r)

pb = pairs[pairs['tlabel'].isin(['empty', 'animal'])]
print('Triplet labels used:', triplets['tlabel'].value_counts().to_dict())
B_lab = strat(pb, ['tlabel'])
B_lab_tr = strat(pb, ['transition', 'tlabel'])
B_cam = strat(pb, ['location', 'tlabel'])
B_ir = strat(pb, ['ir_group', 'tlabel'])
B_lab.to_csv(out('B_dburst_by_label.csv'), index=False)
B_lab_tr.to_csv(out('B_dburst_by_label_transition.csv'), index=False)
B_cam.to_csv(out('B_dburst_by_label_camera.csv'), index=False)
B_ir.to_csv(out('B_dburst_by_label_ir.csv'), index=False)
print('=== B: d_burst, empty vs animal ===')
for t in (B_lab, B_lab_tr, B_ir):
    print(t.round(4).to_string(index=False))
print(B_cam.round(4).to_string(index=False))
try:
    from scipy.stats import mannwhitneyu
    e = pb.loc[pb.tlabel == 'empty', 'd_burst']; a = pb.loc[pb.tlabel == 'animal', 'd_burst']
    if len(e) and len(a):
        print('Mann-Whitney U (animal > empty): p =', mannwhitneyu(a, e, alternative='greater').pvalue,
              '| NOTE: pairs within a camera are not independent; treat as descriptive.')
except ImportError:
    pass

for by, fname in [('ir_group', 'B_boxplot_ir.png'), ('location', 'B_boxplot_camera.png')]:
    cats = sorted(pb[by].unique())
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cats) * 2), 4.5))
    data, labels, pos = [], [], []
    for i, c in enumerate(cats):
        for j, lab in enumerate(['empty', 'animal']):
            v = pb.loc[(pb[by] == c) & (pb.tlabel == lab), 'd_burst'].values
            if len(v):
                data.append(v); labels.append(f'{c}\n{lab}'); pos.append(3 * i + j)
    ax.boxplot(data, positions=pos, widths=0.8, showfliers=False)
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel('d_burst'); ax.set_title(f'Exp B: d_burst by {by} (fliers hidden)')
    fig.tight_layout(); fig.savefig(out(fname), dpi=150); plt.close(fig)

# %%
# ===== 12. SUMMARY =====
summary = {
    'results_dir': str(RESULTS_DIR), 'triplets': int(len(triplets)), 'frames': int(len(frames)),
    'locations': int(triplets['location'].nunique()), 'triplet_variant': pick, 'triplet_stats': tstats,
    'f12_dburst_median': med12,
    'FCR_overall_%': {f'{t}h': float(A_overall[f'FCR_{t}h_%'].iloc[0]) for t in FCR_THRESHOLDS_H},
    'C_counts': dict(zip(C_counts['class'], C_counts['n'].astype(int))),
    'B_median_dburst': dict(zip(B_lab['tlabel'], B_lab['median'].astype(float))),
    'B_p95_dburst': dict(zip(B_lab['tlabel'], B_lab['p95'].astype(float))),
    'gallery': gal_res, 'ir_fraction': float(frames['is_ir'].mean()),
}
cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in dict(
    BASE=BASE, TENCROP=TENCROP, USE_FP16=USE_FP16, BATCH_FRAMES=BATCH_FRAMES, PREV_CACHE=PREV_CACHE,
    FCR_THRESHOLDS_H=FCR_THRESHOLDS_H, DCOS_HOP_MAX=DCOS_HOP_MAX, DCOS_SHIFT_MIN=DCOS_SHIFT_MIN,
    HOP_STEP=HOP_STEP, HOP_TOL_FRAC=HOP_TOL_FRAC, IR_THRESH=IR_THRESH, REPO_COMMIT=REPO_COMMIT,
    CKPT_SHA256=CKPT_SHA256).items()}
out('config.json').write_text(json.dumps(cfg, indent=2, default=str))
out('summary.json').write_text(json.dumps(summary, indent=2, default=str))
out('environment.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                                 capture_output=True, text=True).stdout)
print(json.dumps(summary, indent=2, default=str))
print('\n' + '=' * 60)
print('RUN FINISHED')
print('Results folder:', RESULTS_DIR)
print('=' * 60)
