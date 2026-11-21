# Phase-2 CCT20 results

Write-up of experiments A (false conflict rate), B (empty vs animal \(d_{\mathrm{burst}}\)), and C (latent cosine vs gallery hop) on the canonical CCT20 1 s triplets. All numbers below are taken from the run folder `cct20_phase2_20261120_1751` (Google Drive: [folder](https://drive.google.com/drive/folders/1_HkHSZdCc-Fuu8sbC5ET_LMGBJ8khCsE), id `1_HkHSZdCc-Fuu8sbC5ET_LMGBJ8khCsE`). Compact CSVs and JSON used here are copied under `docs/phase2_data/`; figures under `docs/phase2_figures/`. Larger artefacts (`pairs.csv`, `frames.csv`, `triplets.csv`, `frame_cache.npz`, `cache_shards/`) remain only on Drive.

The notebook that produced the run is `notebooks/CCT20_PHASE2_EXPERIMENTS.ipynb`, which is generated from `scripts/cct20_phase2_experiments.py`. Class definitions below are quoted from that script.

## 1. Setup and configuration

The run wrote results to `/content/drive/MyDrive/Colab Notebooks/CCTV/results/cct20_phase2_20261120_1751`. The folder stamp comes from `datetime.now()` on the runtime (Colab is typically UTC; 17:51 UTC is 19:51 Europe/Athens). `progress.json` records `stage: finished` at `2026-11-20T18:50:37` (if UTC, 20:50:37 Europe/Athens). Encodings were resumed from `cct20_phase2_20261120_1351` (`PREV_CACHE`).

Settings recorded in `config.json`:

| Setting | Value |
|---|---|
| TenCrop | true |
| fp16 | true |
| Batch (frames) | 24 |
| FCR thresholds | 0.5 h, 1.0 h |
| `DCOS_HOP_MAX` | \(10^{-4}\) |
| `DCOS_SHIFT_MIN` | 0.05 |
| Hop step used | 0.52629683 h (`HOP_STEP_H = 'auto'`) |
| Hop tolerance | \(\lvert dH - \mathrm{step}\rvert \le 0.5\times\mathrm{step}\) |
| IR threshold | mean \(\lvert\)channel difference\(\rvert\) (0–255) on a 64×64 thumbnail \(< 2.0\) |
| GT-Loc commit | `6e7d9d2745d15a101b3b4dd05151807af77831ea` |
| Checkpoint SHA-256 | `387a9bc7999ebebe2748df532b93f9beb2365311aefefb63d5957b553a9a4af7` |

`environment.txt` is a `pip freeze` of the Colab runtime. Notable packages: `torch==2.14.1+cu130`, `torchvision==0.29.1+cu130`, `transformers==4.57.6`, `numpy==2.4.6`, `pandas==2.2.3`, `scipy==1.16.3`, `matplotlib==3.10.0`.

The time gallery `cvt.npy` has 16,656 rows and covers only 4.96–22.72 h (about 04:57–22:43). Night frames cannot be predicted at their true hour and snap to dawn or dusk stamps. The hop step is the deck average \(365.25\times 24 / 16{,}656 \approx 0.5263\) h, not a uniform grid: the median nearest-neighbour spacing on the month–hour torus is 0.0311 h.

### Data

From `summary.json` (variant `eq3`):

- 15,976 triplets, 47,928 frames, 18 locations.
- 17,122 sequences had frames 1–2–3; 1,146 were dropped because a gap was not 1 s; 1 was dropped as a duplicate frame; 0 had a bad datetime.
- IR fraction of frames: 70.46% (70.5% to one decimal place). Every triplet was either all day/colour or all night/IR; there was no mixed group.
- Every triplet is labelled `animal` (14,958) or `empty` (1,018). A triplet is `empty` only if all three frames are `empty`, and `animal` if any frame is `animal`.
- Median \(d_{\mathrm{burst}}\) on f1→f2 is 0.04457 (reported as 0.0446). This matches the earlier `gtloc_f12` `transition_distances.csv` median `b12` of 0.0445, with 15,976/15,976 sequence overlap (`f12_comparison.txt`).

\(d_{\mathrm{burst}}=\sqrt{(dH/12)^2+(dM/6)^2}\), with \(dH\) and \(dM\) the circular distances between the two frames' retrieved gallery hour and month. Each triplet contributes two authentic 1 s pairs (f1→f2 and f2→f3), so experiments A and C use 31,952 pairs.

## 2. Experiment A: false conflict rate

A pair is a false conflict if the predicted hour distance satisfies \(dH > t\), for \(t\in\{0.5, 1.0\}\) hours. Rates are percentages with Wilson 95% intervals.

### Overall

| \(n\) | FCR 0.5 h | 95% CI | FCR 1 h | 95% CI |
|---:|---:|---:|---:|---:|
| 31,952 | 24.869% (7,946) | 24.398–25.346 | 10.848% (3,466) | 10.511–11.193 |

To two decimal places this is 24.87% (24.40–25.35) at 0.5 h and 10.85% (10.51–11.19) at 1 h.

### By transition

| Transition | \(n\) | FCR 0.5 h | 95% CI | FCR 1 h | 95% CI |
|---|---:|---:|---:|---:|---:|
| f1→f2 | 15,976 | 26.652% (4,258) | 25.973–27.344 | 11.893% (1,900) | 11.400–12.404 |
| f2→f3 | 15,976 | 23.085% (3,688) | 22.438–23.744 | 9.802% (1,566) | 9.351–10.273 |

### By IR group

The IR group is assigned on the three frames of the parent triplet: `night_ir` if all are IR, `day_color` if none are.

| IR group | \(n\) | FCR 0.5 h | 95% CI | FCR 1 h | 95% CI |
|---|---:|---:|---:|---:|---:|
| day_color | 9,438 | 10.935% (1,032) | 10.321–11.580 | 1.960% (185) | 1.699–2.260 |
| night_ir | 22,514 | 30.710% (6,914) | 30.111–31.316 | 14.573% (3,281) | 14.118–15.040 |

Night IR FCR is about three times the day/colour rate at 0.5 h, and about seven times at 1 h. This split should be read with the gallery caveat above: night frames cannot land on a true night stamp.

### By camera

Cameras are sorted by FCR at 0.5 h, as in `A_fcr_per_camera.csv`. The per-camera × transition table is in `docs/phase2_data/A_fcr_per_camera_transition.csv`.

| Camera | \(n\) | FCR 0.5 h | 95% CI | FCR 1 h | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 100 | 2,700 | 44.889% | 43.021–46.771 | 30.333% | 28.628–32.094 |
| 40 | 434 | 35.484% | 31.128–40.095 | 21.198% | 17.614–25.288 |
| 115 | 1,742 | 30.195% | 28.085–32.393 | 17.681% | 15.961–19.543 |
| 88 | 1,842 | 29.913% | 27.866–32.044 | 14.821% | 13.271–16.517 |
| 130 | 932 | 28.970% | 26.149–31.964 | 7.189% | 5.700–9.029 |
| 46 | 3,568 | 28.139% | 26.688–29.637 | 10.510% | 9.546–11.559 |
| 43 | 2,588 | 26.700% | 25.031–28.438 | 8.964% | 7.924–10.127 |
| 28 | 838 | 25.656% | 22.815–28.720 | 11.456% | 9.473–13.790 |
| 125 | 1,108 | 25.632% | 23.148–28.284 | 9.296% | 7.724–11.149 |
| 7 | 1,180 | 23.729% | 21.389–26.239 | 11.441% | 9.748–13.383 |
| 105 | 1,364 | 23.167% | 21.005–25.480 | 11.070% | 9.513–12.846 |
| 90 | 620 | 21.129% | 18.098–24.515 | 10.161% | 8.023–12.790 |
| 33 | 2,582 | 20.449% | 18.938–22.048 | 5.500% | 4.685–6.447 |
| 51 | 944 | 19.280% | 16.890–21.919 | 9.640% | 7.917–11.689 |
| 38 | 6,158 | 17.830% | 16.895–18.806 | 6.106% | 5.535–6.732 |
| 78 | 1,410 | 17.163% | 15.285–19.220 | 5.177% | 4.138–6.460 |
| 61 | 1,278 | 13.615% | 11.843–15.605 | 3.834% | 2.912–5.033 |
| 108 | 664 | 13.253% | 10.884–16.045 | 3.163% | 2.078–4.786 |

Camera 100 is the clear worst case (44.9% / 30.3%). The lowest rates are on cameras 108 (13.3% / 3.2%), 61 (13.6% / 3.8%) and 78 (17.2% / 5.2%). Camera 40 is high but has only 434 pairs, so its interval is wide.

## 3. Experiment B: empty vs animal \(d_{\mathrm{burst}}\)

`d_burst` is computed on the same 31,952 pairs, then restricted to labels `animal` and `empty` (here, all pairs). Boxplots with fliers hidden: [by IR group](phase2_figures/B_boxplot_ir.png) and [by camera](phase2_figures/B_boxplot_camera.png).

### Overall

| Label | \(n\) | Median | IQR | p90 | p95 | p99 | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| animal | 29,916 | 0.0407 | 0.1060 | 0.2139 | 0.3471 | 0.8015 | 0.0926 |
| empty | 2,036 | 0.0266 | 0.0746 | 0.1615 | 0.2689 | 0.7675 | 0.0709 |

The animal median is higher than the empty median (0.0407 vs 0.0266). Pairs that share a camera are not independent, so this is descriptive only.

### By IR group

| IR group | Label | \(n\) | Median | p95 | Mean |
|---|---|---:|---:|---:|---:|
| day_color | animal | 8,288 | 0.0334 | 0.3313 | 0.0812 |
| day_color | empty | 1,150 | 0.0310 | 0.3454 | 0.0859 |
| night_ir | animal | 21,628 | 0.0465 | 0.3506 | 0.0970 |
| night_ir | empty | 886 | 0.0186 | 0.1618 | 0.0514 |

By day, animal and empty are essentially the same (medians 0.0334 vs 0.0310; the empty mean is slightly higher). At night the animal median is about 2.5× the empty median (0.0465 vs 0.0186), and the empty tail is also lower (p95 0.351 vs 0.162). The overall animal–empty gap is therefore concentrated in night IR.

By transition, the animal > empty gap is present on both f1→f2 (medians 0.0477 vs 0.0238) and f2→f3 (0.0361 vs 0.0270); see `B_dburst_by_label_transition.csv`.

### By camera

| Camera | \(n\) animal | \(n\) empty | Median animal | Median empty | Ratio | p95 animal | p95 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1,092 | 88 | 0.0324 | 0.0185 | 1.76 | 0.296 | 0.146 |
| 28 | 756 | 82 | 0.0370 | 0.0163 | 2.27 | 0.470 | 0.161 |
| 33 | 2,396 | 186 | 0.0367 | 0.0343 | 1.07 | 0.178 | 0.223 |
| 38 | 6,058 | 100 | 0.0272 | 0.0211 | 1.29 | 0.224 | 0.166 |
| 40 | 412 | 22 | 0.0619 | 0.0224 | 2.77 | 0.571 | 0.098 |
| 43 | 2,260 | 328 | 0.0537 | 0.0280 | 1.92 | 0.334 | 0.488 |
| 46 | 3,154 | 414 | 0.0558 | 0.0316 | 1.76 | 0.415 | 0.402 |
| 51 | 746 | 198 | 0.0387 | 0.0184 | 2.10 | 0.269 | 0.194 |
| 61 | 1,236 | 42 | 0.0268 | 0.0143 | 1.87 | 0.168 | 0.097 |
| 78 | 1,318 | 92 | 0.0363 | 0.0297 | 1.22 | 0.260 | 0.262 |
| 88 | 1,798 | 44 | 0.0650 | 0.0178 | 3.65 | 0.486 | 0.579 |
| 90 | 560 | 60 | 0.0443 | 0.0304 | 1.45 | 0.321 | 0.126 |
| 100 | 2,628 | 72 | 0.0970 | 0.0301 | 3.22 | 0.551 | 0.270 |
| 105 | 1,326 | 38 | 0.0330 | 0.0272 | 1.21 | 0.403 | 0.404 |
| 108 | 598 | 66 | 0.0211 | 0.0161 | 1.31 | 0.186 | 0.132 |
| 115 | 1,618 | 124 | 0.0448 | 0.0269 | 1.66 | 0.687 | 0.182 |
| 125 | 1,082 | 26 | 0.0483 | 0.0186 | 2.60 | 0.408 | 0.358 |
| 130 | 878 | 54 | 0.0605 | 0.0324 | 1.87 | 0.303 | 0.190 |

Camera 100 stands out: animal median 0.097 vs empty 0.030, p95 0.551 vs 0.270, and it also has the highest FCR. Several other cameras have a large median ratio (88, 40, 125) but very few empty pairs (44, 22, 26), so those ratios are unstable. Camera 33 is close to parity.

## 4. Experiment C: latent cosine vs gallery hop

### Class definitions

On each pair the model retrieves a gallery stamp for each frame and also stores the 512-d latent \(z\). Let \(d_{\cos}=1-\cos(z_1,z_2)\), and let `idx_changed` be true when the two frames' argmax gallery indices differ. The hop step is \(s=0.52629683\) h, with tolerance fraction 0.5, so "near one gallery step" means \(\lvert dH-s\rvert\le 0.5s\), i.e. \(dH\in[0.263,0.789]\) h.

The four planned labels are assigned by `classify_jump` in `scripts/cct20_phase2_experiments.py` (identical in the notebook). Default arguments are `tol_frac=0.5`, `hop_max=1e-4`, `shift_min=0.05`. The logic, quoted verbatim, is:

```python
def classify_jump(idx_changed, dH, dcos, step_h, tol_frac=0.5, hop_max=1e-4, shift_min=0.05):
    """Exp C classes. Pairs whose retrieved stamp did not change -> 'no_jump'."""
    idx_changed = np.asarray(idx_changed, bool); dH = np.asarray(dH, float); dcos = np.asarray(dcos, float)
    near_step = np.abs(dH - step_h) <= tol_frac * step_h
    cls = np.full(dH.shape, 'other', dtype=object)
    cls[dcos > shift_min] = 'feature_shift'
    cls[(dcos < hop_max) & near_step] = 'gallery_hop'
    cls[~idx_changed] = 'no_jump'
    return cls
```

Assignment is sequential and later rules overwrite earlier ones:

1. Start as `other`.
2. If \(d_{\cos}>0.05\), relabel as `feature_shift`.
3. If \(d_{\cos}<10^{-4}\) **and** \(dH\) is within half a hop step of \(s\), relabel as `gallery_hop`.
4. If the retrieved index did **not** change, relabel as `no_jump` (this wins even when \(d_{\cos}\) is large).

The two cosine cuts are disjoint, so a pair cannot be both `feature_shift` and `gallery_hop`. `other` is therefore the residual after those rules: the stamp **did** change, \(d_{\cos}\) is **not** above 0.05, and the pair is **not** a near-step hop with \(d_{\cos}<10^{-4}\). In other words, `other` is **not** a synonym for the planned gallery-hop category. Gallery hop is a separate, stricter class. `other` covers mid-range latent change (\(10^{-4}\le d_{\cos}\le 0.05\)) at any \(dH\), and would also include a tiny-\(d_{\cos}\) jump whose \(dH\) is outside the step window.

The notebook also computes a diagnostic `C_hop_any_dH = idx_changed & (d_cos < DCOS_HOP_MAX)` (a hop at any \(dH\), not only near the average step). That count is printed at run time and is not stored in the small C CSVs; it is recovered below from `pairs.csv` by applying the same function.

### Overall counts

From `C_class_counts_overall.csv` (31,952 pairs):

| Class | \(n\) | % |
|---|---:|---:|
| other | 25,549 | 79.96 |
| no_jump | 5,185 | 16.23 |
| feature_shift | 1,218 | 3.81 |
| gallery_hop | 0 | 0.00 |

`gallery_hop` does not appear in the CSV because its count is zero. Recomputing `classify_jump` on the saved `pairs.csv` reproduces these three counts exactly and confirms zero gallery hops.

The same recomputation on `pairs.csv` gives:

- Every one of the 25,549 `other` pairs has mid-band cosine distance (\(10^{-4}\le d_{\cos}\le 0.05\)) and a changed index. None have \(d_{\cos}<10^{-4}\).
- The diagnostic "hop at any \(dH\)" is also 0. The smallest \(d_{\cos}\) among stamp-changing pairs is \(1.58\times 10^{-4}\), above `DCOS_HOP_MAX`.
- 6,332 pairs have \(dH\) inside the hop window; all of them changed index. Of those, 5,982 are `other` (mid-band \(d_{\cos}\)) and 350 are `feature_shift`. None meet the gallery-hop cut.
- 2,094 jumps land on the previous frame's second-best stamp (2,083 `other`, 11 `feature_shift`).
- 31 `no_jump` pairs have \(d_{\cos}>0.05\): the latent moved enough to be a feature shift, but the argmax stamp stayed the same, and rule 4 wins.
- \(d_{\cos}\) quantiles over all pairs: p1 \(=9.14\times 10^{-4}\), p10 \(=0.00212\), p50 \(=0.00869\), p90 \(=0.0322\), p99 \(=0.0836\).

So the planned "gallery hop" cell of proposal C is empty. The large `other` mass is mid-band latent change with a stamp change, not a near-zero-\(d_{\cos}\) discretisation hop.

### Pairs with \(dH\ge 0.5\) h

From `C_class_counts_dH_ge_0p5h.csv` (7,949 pairs):

| Class | \(n\) |
|---|---:|
| other | 7,310 |
| feature_shift | 639 |

Again there is no `gallery_hop` (and no `no_jump`, as expected when \(dH\ge 0.5\)). These 7,949 pairs use \(dH\ge 0.5\). Experiment A's conflict count uses the strict inequality \(dH>0.5\) and is 7,946; the three-pair difference is three pairs with \(dH\) exactly 0.5.

### By transition

| Transition | feature_shift | no_jump | other | All |
|---|---:|---:|---:|---:|
| f1→f2 | 710 | 2,367 | 12,899 | 15,976 |
| f2→f3 | 508 | 2,818 | 12,650 | 15,976 |
| All | 1,218 | 5,185 | 25,549 | 31,952 |

### By camera

| Camera | feature_shift | no_jump | other | All |
|---:|---:|---:|---:|---:|
| 7 | 33 | 210 | 937 | 1,180 |
| 28 | 30 | 119 | 689 | 838 |
| 33 | 114 | 606 | 1,862 | 2,582 |
| 38 | 71 | 1,183 | 4,904 | 6,158 |
| 40 | 24 | 61 | 349 | 434 |
| 43 | 124 | 437 | 2,027 | 2,588 |
| 46 | 181 | 534 | 2,853 | 3,568 |
| 51 | 21 | 136 | 787 | 944 |
| 61 | 66 | 234 | 978 | 1,278 |
| 78 | 20 | 202 | 1,188 | 1,410 |
| 88 | 111 | 222 | 1,509 | 1,842 |
| 90 | 23 | 83 | 514 | 620 |
| 100 | 199 | 259 | 2,242 | 2,700 |
| 105 | 18 | 202 | 1,144 | 1,364 |
| 108 | 11 | 150 | 503 | 664 |
| 115 | 41 | 281 | 1,420 | 1,742 |
| 125 | 67 | 127 | 914 | 1,108 |
| 130 | 64 | 139 | 729 | 932 |
| All | 1,218 | 5,185 | 25,549 | 31,952 |

Camera 100 has both the most feature shifts (199) and a high share of `other` (2,242 / 2,700). Camera 38, the largest slice, is dominated by `other` and has a relatively high `no_jump` count (1,183). No camera has a `gallery_hop` column.

Figures: [histogram of \(\log_{10} d_{\cos}\)](phase2_figures/C_dcos_hist.png) and [\(d_{\cos}\) vs \(dH\)](phase2_figures/C_dcos_vs_dH.png). Vertical lines mark `DCOS_HOP_MAX` (green) and `DCOS_SHIFT_MIN` (red); the histogram mass sits between those two cuts, consistent with the `other` majority.

## 5. fp16 noise floor

`fp16_noise_floor.csv` holds \(d_{\cos}\) between the fp16 and fp32 encodings of the same 256 calibration frames (`FP16_CALIB_N`).

| \(n\) | Min | Median | Mean | p99 | Max |
|---:|---:|---:|---:|---:|---:|
| 256 | \(1.11\times 10^{-7}\) | \(8.23\times 10^{-7}\) | \(1.13\times 10^{-6}\) | \(4.84\times 10^{-6}\) | \(1.25\times 10^{-5}\) |

Every value is well below `DCOS_HOP_MAX` \(=10^{-4}\) (max is about eight times smaller). The empty `gallery_hop` class is therefore not an fp16-precision artefact: the hop threshold sits above the noise floor, and authentic 1 s pairs simply do not reach \(d_{\cos}<10^{-4}\) when the stamp changes.

## 6. Interpretation and next steps

The pre-agreed reading of these three experiments was: if C is mostly gallery hops, try a top-\(k\) soft time readout first; if C is mostly feature shifts and B shows animal above empty, a motion-robust / fixed-view attention design is justified; always report night IR separately.

**C is not mostly gallery hops.** The `gallery_hop` class is empty, and so is the looser "tiny \(d_{\cos}\) at any \(dH\)" diagnostic. Proposal C asked for a split of latent \(d_{\cos}\) against gallery hops; under the implemented thresholds, the hop cell is vacant. A top-\(k\) soft readout is therefore not the implication of "hops dominate" on this run. That does not rule it out as a later diagnostic: 5,982 pairs do change stamp inside the average-step window, but their \(d_{\cos}\) is in the mid-band, so they were labelled `other`, not `gallery_hop`. The hop step itself is an average (0.526 h), much coarser than the true median neighbour spacing (0.031 h), which already makes a single-step hop a blunt instrument.

**C is also not mostly feature shifts.** Only 3.81% of pairs are `feature_shift`. Almost 80% are `other` (stamp changed, mid-band \(d_{\cos}\)). The "mostly feature shifts, therefore motion attention" branch of the plan is not supported by these counts.

**B does show animal above empty, but only in night IR.** Overall medians are 0.0407 (animal) vs 0.0266 (empty). By day the two labels barely differ. At night the animal median is about 2.5× empty, and camera 100 is a further standout (0.097 vs 0.030). This is consistent with motion or appearance change mattering in some strata, but it is not a uniform empty-vs-animal effect, and it is entangled with the missing night gallery.

**Night IR should stay a separate report.** Frame IR rate is 70.5%. FCR is 10.94% / 1.96% by day and 30.71% / 14.57% at night. Because `cvt.npy` has no stamps between 22:43 and 04:57, a large night FCR is expected even without a true time jump: night frames are forced onto dawn or dusk. Any follow-up that pools day and night will mix that gallery limitation with genuine instability.

Taken together, the argmax run does not yet justify jumping to a motion-attention design on the C criterion, and it does not show that gallery discretisation (as implemented) is the main source of 0.5 h / 1 h conflicts. The sharper empirical facts are the day/night FCR gap, the night-only animal–empty burst gap, and the poor behaviour of camera 100. A top-k softmax readout was run afterwards as a follow-up, not because hops dominated; the results, and what they change about this reading, are in section 7 and in `docs/PHASE2_TOPK.md`. Codec work remains deferred.

## 7. Corrections after top-k analysis

A training-free top-k softmax readout was applied offline to the same 47,928 embeddings (no new image inference). The recommended setting is \(k=200\), \(\tau=10^{-4}\). Cosine scores on this gallery are almost flat (median top-1 score 0.0052; median top-1 to top-5 gap \(1.4\times 10^{-5}\)), so any \(\tau\gtrsim 10^{-3}\) is effectively uniform. \(k=1\) reproduces the argmax baseline in section 2. Full tables, figures and the method are in [`PHASE2_TOPK.md`](PHASE2_TOPK.md) and `docs/phase2_data/topk/`. The points below are the ones that correct or qualify the argmax write-up.

### (a) GT-Loc vs a constant predictor on CCT20

The top-k readout left accuracy essentially unchanged (ToD MAE 5.98 h → 6.00 h overall; 3.16 → 3.16 h by day; 7.17 → 7.19 h at night; ToY 77.1 → 77.0 days). That is not a success on accuracy: a single constant predictor — the circular mean of the true capture times, 3.80 h and month 4.86 — has ToD MAE **5.05 h** and ToY MAE 76.3 days, which is better than GT-Loc on this set (`sanity.json`). Uniform guessing has expected ToD MAE 6.00 h and ToY MAE 91.3 days, so ToY is near chance for both the model and the constant.

The same overall constant, scored separately on each IR group, is not a fair per-group baseline (day 7.36 h, night 4.08 h), because 3.80 h is pulled toward the night-heavy sample. This run did not save circular-mean constants fitted inside each group; those figures are therefore not reported here. `scripts/topk_readout.py` now records `per_group_constant_predictor` if the analysis is re-run.

### (b) Night gallery gap (22:43–04:57) and the MAE floor

`cvt.npy` covers only 4.96–22.72 h (about 04:57–22:43), as already noted in section 1. Of all frames, 32.8% were captured inside that gap, and **46.4% of night-IR frames**. The oracle ToD MAE floor — distance to the nearer gallery edge, else zero — is 0.00 h by day, **0.70 h at night**, and 0.50 h overall. Neither argmax nor top-k ever predicts a time inside the gap (0% of predictions). The readout cannot invent night stamps the gallery does not contain. Night ToD MAE remains ~7.17 h, far above the 0.70 h floor, so the gap is a real constraint but not the whole accuracy failure.

### (c) Experiment B, night animal/empty ratio, under top-k vs argmax

Under argmax the night animal/empty median \(d_{\mathrm{burst}}\) ratio was 2.5× (0.0465 / 0.0186). Under \(k=200\), \(\tau=10^{-4}\) it narrows to 1.8× (0.0474 / 0.0262). The animal median barely moves; the empty median rises because a continuous readout almost never yields \(dH=0\) (0% of pairs), whereas argmax had \(dH=0\) on 16.9% of pairs, many of them near-static empty IR frames that stayed on the same stamp. Day remains close to equal (0.0358 vs 0.0338). The Phase-2 night B contrast is therefore partly an argmax-discreteness effect and should not be read as a pure motion signal.

### (d) Experiment C: hop/other does not apply to continuous output

The `gallery_hop` / `other` / `no_jump` split is defined on a change of retrieved gallery index and on \(dH\) near the 0.526 h average step. A weighted circular mean has no index to hop, and \(dH=0\) almost never occurs, so those labels do not apply. In their place, C' reports \(P(dH>0.5\,\mathrm{h}\mid d_{\cos}\le 0.05)\): **23.8% under argmax, 17.0% under \(k=200\), \(\tau=10^{-4}\)**. Spearman(\(d_{\cos}\), \(d_{\mathrm{burst}}\)) rises from 0.37 to 0.45. Pair stability improves, but this does not repair time-of-day skill.

### FCR under the recommended readout

For reference, FCR at 0.5 h / 1 h, argmax → \(k=200\), \(\tau=10^{-4}\):

- Overall: 24.87% / 10.85% → 18.34% / 6.31%.
- Day/colour: 10.94% / 1.96% → 6.52% / 0.94%.
- Night IR: 30.71% / 14.57% → 23.30% / 8.55%.

The night/day gap remains large. Because GT-Loc already loses to a constant ToD predictor, a further readout tweak is the wrong next step; see `PHASE2_TOPK.md` for the method and the full grid.

## Sources

- Drive folder `MyDrive/Colab Notebooks/CCTV/results/cct20_phase2_20261120_1751/` ([open](https://drive.google.com/drive/folders/1_HkHSZdCc-Fuu8sbC5ET_LMGBJ8khCsE)).
- Top-k analysis artefacts: [Drive](https://drive.google.com/drive/folders/1L23MFWeC3e9klmUJHOK1YlUXB-_awVIl); local copies in `docs/phase2_data/topk/` and `docs/phase2_figures/topk_fcr_vs_*.png`. Write-up: `docs/PHASE2_TOPK.md`.
- Local copies of the argmax run: `docs/phase2_data/` (CSVs and JSON under ~200 KB) and `docs/phase2_figures/`.
- Class logic: `scripts/cct20_phase2_experiments.py`, function `classify_jump`, also in `notebooks/CCT20_PHASE2_EXPERIMENTS.ipynb`. Mid-band and hop-at-any-\(dH\) counts were recomputed from Drive `pairs.csv` with that function and checked against `C_class_counts_*.csv`.
- Top-k readout: `scripts/topk_readout.py`.
