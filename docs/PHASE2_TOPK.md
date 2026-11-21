# Phase-2 follow-up: top-k softmax time readout

Offline CPU analysis of a training-free top-k weighted readout on the same CCT20 eq3 triplets as `docs/PHASE2_RESULTS.md` (run `cct20_phase2_20261120_1751`, 15,976 triplets / 47,928 frames). No image inference was redone. Date: 21 November 2026, Europe/Athens.

All numbers below are taken from `docs/phase2_data/topk/` (copied from the analysis folder also stored on [Drive](https://drive.google.com/drive/folders/1L23MFWeC3e9klmUJHOK1YlUXB-_awVIl)). The scripts are `scripts/topk_readout.py` and `scripts/make_topk_results_md.py`; both take command-line paths and contain no hardcoded machine paths.

## Method

For a frame embedding \(z\) and gallery time embeddings \(T_i\) (16,656 stamps from `cvt.npy`, recomputed on CPU from GT-Loc commit `6e7d9d2` and `gtloc.pt`, SHA-256 `387a9bc7…a4af7`):

1. Score every stamp by the cosine \(s_i = z\cdot T_i\).
2. Keep the top \(k\) stamps and set \(w = \mathrm{softmax}(s/\tau)\). \(\tau=\infty\) means uniform weights \(1/k\).
3. Predicted hour = circular weighted mean of the \(k\) hours (period 24 h). Predicted month = circular weighted mean of the fractional months (period 12).

**\(k=1\) is the Phase-2 argmax baseline.** The weight is 1 on the top stamp, and \(\tau\) is ignored.

Grid: \(k\in\{1,5,10,20,50,100,200,500\}\), \(\tau\in\{10^{-6},3\times 10^{-6},10^{-5},3\times 10^{-5},10^{-4},\infty\}\). Image embeddings \(z\) (512-d, TenCrop mean, fp16 autocast) were read from the 1751 `cache_shards/` plus the nine earlier shards of `cct20_phase2_20261120_1351`, covering all 47,928 frames. IR groups are the same triplet rule as Phase 2 (`ir_score < 2.0` on all three frames → `night_ir`; on none → `day_color`).

**Why \(\tau\) is so small.** The cosine scores are tiny and almost flat. From `sanity.json`: median top-1 score 0.0052; median gap from top-1 to top-5 is \(1.4\times 10^{-5}\), and to top-50 is \(1.1\times 10^{-4}\). Any \(\tau\gtrsim 10^{-3}\), including the 0.01–0.1 range one might have planned, is therefore indistinguishable from uniform weights. The recommended setting is **\(k=200\), \(\tau=10^{-4}\)**: small enough that the softmax still concentrates, large enough that more than the first few neighbours contribute.

## Sanity: \(k=1\) reproduces the Phase-2 baseline

`k1_colab` uses the cached Colab top-1 indices and matches the Phase-2 write-up:

- FCR 24.87% / 10.85% overall; day 10.94% / 1.96%; night IR 30.71% / 14.57%.
- \(d_{\mathrm{burst}}\) medians: animal 0.0407 vs empty 0.0266; day 0.0334 / 0.0310; night 0.0465 / 0.0186.
- C classes: other 79.96%, no_jump 16.23%, feature_shift 3.81%, gallery_hop 0 (`sanity_C_classes_k1_colab.csv`).

A CPU recomputation of the gallery agrees with the Colab top-1 index on 97.2% of frames. The disagreements are fp32 GPU vs CPU rounding at score margins of about \(10^{-6}\). The CPU \(k=1\) FCR is 24.86% / 10.86%. All \(k>1\) configs use the CPU gallery.

## Recommended readout: \(k=200\), \(\tau=10^{-4}\)

Compared with `k1_colab` (from `summary_by_config.csv` / `A_fcr.csv`):

| Split | FCR 0.5 h (k=1) | FCR 0.5 h (k=200) | FCR 1 h (k=1) | FCR 1 h (k=200) |
|---|---:|---:|---:|---:|
| Overall (31,952 pairs) | 24.87% | 18.34% | 10.85% | 6.31% |
| day_color (9,438) | 10.94% | 6.52% | 1.96% | 0.94% |
| night_ir (22,514) | 30.71% | 23.30% | 14.57% | 8.55% |

Relative drop overall: −26% at 0.5 h and −42% at 1 h. The gain is similar on both transitions (f1→f2 26.65% → 19.95%; f2→f3 23.08% → 16.73% at 0.5 h).

With \(\tau=10^{-4}\), FCR almost saturates by \(k=200\): \(k=500\) at the same \(\tau\) is 18.16% / 6.17%, not identical but close. Uniform \(k=500\) goes further, to 15.56% / 4.80%, but ToD MAE at night rises from 7.17 h to 7.28 h.

ToD MAE is essentially unchanged: 5.984 h → 6.000 h overall, 3.163 → 3.164 h by day, 7.167 → 7.189 h at night. ToY MAE is 77.1 → 77.0 days. Stability improved; accuracy did not.

Figures: [FCR vs ToD MAE](phase2_figures/topk_fcr_vs_mae.png) and [FCR vs \(k\)](phase2_figures/topk_fcr_vs_k.png). The full grid is in `phase2_data/topk/summary_by_config.csv`; a short slice is in `key_numbers.csv`.

## Empty vs animal (\(d_{\mathrm{burst}}\))

A continuous readout almost never gives exactly \(dH=0\) (0% of pairs at \(k=200\)), whereas argmax had \(dH=0\) on 16.9% of pairs (the Phase-2 `no_jump` share was 16.23%; the rest are different stamps with the same hour). Median \(d_{\mathrm{burst}}\) therefore rises slightly. The p95 tail shrinks: animal 0.347 → 0.328, empty 0.269 → 0.247.

The night animal/empty median ratio narrows from 2.5× (0.0465 / 0.0186) to 1.8× (0.0474 / 0.0262). Part of the Phase-2 night contrast therefore came from argmax discreteness on near-static empty IR frames, which often stayed on the same stamp (\(dH=0\)). Day stays close to equal (0.0358 vs 0.0338).

## Prediction change vs latent change (C')

Once the output is continuous, the Phase-2 `gallery_hop` / `other` / `no_jump` labels do not apply: there is no retrieved index to “hop”, and \(dH=0\) almost never occurs. C' instead tabulates \(dH\) bins inside each \(d_{\cos}\) bin (`C_dH_vs_dcos.csv`). No pair has \(d_{\cos}<10^{-4}\).

Large jumps under small feature change, \(P(dH>0.5\,\mathrm{h}\mid d_{\cos}\le 0.05)\), fall from 23.8% to 17.0% at \(k=200\), \(\tau=10^{-4}\). Spearman(\(d_{\cos}\), \(d_{\mathrm{burst}}\)) rises from 0.37 to 0.45.

| config | \(d_{\cos}\) bin | \(n\) | \(dH=0\) % | 0–3 min % | 3 min–0.5 h % | 0.5–1 h % | \(>1\) h % |
|---|---|---:|---:|---:|---:|---:|---:|
| k1_colab | \(10^{-4}\)–\(10^{-3}\) | 426 | 36.9 | 16.4 | 39.7 | 5.4 | 1.6 |
| k1_colab | \(10^{-3}\)–\(10^{-2}\) | 17,014 | 23.4 | 15.1 | 45.2 | 10.7 | 5.5 |
| k1_colab | \(10^{-2}\)–0.05 | 13,263 | 9.2 | 11.5 | 45.2 | 18.0 | 16.1 |
| k1_colab | \(>0.05\) | 1,249 | 3.0 | 7.3 | 38.7 | 20.3 | 30.8 |
| k200_t0.0001 | \(10^{-4}\)–\(10^{-3}\) | 426 | 0.0 | 41.8 | 57.0 | 1.2 | 0.0 |
| k200_t0.0001 | \(10^{-3}\)–\(10^{-2}\) | 17,014 | 0.0 | 23.6 | 68.3 | 6.0 | 2.1 |
| k200_t0.0001 | \(10^{-2}\)–0.05 | 13,263 | 0.0 | 10.6 | 60.4 | 18.9 | 10.1 |
| k200_t0.0001 | \(>0.05\) | 1,249 | 0.0 | 6.5 | 43.3 | 24.6 | 25.6 |

## Accuracy caveat and the night gallery gap

The stability gain did not cost ToD accuracy, but there was little accuracy to protect.

A single constant predictor — the circular mean of the true capture times, 3.80 h and month 4.86 — has ToD MAE **5.05 h** and ToY MAE 76.3 days, which is better than GT-Loc's 5.98 h and 77.1 days (`sanity.json`, `constant_predictor`). Uniform guessing has expected ToD MAE 6.00 h and ToY MAE 91.3 days, so ToY is near chance for both the constant and the model.

That overall constant, evaluated on each IR group, is not a fair per-group baseline (day 7.36 h, night 4.08 h) because 3.80 h is pulled toward the night-heavy sample. This run's `sanity.json` does not record separate circular-mean constants fitted inside each group. `scripts/topk_readout.py` now writes `per_group_constant_predictor` on a re-run; those values are not claimed here.

`cvt.npy` covers only 4.96–22.72 h (about 04:57–22:43). 32.8% of all frames, and **46.4% of night-IR frames**, were captured inside that gap. The oracle ToD MAE floor (distance to the nearer gallery edge, else 0) is 0.00 h by day, **0.70 h at night**, and 0.50 h overall. Neither argmax nor the weighted readout ever predicts a time inside the gap (0% of predictions, all configs). Top-k neighbourhoods do not straddle dawn and dusk, so the readout cannot invent night stamps that the gallery does not contain. The observed night ToD MAE (7.17 h at \(k=1\), 7.19 h at \(k=200\)) is far above the 0.70 h floor: the gap is a real limitation, but it is not the whole accuracy problem.

## Recommendation

- Use **\(k=200\), \(\tau=10^{-4}\)** as the default readout next to the \(k=1\) Phase-2 reference. It is training-free, essentially free in ToD/ToY MAE, and cuts FCR by about 26% / 42% at 0.5 h / 1 h.
- Re-read the Phase-2 night animal/empty \(d_{\mathrm{burst}}\) contrast with this discreteness effect in mind (2.5× → 1.8×).
- The main limitation is **accuracy, not pair stability**. GT-Loc underperforms a constant ToD predictor on CCT20, ToY is near chance, and night MAE remains ~7.2 h against a 0.70 h gallery-gap floor. The next experiment should target time-of-day skill (see the corrections section in `PHASE2_RESULTS.md`), not a further readout tweak.

## Reproduce

From the repo root, with local copies of the Phase-2 run (frames/triplets + cache shards), a GT-Loc checkout and `gtloc.pt`:

```text
python scripts/topk_readout.py \
    --run_dir path/to/cct20_phase2_20261120_1751 \
    --cache path/to/cct20_phase2_20261120_1351/cache_shards \
            path/to/cct20_phase2_20261120_1751/cache_shards \
    --gtloc_repo path/to/gtloc --ckpt path/to/gtloc.pt \
    --out docs/phase2_data/topk \
    --ks 1,5,10,20,50,100,200,500
```

`scripts/make_topk_results_md.py --results-dir docs/phase2_data/topk` rebuilds a table-only markdown from the CSVs.
