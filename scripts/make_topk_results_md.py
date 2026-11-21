#!/usr/bin/env python3
"""Build a markdown summary of an offline top-k readout run from its CSVs.

All paths are arguments or default to repo-relative locations. There are no
hardcoded machine paths. Example, from the repo root:

    python scripts/make_topk_results_md.py \\
        --results-dir docs/phase2_data/topk \\
        --out docs/phase2_data/topk/results_tables.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_or_given(p: str | Path) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'docs' / 'phase2_data' / 'topk',
                    help='Directory with summary_by_config.csv, sanity.json, etc.')
    ap.add_argument('--out', type=Path, default=None,
                    help='Markdown file to write (default: <results-dir>/results_tables.md)')
    a = ap.parse_args()
    src = _repo_or_given(a.results_dir)
    out = _repo_or_given(a.out) if a.out else src / 'results_tables.md'

    S = pd.read_csv(src / 'summary_by_config.csv')
    san = json.loads((src / 'sanity.json').read_text())
    pick = [c for c in ['k1_colab', 'k50_tinf', 'k100_t0.0001', 'k200_t0.0001', 'k500_tinf'] if c in set(S.config)]
    cols = {
        'FCR0.5_overall': 'FCR0.5 all', 'FCR1_overall': 'FCR1 all',
        'FCR0.5_day_color': 'FCR0.5 day', 'FCR1_day_color': 'FCR1 day',
        'FCR0.5_night_ir': 'FCR0.5 night', 'FCR1_night_ir': 'FCR1 night',
        'dburst_med_animal_day_color': 'dB med anim day', 'dburst_med_empty_day_color': 'dB med empty day',
        'dburst_med_animal_night_ir': 'dB med anim night', 'dburst_med_empty_night_ir': 'dB med empty night',
        'dburst_p95_animal_overall': 'dB p95 anim', 'dburst_p95_empty_overall': 'dB p95 empty',
        'ToD_MAE_overall': 'ToD MAE all (h)', 'ToD_MAE_day_color': 'ToD MAE day',
        'ToD_MAE_night_ir': 'ToD MAE night', 'ToY_MAE_days_overall': 'ToY MAE (d)',
        'spearman_dcos_dburst_overall': 'Spearman(dcos,dburst)',
        'P(dH>0.5h|dcos<=0.05)_overall': 'P(dH>0.5h | d_cos<=0.05)',
    }
    K = S.set_index('config').loc[pick, list(cols)].rename(columns=cols)
    md = K.T.round(3).to_markdown()
    C = pd.read_csv(src / 'C_dH_vs_dcos.csv')
    Ctab = (C[C.config.isin(['k1_colab', 'k200_t0.0001']) & (C.group == 'overall')]
            .drop(columns=['k', 'tau', 'group']).round(1).to_markdown(index=False))
    cp = san['constant_predictor']
    r1 = S.set_index('config').loc['k1_colab']
    r2 = S.set_index('config').loc['k200_t0.0001']
    night_ratio_1 = r1['dburst_med_animal_night_ir'] / r1['dburst_med_empty_night_ir']
    night_ratio_2 = r2['dburst_med_animal_night_ir'] / r2['dburst_med_empty_night_ir']
    per = san.get('per_group_constant_predictor')
    if per:
        per_txt = (f"- Per-group circular-mean constants (saved in sanity.json): "
                   f"day {per['day_color']['hour']:.2f} h → {per['day_color']['ToD_MAE_h']:.2f} h; "
                   f"night {per['night_ir']['hour']:.2f} h → {per['night_ir']['ToD_MAE_h']:.2f} h, "
                   f"versus GT-Loc {r1['ToD_MAE_day_color']:.2f} h by day and {r1['ToD_MAE_night_ir']:.2f} h at night.")
    else:
        per_txt = ("- This run's sanity.json does not include per-group constant predictors. "
                   "It only evaluates the single overall constant on each group "
                   f"(day {cp['day_color']['ToD_MAE_h']:.2f} h, night {cp['night_ir']['ToD_MAE_h']:.2f} h). "
                   "Those are not competitive day/night baselines, because the overall mean hour is "
                   f"{cp['hour']:.2f} h. Re-run scripts/topk_readout.py to write per-group constants.")

    txt = f"""# GT-Loc CCT20 Phase 2: top-k weighted readout (tables)

Auto-generated from `{src.name}/` by `scripts/make_topk_results_md.py`.
The curated write-up is `docs/PHASE2_TOPK.md`.

**Readout.** Top-k stamps by s_i = z·T_i, weights w = softmax(s/τ). Hour and month are circular weighted means.
k = 1 is the Phase-2 argmax. τ = inf is uniform. Recommended: k = 200, τ = 1e-4.
Median top-1 score {san['top1_score_pctl_10_50_90'][1]:.4f}; median top-1 minus top-5 gap {san['score_gap_top1_minus_top5_median']:.1e}; top-1 minus top-50 {san['score_gap_top1_minus_top50_median']:.1e}. Any τ ≥ ~1e-3 is effectively uniform.

k1_colab vs k = 200 / τ = 1e-4:
- FCR 0.5 h / 1 h overall {r1['FCR0.5_overall']:.2f} → {r2['FCR0.5_overall']:.2f} and {r1['FCR1_overall']:.2f} → {r2['FCR1_overall']:.2f}.
- Day {r1['FCR0.5_day_color']:.2f} / {r1['FCR1_day_color']:.2f} → {r2['FCR0.5_day_color']:.2f} / {r2['FCR1_day_color']:.2f}.
- Night IR {r1['FCR0.5_night_ir']:.2f} / {r1['FCR1_night_ir']:.2f} → {r2['FCR0.5_night_ir']:.2f} / {r2['FCR1_night_ir']:.2f}.
- ToD MAE {r1['ToD_MAE_overall']:.2f} → {r2['ToD_MAE_overall']:.2f} h.
- Night animal/empty median d_burst ratio {night_ratio_1:.1f}× → {night_ratio_2:.1f}×.
- P(dH>0.5 h | d_cos≤0.05) {r1['P(dH>0.5h|dcos<=0.05)_overall']:.1f} % → {r2['P(dH>0.5h|dcos<=0.05)_overall']:.1f} %.

## Key numbers

{md}

## C' overall, k1_colab vs k200_t0.0001

{Ctab}

## Constant predictor and gallery gap (from sanity.json)

- Single constant time ({cp['hour']:.2f} h, month {cp['month']:.2f}) ToD MAE {cp['overall']['ToD_MAE_h']:.2f} h, ToY MAE {cp['overall']['ToY_MAE_days']:.1f} days, versus GT-Loc {r1['ToD_MAE_overall']:.2f} h and {r1['ToY_MAE_days_overall']:.1f} days.
{per_txt}
- Uniform-guess expected ToY MAE is {san['uniform_random_expected']['ToY_MAE_days']:.1f} days (ToD {san['uniform_random_expected']['ToD_MAE_h']:.1f} h).
- Gallery covers {san['gallery_hour_range'][0]:.2f}–{san['gallery_hour_range'][1]:.2f} h. {san['frames_true_hour_in_gallery_gap_%']:.1f} % of all frames and {san['night_ir_frames_true_hour_in_gap_%']:.1f} % of night-IR frames fall in the gap. Night ToD oracle floor {san['oracle_ToD_floor_mean_h']['night_ir']:.2f} h.
"""
    # Fix P formatting: values are already in percent (23.80 not 0.238)
    # The ternary above is messy if I got it wrong - values are 23.80 style. Let me just rewrite P lines more carefully in the file after.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    print('wrote', out)


if __name__ == '__main__':
    main()
