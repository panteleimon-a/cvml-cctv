#!/usr/bin/env python3
"""Offline top-k weighted readout for the GT-Loc time-only pathway on CCT20 eq3 triplets (Phase-2 follow-up).

No image inference. Inputs are the Phase-2 per-frame image embeddings z (cache shards / frame_cache.npz),
frames.csv / triplets.csv of the same run, the GT-Loc repo (time encoder + cvt.npy) and gtloc.pt.

Readout for config (k, tau): s_i = z . T_i over the gallery, take the top-k stamps, w = softmax(s/tau),
predicted hour = circular weighted mean of the k hours (period 24 h), predicted month = circular weighted mean of
the k fractional months (period 12). k=1 is the Phase-2 argmax baseline. tau=inf means uniform weights.

Metrics per config, split overall / day_color / night_ir (triplet IR group, IR = ir_score < 2.0 on all 3 frames):
  A  FCR at 0.5 h and 1 h on f1->f2 and f2->f3 pairs (pair flagged if circular predicted dH > threshold)
  B  d_burst = sqrt((dH/12)^2 + (dM/6)^2): median / p95 for animal vs empty triplets
  C' prediction change (dH bins) vs latent cosine distance d_cos(z_a, z_b) bins; Spearman(d_cos, d_burst)
  Acc ToD MAE (h, circular) and ToY MAE (months, circular) against the true capture datetime.

Recommended default from the offline 2026-11-21 run: k=200, tau=1e-4.
k=1 reproduces the Phase-2 argmax baseline. The gallery cosine scores are nearly
flat, so tau >= about 1e-3 (including 0.01-0.1) is effectively uniform weights.

Every input and output path is a command-line argument; the script has no
hardcoded machine paths. Run it from the repo root, for example:

    python scripts/topk_readout.py --run_dir <phase2-run-dir> \
        --cache <earlier-cache_shards> <run-cache_shards> \
        --gtloc_repo <gtloc-checkout> --ckpt <gtloc.pt> \
        --out docs/phase2_data/topk \
        --ks 1,5,10,20,50,100,200,500
"""
import argparse, glob, json, math, os, sys, types
from pathlib import Path
import numpy as np, pandas as pd

def frac_month(month, day):
    return np.asarray(month, float) + (np.asarray(day, float) - 1.0) / 31.0

def frac_hour(h, m, s):
    return np.asarray(h, float) + np.asarray(m, float) / 60.0 + np.asarray(s, float) / 3600.0

def circ_dist(a, b, period):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % period
    return np.minimum(d, period - d)

def d_burst(dH, dM):
    return np.sqrt((np.asarray(dH) / 12.0) ** 2 + (np.asarray(dM) / 6.0) ** 2)

def cos_dist(z1, z2):
    z1 = np.asarray(z1, np.float64); z2 = np.asarray(z2, np.float64)
    n = np.linalg.norm(z1, axis=-1) * np.linalg.norm(z2, axis=-1)
    return 1.0 - np.sum(z1 * z2, axis=-1) / np.maximum(n, 1e-12)

def classify_jump(idx_changed, dH, dcos, step_h, tol_frac=0.5, hop_max=1e-4, shift_min=0.05):
    idx_changed = np.asarray(idx_changed, bool); dH = np.asarray(dH, float); dcos = np.asarray(dcos, float)
    near_step = np.abs(dH - step_h) <= tol_frac * step_h
    cls = np.full(dH.shape, 'other', dtype=object)
    cls[dcos > shift_min] = 'feature_shift'
    cls[(dcos < hop_max) & near_step] = 'gallery_hop'
    cls[~idx_changed] = 'no_jump'
    return cls

def circ_wmean(vals, w, period, offset=0.0):
    ang = 2 * np.pi * (vals - offset) / period
    c = (w * np.cos(ang)).sum(1); s = (w * np.sin(ang)).sum(1)
    m = (np.arctan2(s, c) % (2 * np.pi)) * period / (2 * np.pi) + offset
    return m, np.sqrt(c ** 2 + s ** 2)          # mean, resultant length R in [0, 1]

def readout(TS, TI, GH, GM, k, tau):
    s = TS[:, :k].astype(np.float64)
    if k == 1:
        w = np.ones_like(s)
    elif np.isinf(tau):
        w = np.full_like(s, 1.0 / k)
    else:
        e = np.exp((s - s[:, :1]) / tau); w = e / e.sum(1, keepdims=True)
    h, Rh = circ_wmean(GH[TI[:, :k]], w, 24.0)
    m, Rm = circ_wmean(GM[TI[:, :k]], w, 12.0, offset=1.0)   # fractional month in [1, 13)
    return h, m, Rh, w.max(1)

def load_gallery(gtloc_repo, ckpt):
    import torch, torch.nn.functional as F
    pkg = types.ModuleType('model'); pkg.__path__ = [str(Path(gtloc_repo) / 'src' / 'model')]
    sys.modules['model'] = pkg                       # avoid model/__init__ (needs transformers / CLIP)
    from model.time_encoder import TimeEncoder
    te = TimeEncoder(sigma=[2 ** 0, 2 ** 4, 2 ** 8], embedding_dim=512, dropout_prob=None)
    raw = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = raw.get('state_dict', raw); sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    te.load_state_dict({k.removeprefix('time_encoder.'): v for k, v in sd.items() if k.startswith('time_encoder.')},
                       strict=True)
    g = np.load(Path(gtloc_repo) / 'src' / 'model' / 'time_galleries' / 'cvt.npy')
    with torch.inference_mode():
        G = F.normalize(te.eval()(torch.from_numpy(g).float()), dim=1).numpy()
    return g, G

def load_cache(frames, sources):
    pos = {k: i for i, k in enumerate(frames['image_id'].astype(str))}; n = len(frames)
    Z = np.zeros((n, 512), np.float32); T1 = np.full(n, -1); IR = np.zeros(n); seen = np.zeros(n, bool)
    files = []
    for s in sources:
        p = Path(s)
        files += [p] if p.suffix == '.npz' else sorted(p.glob('*.npz'))
    for f in files:
        d = np.load(f); ids = d['image_id'].astype(str); z = d['z']; t1 = d['top1_idx']; ir = d['ir_score']
        for j, i in enumerate(ids):
            q = pos.get(i)
            if q is None or seen[q]:
                continue
            seen[q] = True; Z[q] = z[j]; T1[q] = t1[j]; IR[q] = ir[j]
    if not seen.all():
        raise RuntimeError(f'{(~seen).sum()} frames missing from the cache sources')
    return Z, T1, IR

def topk_scores(Z, G, K, chunk=2000):
    TI = np.zeros((len(Z), K), np.int32); TS = np.zeros((len(Z), K), np.float32)
    for st in range(0, len(Z), chunk):
        s = Z[st:st + chunk] @ G.T
        ix = np.argpartition(-s, K, axis=1)[:, :K]; sc = np.take_along_axis(s, ix, 1)
        o = np.argsort(-sc, 1)
        TI[st:st + chunk] = np.take_along_axis(ix, o, 1); TS[st:st + chunk] = np.take_along_axis(sc, o, 1)
    return TI, TS

DCOS_BINS = [0, 1e-4, 1e-3, 1e-2, 5e-2, np.inf]
DCOS_LBL = ['<1e-4', '1e-4..1e-3', '1e-3..1e-2', '1e-2..0.05', '>0.05']
DH_BINS = [-1, 1e-9, 0.05, 0.5, 1.0, np.inf]
DH_LBL = ['0', '0..3min', '3min..0.5h', '0.5..1h', '>1h']

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True, type=Path,
                    help='Phase-2 results folder with frames.csv and triplets.csv')
    ap.add_argument('--cache', nargs='+', required=True, type=Path,
                    help='frame_cache.npz files and/or cache_shards directories')
    ap.add_argument('--gtloc_repo', required=True, type=Path,
                    help='Checkout of dshatwell23/gtloc (time encoder + cvt.npy)')
    ap.add_argument('--ckpt', required=True, type=Path, help='Path to gtloc.pt')
    ap.add_argument('--out', required=True, type=Path, help='Output directory (created if needed)')
    ap.add_argument('--ks', default='1,5,10,20,50,100,200,500',
                    help='Comma-separated k values (default: the documented grid)')
    ap.add_argument('--taus', default='1e-6,3e-6,1e-5,3e-5,1e-4,inf',
                    help='Comma-separated softmax temperatures; inf = uniform')
    ap.add_argument('--ir_thresh', type=float, default=2.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(Path(a.run_dir) / 'frames.csv'); trip = pd.read_csv(Path(a.run_dir) / 'triplets.csv')
    g, G = load_gallery(a.gtloc_repo, a.ckpt)
    GM = frac_month(g[:, 0], g[:, 1]); GH = frac_hour(g[:, 2], g[:, 3], g[:, 4])
    step_h = 365.25 * 24 / len(g)
    Z, T1c, IR = load_cache(frames, a.cache)
    ks = [int(x) for x in a.ks.split(',')]; taus = [float(x) for x in a.taus.split(',')]
    TI, TS = topk_scores(Z, G, max(ks))
    dt = pd.to_datetime(frames['datetime'])
    TH = frac_hour(dt.dt.hour, dt.dt.minute, dt.dt.second); TM = frac_month(dt.dt.month, dt.dt.day)
    fi = {iid: k for k, iid in enumerate(frames['image_id'].astype(str))}
    idx = {j: np.array([fi[x] for x in trip[f'id{j}'].astype(str)]) for j in (1, 2, 3)}
    irf = np.stack([IR[idx[j]] < a.ir_thresh for j in (1, 2, 3)], 1)
    tgroup = np.where(irf.all(1), 'night_ir', np.where(~irf.any(1), 'day_color', 'mixed'))
    fgroup = np.empty(len(frames), object)
    for j in (1, 2, 3):
        fgroup[idx[j]] = tgroup
    gap_floor = np.where((TH >= GH.min()) & (TH <= GH.max()), 0.0,
                         np.minimum(circ_dist(TH, GH.min(), 24), circ_dist(TH, GH.max(), 24)))
    sanity = {'cpu_top1_vs_colab_top1_agreement': float((TI[:, 0] == T1c).mean()),
              'top1_score_pctl_10_50_90': np.percentile(TS[:, 0], [10, 50, 90]).tolist(),
              'score_gap_top1_minus_top5_median': float(np.median(TS[:, 0] - TS[:, 4])),
              'score_gap_top1_minus_top50_median': float(np.median(TS[:, 0] - TS[:, 49])) if TS.shape[1] >= 50 else None,
              'gallery_hour_range': [float(GH.min()), float(GH.max())],
              'frames_true_hour_in_gallery_gap_%': float(100 * (gap_floor > 0).mean()),
              'night_ir_frames_true_hour_in_gap_%': float(100 * (gap_floor[fgroup == 'night_ir'] > 0).mean()),
              'oracle_ToD_floor_mean_h': {gname: float(gap_floor[fgroup == gname].mean()) if gname != 'overall'
                                          else float(gap_floor.mean()) for gname in ['overall', 'day_color', 'night_ir']}}

    # Trivial reference: one constant time for every frame (circular mean of the true times).
    ch, _ = circ_wmean(TH[None, :], np.full((1, len(TH)), 1.0 / len(TH)), 24.0)
    cm, _ = circ_wmean(TM[None, :], np.full((1, len(TM)), 1.0 / len(TM)), 12.0, offset=1.0)
    sanity['constant_predictor'] = {'hour': float(ch[0]), 'month': float(cm[0])}
    for gname in ['overall', 'day_color', 'night_ir']:
        fm = np.ones(len(TH), bool) if gname == 'overall' else (fgroup == gname)
        sanity['constant_predictor'][gname] = {
            'ToD_MAE_h': float(circ_dist(TH[fm], ch[0], 24).mean()),
            'ToY_MAE_days': float(circ_dist(TM[fm], cm[0], 12).mean() * 365.25 / 12)}
    # Per-group constants: circular mean of the true times inside each IR group, scored on that group.
    per = {}
    for gname in ['day_color', 'night_ir']:
        fm = (fgroup == gname)
        gh, _ = circ_wmean(TH[None, fm], np.full((1, int(fm.sum())), 1.0 / int(fm.sum())), 24.0)
        gm, _ = circ_wmean(TM[None, fm], np.full((1, int(fm.sum())), 1.0 / int(fm.sum())), 12.0, offset=1.0)
        per[gname] = {
            'hour': float(gh[0]), 'month': float(gm[0]),
            'ToD_MAE_h': float(circ_dist(TH[fm], gh[0], 24).mean()),
            'ToY_MAE_days': float(circ_dist(TM[fm], gm[0], 12).mean() * 365.25 / 12)}
    sanity['per_group_constant_predictor'] = per
    sanity['uniform_random_expected'] = {'ToD_MAE_h': 6.0, 'ToY_MAE_days': 365.25 / 4}
    configs = [('k1_colab', 1, 0.0, None)] + [(f'k1', 1, 0.0, 'cpu')] + \
              [(f'k{k}_t{t:g}', k, t, 'cpu') for k in ks if k > 1 for t in taus]
    rows_A, rows_B, rows_C, rows_acc, rows_sum = [], [], [], [], []
    groups = ['overall', 'day_color', 'night_ir']
    for name, k, tau, src in configs:
        if src is None:                               # Phase-2 Colab argmax (exact baseline)
            ph, pm = GH[T1c], GM[T1c]; pidx = T1c; wmax = np.ones(len(ph))
        else:
            ph, pm, _, wmax = readout(TS, TI, GH, GM, k, tau); pidx = TI[:, 0]
        P = []
        for tr, (x, y) in {'f1->f2': (1, 2), 'f2->f3': (2, 3)}.items():
            ia, ib = idx[x], idx[y]
            dH = circ_dist(ph[ia], ph[ib], 24); dM = circ_dist(pm[ia], pm[ib], 12)
            P.append(pd.DataFrame({'transition': tr, 'tlabel': trip['tlabel'].values, 'ir_group': tgroup,
                                   'dH': dH, 'dM': dM, 'd_burst': d_burst(dH, dM),
                                   'd_cos': cos_dist(Z[ia], Z[ib]), 'idx_changed': pidx[ia] != pidx[ib]}))
        P = pd.concat(P, ignore_index=True)
        if name == 'k1_colab':
            P['C_class'] = classify_jump(P['idx_changed'], P['dH'], P['d_cos'], step_h)
            P['C_class'].value_counts().rename_axis('class').reset_index(name='n').assign(
                pct=lambda d: (100 * d.n / len(P)).round(2)).to_csv(out / 'sanity_C_classes_k1_colab.csv', index=False)
        s = {'config': name, 'k': k, 'tau': tau, 'gallery_feats': 'colab_cached_top1' if src is None else 'cpu'}
        for gname in groups:
            pg = P if gname == 'overall' else P[P.ir_group == gname]
            fm = np.ones(len(frames), bool) if gname == 'overall' else (fgroup == gname)
            for tr in ['all', 'f1->f2', 'f2->f3']:
                pt = pg if tr == 'all' else pg[pg.transition == tr]
                r = {'config': name, 'k': k, 'tau': tau, 'group': gname, 'transition': tr, 'n_pairs': len(pt)}
                for th in (0.5, 1.0):
                    r[f'FCR_{th}h_%'] = 100 * float((pt['dH'] > th).mean())
                rows_A.append(r)
                if tr == 'all':
                    s[f'FCR0.5_{gname}'] = r['FCR_0.5h_%']; s[f'FCR1_{gname}'] = r['FCR_1.0h_%']
            for lab in ['animal', 'empty']:
                v = pg.loc[pg.tlabel == lab, 'd_burst'].values
                rb = {'config': name, 'k': k, 'tau': tau, 'group': gname, 'tlabel': lab, 'n': len(v),
                      'median': float(np.median(v)) if len(v) else np.nan,
                      'p95': float(np.percentile(v, 95)) if len(v) else np.nan,
                      'mean': float(v.mean()) if len(v) else np.nan}
                rows_B.append(rb); s[f'dburst_med_{lab}_{gname}'] = rb['median']; s[f'dburst_p95_{lab}_{gname}'] = rb['p95']
            ct = pd.crosstab(pd.cut(pg['d_cos'], DCOS_BINS, labels=DCOS_LBL, right=False),
                             pd.cut(pg['dH'], DH_BINS, labels=DH_LBL), normalize='index') * 100
            for dc in ct.index:
                rows_C.append({'config': name, 'k': k, 'tau': tau, 'group': gname, 'd_cos_bin': dc,
                               'n_pairs': int((pd.cut(pg['d_cos'], DCOS_BINS, labels=DCOS_LBL, right=False) == dc).sum()),
                               **{f'dH {c} %': float(ct.loc[dc, c]) if c in ct.columns else 0.0 for c in DH_LBL}})
            small = pg['d_cos'] <= 0.05
            s[f'P(dH>0.5h|dcos<=0.05)_{gname}'] = 100 * float((pg.loc[small, 'dH'] > 0.5).mean())
            s[f'spearman_dcos_dburst_{gname}'] = float(pg['d_cos'].rank().corr(pg['d_burst'].rank()))
            tod = circ_dist(ph[fm], TH[fm], 24); toy = circ_dist(pm[fm], TM[fm], 12)
            ra = {'config': name, 'k': k, 'tau': tau, 'group': gname, 'n_frames': int(fm.sum()),
                  'ToD_MAE_h': float(tod.mean()), 'ToD_median_h': float(np.median(tod)),
                  'ToD_within_1h_%': 100 * float((tod <= 1).mean()),
                  'ToY_MAE_months': float(toy.mean()), 'ToY_MAE_days': float(toy.mean() * 365.25 / 12),
                  'pred_in_gallery_gap_%': 100 * float(((ph[fm] < GH.min()) | (ph[fm] > GH.max())).mean()),
                  'mean_max_weight': float(wmax[fm].mean())}
            rows_acc.append(ra)
            s[f'ToD_MAE_{gname}'] = ra['ToD_MAE_h']; s[f'ToY_MAE_days_{gname}'] = ra['ToY_MAE_days']
            s[f'pred_in_gap_%_{gname}'] = ra['pred_in_gallery_gap_%']
        rows_sum.append(s)
        print(f"{name:14s} FCR0.5 {s['FCR0.5_overall']:6.2f} day {s['FCR0.5_day_color']:6.2f} night {s['FCR0.5_night_ir']:6.2f} | "
              f"ToD MAE {s['ToD_MAE_overall']:.3f} h | ToY MAE {s['ToY_MAE_days_overall']:.1f} d", flush=True)
    pd.DataFrame(rows_A).to_csv(out / 'A_fcr.csv', index=False)
    pd.DataFrame(rows_B).to_csv(out / 'B_dburst.csv', index=False)
    pd.DataFrame(rows_C).to_csv(out / 'C_dH_vs_dcos.csv', index=False)
    pd.DataFrame(rows_acc).to_csv(out / 'accuracy.csv', index=False)
    S = pd.DataFrame(rows_sum); S.to_csv(out / 'summary_by_config.csv', index=False)
    (out / 'sanity.json').write_text(json.dumps(sanity, indent=2))
    cfg = dict(vars(a))
    for key in ('run_dir', 'gtloc_repo', 'ckpt', 'out'):
        cfg[key] = str(cfg[key])
    cfg['cache'] = [str(x) for x in cfg['cache']]
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    # plots
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, gname in zip(axes, groups):
        for tau in taus:
            sub = S[(S.tau == tau) & (S.k > 1)]
            ax.plot(sub[f'ToD_MAE_{gname}'], sub[f'FCR0.5_{gname}'], 'o-', label=f'tau={tau:g}')
            for _, r in sub.iterrows():
                ax.annotate(f"k{int(r.k)}", (r[f'ToD_MAE_{gname}'], r[f'FCR0.5_{gname}']), fontsize=7)
        b = S[S.config == 'k1_colab'].iloc[0]
        ax.plot(b[f'ToD_MAE_{gname}'], b[f'FCR0.5_{gname}'], 'k*', ms=14, label='k=1 (Phase-2)')
        ax.set_title(gname); ax.set_xlabel('ToD MAE (h)'); ax.set_ylabel('FCR @0.5 h (%)')
    axes[0].legend(fontsize=7); fig.suptitle('Top-k readout: stability (FCR) vs accuracy (ToD MAE)')
    fig.tight_layout(); fig.savefig(out / 'fcr_vs_mae.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, gname in zip(axes, ['day_color', 'night_ir']):
        for tau in taus:
            sub = S[(S.tau == tau) | (S.k == 1) & (S.config == 'k1')].sort_values('k')
            sub = pd.concat([S[S.config == 'k1'], S[(S.tau == tau) & (S.k > 1)]])
            ax.plot(sub.k, sub[f'FCR0.5_{gname}'], 'o-', label=f'tau={tau:g}')
        ax.set_xscale('log'); ax.set_xlabel('k'); ax.set_ylabel('FCR @0.5 h (%)'); ax.set_title(gname)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / 'fcr_vs_k.png', dpi=150); plt.close(fig)
    print('wrote', out)

if __name__ == '__main__':
    main()
