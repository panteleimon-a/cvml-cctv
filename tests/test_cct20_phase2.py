"""Offline tests for scripts/cct20_phase2_experiments.py (no GPU, no Drive).

Run: python tests/test_cct20_phase2.py   (or: pytest tests/)
"""
import datetime as dt
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'cct20_phase2_experiments.py'
CELLS = re.split(r'^# %%.*$', SCRIPT.read_text(), flags=re.M)


def cell(tag):
    return next(c for c in CELLS if tag in c)


def pure():
    g = {}
    exec(cell('===== 3. PURE'), g)
    return g


def _synthetic_coco():
    images, anns, iid = [], [], 0

    def add(seq, loc, n, t0, gaps, label):
        nonlocal iid
        t = t0
        for k in range(n):
            if k:
                t = t + dt.timedelta(seconds=gaps[k - 1])
            images.append(dict(id=f'im{iid}', file_name=f'im{iid}.jpg', location=loc,
                               date_captured=t.strftime('%Y-%m-%d %H:%M:%S'),
                               seq_id=seq, frame_num=1 + k, seq_num_frames=n))
            anns.append(dict(id=f'a{iid}', image_id=f'im{iid}', category_id=30 if label == 'empty' else 1))
            iid += 1

    t0 = dt.datetime(2012, 5, 1, 1, 42, 37)
    add('s_ok3', '10', 3, t0, [1, 1], 'animal')
    add('s_ok5', '10', 5, t0, [1, 1, 1, 1], 'empty')
    add('s_gap2', '11', 3, t0, [2, 1], 'animal')
    add('s_two', '11', 2, t0, [1], 'animal')
    add('s_emp', '12', 3, t0, [1, 1], 'empty')
    cats = [{'id': 30, 'name': 'empty'}, {'id': 1, 'name': 'opossum'}]
    # overlapping splits -> must be de-duplicated by image id
    return {'train': {'images': images[:8], 'annotations': anns, 'categories': cats},
            'cis_val': {'images': images[5:], 'annotations': anns[5:], 'categories': cats[:1]}}, len(images)


def test_distances():
    g = pure()
    assert np.isclose(g['circ_dist'](23.5, 0.5, 24), 1.0)
    assert np.isclose(g['circ_dist'](12.9, 1.1, 12), 0.2)
    assert np.isclose(g['d_burst'](12, 6), np.sqrt(2))
    assert np.isclose(g['cos_dist']([1, 0], [0, 1]), 1.0)


def test_triplets_and_labels():
    g = pure()
    ann, n_unique = _synthetic_coco()
    im, an, cat, n_raw = g['load_cct_images'](ann)
    assert n_raw > len(im) == n_unique
    im['datetime'] = g['parse_datetimes'](im['datetime'])
    im['label'] = g['image_labels'](im, an, cat)
    t_ge, _ = g['build_triplets'](im, 'ge3')
    t_eq, _ = g['build_triplets'](im, 'eq3')
    assert set(t_ge.seq_id) == {'s_ok3', 's_ok5', 's_emp'}
    assert set(t_eq.seq_id) == {'s_ok3', 's_emp'}
    labels = dict(zip(t_ge.seq_id, g['triplet_label'](t_ge)))
    assert labels == {'s_ok3': 'animal', 's_ok5': 'empty', 's_emp': 'empty'}
    im0 = im.copy()
    im0['frame_num'] = im0['frame_num'] - 1       # zero-based numbering
    assert len(g['build_triplets'](im0, 'ge3')[0]) == 3


def test_classifiers_and_rates():
    g = pure()
    cls = g['classify_jump']([1, 1, 1, 0, 1], [0.5, 2.0, 0.5, 0, 0.01], [5e-5, 0.2, 0.01, 0, 5e-5], step_h=0.526)
    assert list(cls) == ['gallery_hop', 'feature_shift', 'other', 'no_jump', 'other']
    pairs = pd.DataFrame({'dH': [0.1, 0.6, 1.2, 0.0], 'transition': ['f1->f2', 'f1->f2', 'f2->f3', 'f2->f3'],
                          'location': ['a', 'b', 'a', 'b']})
    t = g['fcr_table'](pairs, [0.5, 1.0])
    assert t['FCR_0.5h_%'].iloc[0] == 50.0 and t['FCR_1.0h_%'].iloc[0] == 25.0
    assert len(g['fcr_table'](pairs, [0.5], by=['location', 'transition'])) == 4
    assert list(g['ir_group'](np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1]]))) == ['night_ir', 'day_color', 'mixed']


def test_ir_score():
    from PIL import Image
    g = pure()
    rng = np.random.default_rng(0)
    gray = Image.fromarray(rng.integers(0, 255, (100, 120), dtype=np.uint8)).convert('RGB')
    color = Image.fromarray(rng.integers(0, 255, (100, 120, 3), dtype=np.uint8))
    assert g['ir_score_from_pil'](gray) < 0.5 < g['ir_score_from_pil'](color)


def test_postprocessing_cells_end_to_end():
    """Cells 8-12 (pairs, A, C, B, summary) on fake model outputs and a synthetic gallery."""
    import matplotlib
    matplotlib.use('Agg')
    g = {}
    exec(cell('===== 1. CONFIG'), g)
    exec(cell('===== 3. PURE'), g)
    tmp = Path(tempfile.mkdtemp())

    def out(name):
        p = tmp / name
        if p.exists():
            raise FileExistsError(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    rng = np.random.default_rng(1)
    gal = np.stack([rng.integers(1, 13, 2000), rng.integers(1, 29, 2000), rng.integers(5, 23, 2000),
                    rng.integers(0, 60, 2000), rng.integers(0, 60, 2000)], 1)
    nt = 300
    trip = pd.DataFrame({'seq_id': [f's{i}' for i in range(nt)],
                         'location': rng.choice([str(x) for x in range(18)], nt),
                         'tlabel': rng.choice(['empty', 'animal'], nt)})
    for j in (1, 2, 3):
        trip[f'id{j}'] = [f'i{i}_{j}' for i in range(nt)]
    frames = pd.DataFrame({'image_id': [trip[f'id{j}'][i] for j in (1, 2, 3) for i in range(nt)]})
    frames['true_hour'] = rng.uniform(0, 24, len(frames))
    frames['is_ir'] = rng.random(len(frames)) < 0.4
    n = len(frames)
    base = rng.normal(size=(nt, 512))
    cz = np.concatenate([base + rng.normal(scale=s, size=base.shape) for s in (0, 1e-3, 0.3)]).astype(np.float32)
    c1 = rng.integers(0, len(gal), n)
    s1 = rng.random(n)
    g.update(out=out, triplets=trip, frames=frames, order=frames.image_id.tolist(), CZ=cz, C1=c1,
             C2=rng.integers(0, len(gal), n), S1=s1, S2=s1 - 0.01, IRS=np.where(frames.is_ir, 0.5, 20.0),
             G_HOUR=g['frac_hour'](gal[:, 2], gal[:, 3], gal[:, 4]), G_MONTH=g['frac_month'](gal[:, 0], gal[:, 1]),
             f12_td=None, HOP_STEP=0.526, tstats={}, pick='ge3', gal_res={}, REPO_COMMIT='x', CKPT_SHA256='y',
             subprocess=subprocess, sys=sys, Path=Path)
    for tag in ['===== 8. PAIR', '===== 9. EXPERIMENT A', '===== 10. EXPERIMENT C',
                '===== 11. EXPERIMENT B', '===== 12. SUMMARY']:
        exec(cell(tag), g)
    names = {p.name for p in tmp.iterdir()}
    for f in ['pairs.csv', 'A_fcr_per_camera.csv', 'C_dcos_hist.png', 'B_boxplot_ir.png', 'summary.json']:
        assert f in names, f
    assert len(pd.read_csv(tmp / 'pairs.csv')) == 2 * nt


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
