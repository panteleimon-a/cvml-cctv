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


def test_label_rule_union_and_order_invariance():
    g = pure()
    cats = [{'id': 30, 'name': 'empty'}, {'id': 1, 'name': 'opossum'}]
    imgs = [{'id': i, 'file_name': f'{i}.jpg', 'datetime': '2012-01-01 00:00:00'} for i in ('a', 'b', 'c', 'd')]
    a1 = {'images': imgs[:2], 'categories': cats,
          'annotations': [{'image_id': 'a', 'category_id': 30}, {'image_id': 'b', 'category_id': 30}]}
    # second split repeats image b with an animal annotation, and adds an image with no annotation
    a2 = {'images': imgs[1:], 'categories': cats,
          'annotations': [{'image_id': 'b', 'category_id': 1}, {'image_id': 'c', 'category_id': 30},
                          {'image_id': 'c', 'category_id': 30}]}
    results = []
    for order in (['s1', 's2'], ['s2', 's1']):
        src = {'s1': a1, 's2': a2}
        im, an, cat, _ = g['load_cct_images']({k: src[k] for k in order})
        lab = dict(zip(im['id'], g['image_labels'](im, an, cat)))
        results.append(lab)
        diag = g['label_diagnostics'](an, cat)
        assert diag['mixed_empty_and_animal'] == 1
    assert results[0] == results[1] == {'a': 'empty', 'b': 'animal', 'c': 'empty', 'd': 'unlabeled'}
    assert 'ALL' in g['LABEL_RULE']


def test_conflicting_category_ids_raise():
    g = pure()
    d1 = {'images': [], 'annotations': [], 'categories': [{'id': 30, 'name': 'empty'}]}
    d2 = {'images': [], 'annotations': [], 'categories': [{'id': 30, 'name': 'deer'}]}
    try:
        g['load_cct_images']({'a': d1, 'b': d2})
    except ValueError:
        return
    raise AssertionError('conflicting category ids were accepted')


def test_missing_frame_guard_and_empty_frames():
    g = pure()
    assert g['check_missing_frames'](0, 47928, []) == 0
    assert g['check_missing_frames'](479, 47928, ['x.jpg']) < 0.01
    for n_missing, n_total in [(47928, 47928), (480, 47928), (0, 0)]:
        try:
            g['check_missing_frames'](n_missing, n_total, ['x.jpg'], 0.01)
        except RuntimeError as exc:
            assert 'image file' in str(exc) or 'No triplet frames' in str(exc)
        else:
            raise AssertionError((n_missing, n_total))
    try:
        g['require_frames'](pd.DataFrame())
    except RuntimeError:
        pass
    else:
        raise AssertionError('empty frames accepted')


def test_list_dir_retry():
    g = pure()
    d = Path(tempfile.mkdtemp())
    (d / 'a.jpg').write_bytes(b'x')
    assert set(g['list_dir_retry'](d)) == {'a.jpg'}
    try:
        g['list_dir_retry'](d / 'missing', retries=2, backoff=0, sleep=lambda s: None)
    except RuntimeError:
        pass
    else:
        raise AssertionError('missing folder listed')


def test_real_eccv18_annotations_if_available():
    """Optional regression on the official JSONs: set CCT20_ANN_DIR to eccv_18_annotation_files/."""
    import json
    import os
    folder = os.environ.get('CCT20_ANN_DIR')
    if not folder:
        print('skip: CCT20_ANN_DIR not set')
        return
    g = pure()
    splits = ['train', 'cis_val', 'cis_test', 'trans_val', 'trans_test']
    ann = {s: json.loads(Path(folder, f'{s}_annotations.json').read_text()) for s in splits}
    im, an, cat, n_raw = g['load_cct_images'](ann)
    assert n_raw == len(im) == 57864
    im['datetime'] = g['parse_datetimes'](im['datetime'])
    im['label'] = g['image_labels'](im, an, cat)
    assert im['label'].value_counts().to_dict() == {'animal': 53850, 'empty': 4014}
    t, st = g['build_triplets'](im, 'eq3')
    assert (st['triplets'], st['frames'], st['locations']) == (15976, 47928, 18)
    assert g['build_triplets'](im, 'ge3')[1]['triplets'] == 16122


def test_find_resume_dirs():
    g = {'Path': Path}
    exec(cell('===== 3. PURE'), g)
    root = Path(tempfile.mkdtemp())
    f = lambda **k: [d.name for d in g['find_resume_dirs'](root, 'cct20_phase2', **k)]
    assert g['find_resume_dirs'](root / 'nope', 'cct20_phase2') == []

    def mk(name, shard=False, cache=False):
        d = root / name
        (d / 'cache_shards').mkdir(parents=True)
        if shard:
            (d / 'cache_shards' / 'shard_00000.npz').write_bytes(b'x')
        if cache:
            (d / 'frame_cache.npz').write_bytes(b'x')
        return d

    mk('cct20_phase2_20261118_0900', cache=True)            # older than the finished cache: not needed
    mk('cct20_phase2_20261119_0900', shard=True, cache=True)
    mk('cct20_phase2_20261120_1351', shard=True)
    mk('cct20_phase2_20261120_1700')                        # empty: died before inference
    cur = mk('cct20_phase2_20261121_1000', shard=True)
    mk('other_20261122_1000', shard=True)
    assert f() == ['cct20_phase2_20261121_1000', 'cct20_phase2_20261120_1351', 'cct20_phase2_20261119_0900']
    assert f(exclude=cur) == ['cct20_phase2_20261120_1351', 'cct20_phase2_20261119_0900']


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
             subprocess=subprocess, sys=sys, Path=Path, heartbeat=lambda *a, **k: None)
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
