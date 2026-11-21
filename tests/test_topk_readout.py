"""Light tests for scripts/topk_readout.py (no GPU, no gallery checkpoint).

Checks that k=1 equals argmax and that softmax / uniform weights sum to 1.
"""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'topk_readout.py'


def load_mod():
    spec = importlib.util.spec_from_file_location('topk_readout', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scores():
    # 4 frames, gallery of 6 stamps. Hours and months chosen so the argmax is obvious.
    rng = np.random.default_rng(0)
    TI = np.array([[2, 0, 5, 1],
                   [1, 3, 0, 4],
                   [0, 1, 2, 3],
                   [5, 4, 3, 2]], dtype=np.int32)
    TS = np.array([[0.020, 0.019, 0.018, 0.010],
                   [0.008, 0.007, 0.006, 0.001],
                   [0.050, 0.049, 0.048, 0.040],
                   [0.003, 0.002, 0.002, 0.001]], dtype=np.float32)
    GH = np.array([1.0, 4.0, 9.5, 12.0, 18.0, 22.0])
    GM = np.array([1.0, 2.5, 4.0, 6.0, 8.0, 11.0])
    return TS, TI, GH, GM, rng


def test_k1_equals_argmax():
    m = load_mod()
    TS, TI, GH, GM, _ = _scores()
    h, mo, Rh, wmax = m.readout(TS, TI, GH, GM, k=1, tau=1e-4)
    assert h.shape == (len(TS),)
    np.testing.assert_allclose(h, GH[TI[:, 0]])
    np.testing.assert_allclose(mo, GM[TI[:, 0]])
    np.testing.assert_allclose(wmax, 1.0)
    # tau must be ignored at k=1
    h2, mo2, _, _ = m.readout(TS, TI, GH, GM, k=1, tau=np.inf)
    np.testing.assert_allclose(h, h2)
    np.testing.assert_allclose(mo, mo2)


def test_weights_sum_to_one():
    m = load_mod()
    TS, TI, GH, GM, _ = _scores()
    for k, tau in [(2, 1e-4), (4, 1e-4), (4, 1e-6), (3, np.inf), (1, 1e-4), (1, np.inf)]:
        s = TS[:, :k].astype(np.float64)
        if k == 1:
            w = np.ones_like(s)
        elif np.isinf(tau):
            w = np.full_like(s, 1.0 / k)
        else:
            e = np.exp((s - s[:, :1]) / tau)
            w = e / e.sum(1, keepdims=True)
        np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-12)
        h, mo, _, wmax = m.readout(TS, TI, GH, GM, k=k, tau=tau)
        assert np.all(np.isfinite(h)) and np.all(np.isfinite(mo))
        np.testing.assert_allclose(wmax, w.max(1))


def test_no_hardcoded_workspace_paths():
    text = SCRIPT.read_text()
    assert '/workspace/' not in text
    assert '/home/box' not in text
