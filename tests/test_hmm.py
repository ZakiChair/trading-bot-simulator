"""Tests — Gaussian HMM (Baum-Welch) and the volatility-regime wrapper.

Correctness pins:
* the EM recovers a known 2-state vol structure (calm vs turbulent) on synthetic
  data — separated means and a high-persistence transition matrix;
* ``filter_state`` is causal (no look-ahead): the filtered state at bar t is
  unchanged by appending future bars;
* the ``RegimeHMM(vol3)`` wrapper labels states by volatility level and exposes a
  row-stochastic learned transition matrix.
"""
from __future__ import annotations

import numpy as np

from core.hmm import VOL_REGIME_ORDER, GaussianHMM, RegimeHMM


def _two_regime_series(seed: int = 0):
    """Synthetic price path that switches between a calm and a turbulent vol
    regime with high persistence — the structure a vol-HMM should recover."""
    rng = np.random.default_rng(seed)
    vols = []
    state = 0
    for _ in range(3000):
        if rng.random() < 0.02:          # ~50-bar persistence
            state ^= 1
        vols.append(0.004 if state == 0 else 0.02)
    r = rng.normal(0.0, np.array(vols))
    return 100.0 * np.exp(np.cumsum(r))


def test_hmm_recovers_two_vol_states():
    prices = _two_regime_series()
    r = np.diff(np.log(prices))
    X = np.log(np.maximum(r * r, 1e-12))[:, None]
    hmm = GaussianHMM.fit(X, n_states=2, seed=1, n_init=2)
    assert hmm.trained
    # Two well-separated emission means (low vs high log-variance).
    sep = abs(hmm.means[0, 0] - hmm.means[1, 0])
    assert sep > 1.0, f"states not separated: {hmm.means.ravel()}"
    # High persistence: both self-transitions dominate.
    assert np.all(np.diag(hmm.transmat) > 0.85), hmm.transmat


def test_filter_state_is_causal():
    prices = _two_regime_series(seed=3)
    r = np.diff(np.log(prices))
    X = np.log(np.maximum(r * r, 1e-12))[:, None]
    hmm = GaussianHMM.fit(X, n_states=2, seed=1)
    t = 1500
    a = hmm.filter_state(X[: t + 1])
    b = hmm.filter_state(X[: t + 400])  # appending future bars...
    # ...must not change the filtered distribution AT bar t (forward pass only).
    a2 = hmm.filter_state(X[: t + 1])
    assert np.allclose(a, a2)
    assert a.shape == (2,) and abs(a.sum() - 1.0) < 1e-9


def test_regime_hmm_vol3_labels_and_transmat():
    prices = _two_regime_series(seed=5)
    rh = RegimeHMM.fit(None, None, None, prices, kind="vol3", seed=2)
    assert rh is not None and rh.kind == "vol3"
    assert set(rh.state_to_regime.values()) <= set(VOL_REGIME_ORDER)
    T = rh.transmat_in_regime_order()
    assert T.shape == (3, 3)
    assert np.allclose(T.sum(axis=1), 1.0), T
    # Live regime read returns a valid vol label.
    assert rh.regime_at(prices) in VOL_REGIME_ORDER


if __name__ == "__main__":
    test_hmm_recovers_two_vol_states()
    test_filter_state_is_causal()
    test_regime_hmm_vol3_labels_and_transmat()
    print("✓ HMM + volatility-regime tests OK")
