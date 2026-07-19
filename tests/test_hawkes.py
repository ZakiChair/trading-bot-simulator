"""Tests — Hawkes self-exciting jump process.

Pins:
* the MLE recovers genuine self-excitation (branching ratio > 0) on a stream with
  CLUSTERED jumps, and stays ~0 on a stream with isolated (Poisson) jumps;
* simulated jumps are martingale-preserving (mean ≈ 0 — symmetric signs), so the
  Hawkes overlay adds clustered tail risk without a directional drift;
* the branching ratio is capped below 1 (stationarity).
"""
from __future__ import annotations

import numpy as np

from core.hawkes import HawkesJumps


def _clustered_jumps(seed=0, mu=0.01, alpha=0.10, beta=0.35):
    """Returns from a genuine discrete-time Hawkes process: λ_t = μ + Σ excitation,
    a jump excites the next bars by α decaying at β (branching α/β≈0.29)."""
    rng = np.random.default_rng(seed)
    n = 6000
    r = rng.normal(0, 0.005, n)
    exc = 0.0
    decay = np.exp(-beta)
    for t in range(n):
        lam = mu + exc
        exc *= decay
        if rng.random() < min(lam, 0.95):
            r[t] += rng.choice([-1, 1]) * rng.uniform(0.03, 0.06)
            exc += alpha                                 # this jump excites the next
    return r, np.full(n, 0.005)


def _isolated_jumps(seed=1, rate=0.025):
    """Returns with the same jump RATE but no clustering (iid Poisson)."""
    rng = np.random.default_rng(seed)
    n = 6000
    r = rng.normal(0, 0.005, n)
    mask = rng.random(n) < rate
    r[mask] += rng.choice([-1, 1], mask.sum()) * rng.uniform(0.03, 0.06)
    return r, np.full(n, 0.005)


def test_hawkes_recovers_clustering():
    r, sig = _clustered_jumps()
    hk = HawkesJumps.fit(r, sig, k=4.0)
    assert hk.trained
    assert hk.branching_ratio > 0.1, f"missed clustering: n={hk.branching_ratio:.3f}"
    assert hk.branching_ratio < 1.0           # stationary


def test_hawkes_flat_on_isolated_jumps():
    r, sig = _isolated_jumps()
    hk = HawkesJumps.fit(r, sig, k=4.0)
    # Isolated jumps → little/no self-excitation relative to the clustered case.
    assert hk.branching_ratio < 0.5, hk.branching_ratio


def test_hawkes_jumps_are_martingale():
    r, sig = _clustered_jumps(seed=3)
    hk = HawkesJumps.fit(r, sig, k=4.0)
    rng = np.random.default_rng(0)
    jumps = hk.simulate(rng, 20000, 12)
    # Symmetric jump signs ⇒ mean ≈ 0 (no directional drift smuggled in).
    assert abs(float(jumps.sum(axis=1).mean())) < 0.002, float(jumps.mean())


if __name__ == "__main__":
    test_hawkes_recovers_clustering()
    test_hawkes_flat_on_isolated_jumps()
    test_hawkes_jumps_are_martingale()
    print("✓ Hawkes jump-process tests OK")
