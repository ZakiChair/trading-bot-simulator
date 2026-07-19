"""Tests — conformal prediction intervals reach nominal coverage.

The whole point of the conformal layer is the coverage guarantee, so the tests
check exactly that: on a stream with KNOWN fat tails (Student-t innovations), the
conformal interval covers at its nominal rate while a Gaussian band of the same
nominal level under-covers the tails. Also pins the ACI mechanics (α_t moves to
defend the target) and the finite-sample quantile.
"""
from __future__ import annotations

import numpy as np

from core.conformal import ConformalCalibrator


def _t_stream(n=4000, nu=3.5, seed=0):
    """Unit-variance Student-t innovations scaled by a (known) constant σ."""
    rng = np.random.default_rng(seed)
    sigma = 0.01
    z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2) / nu)
    return z * sigma, sigma


def test_conformal_reaches_nominal_coverage():
    r, sigma = _t_stream()
    for level in (0.8, 0.9, 0.95):
        cal = ConformalCalibrator(target_alpha=1 - level, window=400, gamma=0.0)
        covered = 0
        n = 0
        for t in range(300, len(r)):       # warm the calibration window first
            lo, hi = cal.interval(0.0, sigma)
            covered += int(lo <= r[t] <= hi)
            n += 1
            cal.update(r[t], 0.0, sigma)
        cov = covered / n
        assert abs(cov - level) < 0.03, f"level {level}: coverage {cov:.3f}"


def test_conformal_beats_gaussian_on_fat_tails():
    """At the 99 % level the Gaussian band under-covers fat tails (a unit-variance
    t is more peaked at the centre but heavier in the extreme tail, so the gap
    shows at 99 %, not 95 %); conformal stays on nominal. Both use the SAME,
    correct σ — the only difference is the tail model."""
    import math
    from core.conformal import _erfinv

    r, sigma = _t_stream(nu=3.0, seed=2)
    cal = ConformalCalibrator(target_alpha=0.01, window=600, gamma=0.0)
    z = math.sqrt(2.0) * _erfinv(0.99)
    cc = gc = n = 0
    for t in range(400, len(r)):
        lo, hi = cal.interval(0.0, sigma)
        cc += int(lo <= r[t] <= hi)
        gc += int(abs(r[t]) <= z * sigma)
        n += 1
        cal.update(r[t], 0.0, sigma)
    conf_cov, gauss_cov = cc / n, gc / n
    assert abs(conf_cov - 0.99) < 0.02, conf_cov
    assert gauss_cov < conf_cov, (gauss_cov, conf_cov)  # Gaussian under-covers the 99% tail


def test_conformal_holds_coverage_under_regime_shift():
    """Markets are non-exchangeable: volatility shifts. Normalised conformal
    already self-corrects scale, and ACI defends coverage under residual drift —
    so even when σ is a *stale* estimate that lags a regime change, long-run
    coverage stays near nominal. We feed a stream that jumps from calm to
    turbulent vol while the calibrator is fed a fixed, stale σ."""
    rng = np.random.default_rng(4)
    calm = rng.standard_normal(1500) * 0.005
    turb = rng.standard_normal(1500) * 0.02
    r = np.concatenate([calm, turb])
    stale_sigma = 0.005                       # never updated to the turbulent vol
    cal = ConformalCalibrator(target_alpha=0.1, window=250, gamma=0.03)
    covered = n = 0
    for t in range(300, len(r)):
        lo, hi = cal.interval(0.0, stale_sigma)
        covered += int(lo <= r[t] <= hi)
        n += 1
        cal.update(r[t], 0.0, stale_sigma)
    cov = covered / n
    assert abs(cov - 0.9) < 0.04, f"coverage under regime shift {cov:.3f}"


if __name__ == "__main__":
    test_conformal_reaches_nominal_coverage()
    test_conformal_beats_gaussian_on_fat_tails()
    test_conformal_holds_coverage_under_regime_shift()
    print("✓ conformal prediction tests OK")
