"""Self-exciting Hawkes jump process — temporal clustering of shocks.

The Student-t cone (``core/cone.py``) gives each bar fat tails, but the draws are
**iid across bars**: it knows a −8 % hour is possible, not that one −8 % hour
makes the *next* hour more dangerous. Crypto shocks cluster in time — a
liquidation cascade begets another — which is exactly what a **Hawkes process**
(Hawkes 1971; Bacry-Mastromatteo-Muzy 2015 for finance) models: a point process
whose intensity is excited by its own past events,

    λ(t) = μ + Σ_{tᵢ < t} α·exp(−β(t − tᵢ)),        branching ratio n = α/β < 1.

Each past jump bumps the intensity by α and the bump decays at rate β; n = α/β is
the expected number of "offspring" jumps per jump (the self-excitation strength;
n<1 for stationarity).

Pipeline:
* **detect** jumps as bars whose return exceeds a vol-scaled threshold
  ``|r_t| > k·σ̂_t`` (σ̂ from the rough-vol head, so "jump" means *large relative
  to the current regime*, not an absolute move);
* **fit** (μ, α, β) by maximum likelihood (Ogata's recursive log-likelihood,
  numpy grid + refine, no scipy);
* **simulate** clustered jumps over the cone horizon (discrete-time intensity,
  self-exciting through the path's own simulated jumps) and add them to the
  diffusion. Jump signs are **symmetric** (mean-zero) so the cone stays a
  martingale — Hawkes adds *clustered tail risk*, never a directional drift.

Whether this earns its place ON TOP of the Student-t rough cone (which already
fattens single-bar tails) is decided by the ladder A/B in
``tests/measure_model.py`` — clustering only helps where iid fat tails fall short.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_BR_MAX = 0.9   # cap the branching ratio for stationarity (α/β ≤ 0.9)


@dataclass
class HawkesJumps:
    """Fitted exponential-kernel Hawkes process for vol-scaled return jumps."""

    mu: float = 0.02            # baseline jump intensity (jumps per bar)
    alpha: float = 0.0          # excitation jump
    beta: float = 1.0           # decay rate
    jump_scale: float = 0.0     # std of |jump| size in log-return units (≥ threshold)
    jump_scale_z: float = 0.0   # std of jump size in σ̂ units (regime-relative)
    threshold_k: float = 4.0    # detection threshold in σ̂ units
    trained: bool = False

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else 0.0

    # -- detection ------------------------------------------------------- #
    @staticmethod
    def detect_jumps(returns: np.ndarray, sigmas: np.ndarray, k: float = 4.0):
        """Indices and signed sizes of bars exceeding ``k·σ̂``. ``sigmas[t]`` is
        the (causal) vol forecast for bar t, so detection is regime-relative."""
        r = np.asarray(returns, dtype=np.float64)
        s = np.maximum(np.asarray(sigmas, dtype=np.float64), 1e-9)
        m = min(len(r), len(s))
        r, s = r[:m], s[:m]
        z = r / s
        idx = np.where(np.abs(z) > k)[0]
        return idx, r[idx]

    # -- fit ------------------------------------------------------------- #
    @classmethod
    def fit(cls, returns: np.ndarray, sigmas: np.ndarray, *, k: float = 4.0) -> "HawkesJumps":
        r = np.asarray(returns, dtype=np.float64)
        T = float(len(r))
        idx, sizes = cls.detect_jumps(r, sigmas, k)
        if len(idx) < 8 or T < 100:
            # Too few jumps to identify excitation → Poisson baseline only.
            rate = max(len(idx) / max(T, 1.0), 1e-4)
            js = float(np.std(sizes)) if len(sizes) > 1 else 0.0
            return cls(mu=rate, alpha=0.0, beta=1.0, jump_scale=js,
                       threshold_k=k, trained=len(idx) >= 1)
        t = idx.astype(np.float64)

        def neg_ll(mu, alpha, beta):
            # Ogata recursive log-likelihood for an exponential Hawkes process.
            if mu <= 0 or alpha < 0 or beta <= 0 or alpha / beta >= 1.0:
                return np.inf
            A = 0.0
            ll = 0.0
            prev = t[0]
            ll += math.log(mu)                       # first event, A=0
            for i in range(1, len(t)):
                A = math.exp(-beta * (t[i] - prev)) * (1.0 + A)
                lam = mu + alpha * A
                if lam <= 0:
                    return np.inf
                ll += math.log(lam)
                prev = t[i]
            # compensator ∫λ = μT + (α/β) Σ_i (1 − exp(−β(T−t_i)))
            comp = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
            return -(ll - comp)

        base_rate = len(t) / T
        best = (np.inf, base_rate, 0.0, 1.0)
        for beta in np.linspace(0.05, 2.0, 14):
            for br in np.linspace(0.0, _BR_MAX, 12):     # branching ratio = α/β
                alpha = br * beta
                mu = max(base_rate * (1.0 - br), 1e-4)   # moment-ish start for μ
                v = neg_ll(mu, alpha, beta)
                if v < best[0]:
                    best = (v, mu, alpha, beta)
        # Joint (β, br) refinement around the coarse winner — the coarse grid
        # (step 0.15 in β) systematically under-estimated the decay rate (β low
        # ⇒ excitation persists too long, branching ratio overstated).
        _, mu0, alpha0, beta0 = best
        br0 = alpha0 / beta0 if beta0 > 0 else 0.0
        for beta in np.linspace(max(0.02, beta0 * 0.5), min(3.0, beta0 * 1.8), 11):
            for br in np.linspace(max(0.0, br0 - 0.15), min(_BR_MAX, br0 + 0.15), 9):
                v = neg_ll(mu0, br * beta, beta)
                if v < best[0]:
                    best = (v, mu0, br * beta, beta)
        # refine μ around the best on a small grid (α,β fixed)
        _, mu0, alpha, beta = best
        for mu in np.linspace(max(1e-4, mu0 * 0.4), mu0 * 1.6, 9):
            v = neg_ll(mu, alpha, beta)
            if v < best[0]:
                best = (v, mu, alpha, beta)
        _, mu, alpha, beta = best
        js = float(np.std(sizes)) if len(sizes) > 1 else float(np.mean(np.abs(sizes)))
        # Taille de saut aussi en unités de σ̂ (z) : la détection est relative au
        # régime, la taille simulée doit l'être aussi — sinon une magnitude de
        # crise 2021 atterrit telle quelle dans un régime calme.
        s = np.maximum(np.asarray(sigmas, dtype=np.float64), 1e-9)
        z_sizes = sizes / s[idx] if len(sizes) else np.array([])
        jz = float(np.std(z_sizes)) if len(z_sizes) > 1 else float(np.mean(np.abs(z_sizes))) if len(z_sizes) else 0.0
        return cls(mu=float(mu), alpha=float(alpha), beta=float(beta),
                   jump_scale=js, jump_scale_z=jz, threshold_k=k, trained=True)

    # -- recent excitation state ----------------------------------------- #
    def excitation_from(self, returns: np.ndarray, sigmas: np.ndarray) -> float:
        """Current excitation Σ α·exp(−β·age) from jumps in the recent history —
        the live 'how charged is the powder keg' term added to μ for bar t+1."""
        if not self.trained or self.alpha <= 0:
            return 0.0
        idx, _ = self.detect_jumps(returns, sigmas, self.threshold_k)
        if len(idx) == 0:
            return 0.0
        T = len(np.asarray(returns))
        ages = (T - idx).astype(np.float64)
        return float(self.alpha * np.sum(np.exp(-self.beta * ages)))

    # -- simulation into the cone ---------------------------------------- #
    def simulate(self, rng: np.random.Generator, n: int, horizon: int,
                 base_excitation: float = 0.0,
                 sd_path: np.ndarray | None = None) -> np.ndarray:
        """Clustered jump increments ``(n, horizon)`` in log-return units.

        Discrete-time self-exciting recursion: bar k's intensity is
        ``λ_k = μ + excitation_k`` where the excitation decays each bar and is
        bumped by α whenever a jump fires (self-excitation through the path's own
        simulated jumps), seeded with ``base_excitation`` from recent real jumps.
        Jump counts ~ Poisson(λ_k); each jump gets a **symmetric** size (mean
        zero → martingale-preserving). ``sd_path`` (per-bar σ̂ forecast) scales
        jump sizes to the CURRENT vol regime via ``jump_scale_z`` — detection was
        σ̂-relative, so simulation must be too; absolute ``jump_scale`` is the
        fallback when no σ̂ path is supplied."""
        if not self.trained or (self.jump_scale <= 0 and self.jump_scale_z <= 0):
            return np.zeros((n, horizon), dtype=np.float64)
        out = np.zeros((n, horizon), dtype=np.float64)
        exc = np.full(n, base_excitation, dtype=np.float64)
        decay = math.exp(-self.beta)
        use_rel = sd_path is not None and self.jump_scale_z > 0 and len(sd_path) >= horizon
        for kk in range(horizon):
            lam = np.maximum(self.mu + exc, 0.0)
            counts = rng.poisson(lam)                       # jumps this bar, per path
            scale = self.jump_scale_z * float(sd_path[kk]) if use_rel else self.jump_scale
            # symmetric jump sizes: N(0, scale²)·count (sum of count jumps)
            sizes = rng.normal(0.0, scale, n) * np.sqrt(np.maximum(counts, 0))
            sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
            out[:, kk] = sizes * sign
            exc = exc * decay + self.alpha * counts          # self-excite + decay
        return out
