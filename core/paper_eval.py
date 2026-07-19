"""Paper-mode profitability verdict — answers *"rentable ou non ?"*.

A paper run trades the bot **frozen** (deterministic risk policy, fresh book)
and nets fees + slippage. The verdict summarises whether that produced a profit
and — keeping this project's honesty ethos (see ``ANALYSE_CRITIQUE_MODELE.md``)
— whether it actually beat simply **holding** the asset over the same window.

Qualification statistique (audit 2026-07) : un unique backtest de ~300 barres
peut être vert par pure chance. Le verdict porte donc un test de Student sur les
rendements par barre de l'equity et un minimum de barres/trades — sous ces
seuils il est **suspendu** («échantillon insuffisant»), exactement le standard
Wilson que ``core/live_eval.py`` applique déjà à la direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.portfolio import Action, Portfolio

_MIN_BARS = 120        # en dessous, aucun verdict (IC plus large que tout effet)
_MIN_TRADES = 3        # un buy&hold déguisé n'est pas une stratégie évaluée
_T_SIG = 1.96          # |t| du rendement moyen par barre pour un verdict fort


@dataclass
class PaperVerdict:
    ret_pct: float        # net-of-fees total return, %
    buy_hold_pct: float   # benchmark: holding the asset over the same window, %
    n_trades: int
    n_round: int          # completed round-trips (SELL count)
    win_rate: float
    turnover: float       # trades / bars
    bars: int
    forward: bool         # live forward paper-trade (vs cached out-of-sample backtest)
    final: bool           # run complete (cached) vs still accruing (live)
    t_stat: float = 0.0   # Student t du rendement moyen par barre (H0: 0)

    @property
    def profitable(self) -> bool:
        return self.ret_pct > 0.0

    @property
    def beats_hold(self) -> bool:
        return self.ret_pct > self.buy_hold_pct

    @property
    def qualified(self) -> bool:
        """Assez de données pour qu'un verdict veuille dire quelque chose."""
        return self.bars >= _MIN_BARS and self.n_trades >= _MIN_TRADES

    @property
    def significant(self) -> bool:
        """Le rendement moyen par barre est-il statistiquement ≠ 0 ?"""
        return self.qualified and abs(self.t_stat) >= _T_SIG

    @property
    def verdict(self) -> str:
        if not self.qualified:
            return f"⏳ échantillon insuffisant ({self.bars} barres, {self.n_trades} trades)"
        if not self.profitable:
            return "🔴 PAS RENTABLE"
        if not self.beats_hold:
            return "🟠 rentable mais sous le buy & hold"
        return "✅ RENTABLE (et bat le buy & hold)"

    @property
    def significance_note(self) -> str:
        if not self.qualified:
            return ""
        if self.significant:
            return f"t={self.t_stat:+.1f} (significatif)"
        return f"t={self.t_stat:+.1f} — non significatif : compatible avec la chance"

    @property
    def sig_tag(self) -> str:
        """Qualificatif compact pour le bandeau/les panneaux — le verdict ne
        doit JAMAIS s'afficher sans sa significativité (un « ✅ RENTABLE » à
        t=+0.8 lu sans le t se prend pour une preuve)."""
        if not self.qualified:
            return ""
        return f"t={self.t_stat:+.1f}{'' if self.significant else ' n.s.'}"


def _t_stat(equity_curve) -> float:
    """t de Student du rendement log moyen par barre de la courbe d'equity."""
    if equity_curve is None or len(equity_curve) < 8:
        return 0.0
    import numpy as np

    eq = np.asarray(equity_curve, dtype=float)
    eq = eq[eq > 0]
    if len(eq) < 8:
        return 0.0
    r = np.diff(np.log(eq))
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(r) / (sd / math.sqrt(len(r))))


def build_verdict(
    portfolio: Portfolio,
    price: float,
    *,
    start_price: float,
    bars: int,
    forward: bool,
    final: bool,
    equity_curve=None,
) -> PaperVerdict:
    sells = sum(1 for t in portfolio.trades if t.action == Action.SELL)
    ret_pct = portfolio.total_pnl(price) / portfolio.initial_cash * 100.0
    bh = ((price / start_price) - 1.0) * 100.0 if start_price else 0.0
    return PaperVerdict(
        ret_pct=ret_pct,
        buy_hold_pct=bh,
        n_trades=len(portfolio.trades),
        n_round=sells,
        win_rate=portfolio.win_rate(),
        turnover=len(portfolio.trades) / max(bars, 1),
        bars=bars,
        forward=forward,
        final=final,
        t_stat=_t_stat(equity_curve),
    )


def verdict_lines(v: PaperVerdict) -> list[str]:
    """Journal lines for a (running or final) paper verdict."""
    head = "Verdict" if v.final else "En cours"
    kind = "paper-trading live" if v.forward else "backtest hors-échantillon"
    sig = f" · {v.significance_note}" if v.significance_note else ""
    return [
        f"{'■' if v.final else '·'} {head} {kind} — {v.verdict}",
        (
            f"   net {v.ret_pct:+.2f}% · buy&hold {v.buy_hold_pct:+.2f}% · "
            f"{v.n_round} ventes · réussite {v.win_rate:.0%} · "
            f"rotation {v.turnover:.2f}/barre ({v.bars} barres){sig}"
        ),
    ]
