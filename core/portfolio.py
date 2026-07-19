"""Portfolio and trade execution — simulation and exchange-backed.

Refonte v2 : le portefeuille exécute des **rebalancements fractionnaires** vers
une exposition cible (0..1), avec frais ET slippage — l'unité de décision de la
politique de risque (``core/decision.py``). L'ancien mode binaire tout-ou-rien
(BUY 95 % / SELL 100 %) reste disponible via :meth:`execute` (compatibilité
tests + fermeture de fin de backtest), implémenté sur la même machinerie.

Honnêteté d'exécution :
* **Frais** : ``fee_rate`` par côté (10 bps par défaut, l'ordre de grandeur
  Binance spot avec BNB) ;
* **Slippage** : ``slip_base + slip_vol_k·σ̂`` par côté — l'exécution au prix
  exact de la barre était optimiste (ANALYSE_CRITIQUE §4.3). Le slippage croît
  avec la volatilité courante, comme le vrai coût de traversée du spread.
* **Equity = mark-to-market** partout (l'ancienne propriété « book value » à
  l'entry price a été retirée : elle masquait le PnL latent).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exchange.live_client import ExchangeClient


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Trade:
    step: int
    action: Action
    price: float          # prix d'exécution effectif (slippage inclus)
    size: float
    pnl: float = 0.0      # PnL réalisé (jambes SELL uniquement)
    pnl_pct: float = 0.0  # rendement réalisé de la fraction vendue (scale-free)
    fee: float = 0.0      # frais payés sur cette jambe ($)
    slippage: float = 0.0 # coût de slippage de cette jambe ($)
    exposure_after: float = 0.0  # exposition (0..1) après la jambe
    scenario_id: int | None = None
    confidence: float = 0.0
    reason: str = ""
    exchange_order_id: str = ""


@dataclass
class Portfolio:
    initial_cash: float = 10_000.0
    cash: float = 10_000.0
    position: float = 0.0
    entry_price: float = 0.0     # prix d'entrée moyen (coût moyen pondéré)
    fee_rate: float = 0.001      # frais par côté
    slip_base: float = 0.0001    # slippage plancher par côté (1 bp)
    slip_vol_k: float = 0.05     # composante ∝ σ̂ per-bar du slippage
    fees_paid: float = 0.0       # frais cumulés ($) — visibles dans l'UI
    slippage_paid: float = 0.0   # slippage cumulé ($)
    trades: list[Trade] = field(default_factory=list)
    exchange: ExchangeClient | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Valorisation                                                       #
    # ------------------------------------------------------------------ #
    def mark_to_market(self, price: float) -> float:
        if self.position:
            return self.cash + self.position * price
        return self.cash

    def total_pnl(self, price: float) -> float:
        return self.mark_to_market(price) - self.initial_cash

    def exposure(self, price: float) -> float:
        """Part de l'equity investie (0..1) au prix courant."""
        eq = self.mark_to_market(price)
        if eq <= 0:
            return 0.0
        return float(self.position * price / eq)

    def slippage_frac(self, sigma_bar: float = 0.0) -> float:
        """Slippage par côté (fraction du prix) pour la vol courante."""
        return self.slip_base + self.slip_vol_k * max(float(sigma_bar), 0.0)

    def cost_per_side(self, sigma_bar: float = 0.0) -> float:
        """Coût total par côté (frais + slippage), fraction du notionnel."""
        return self.fee_rate + self.slippage_frac(sigma_bar)

    # ------------------------------------------------------------------ #
    # Rebalancement fractionnaire (le chemin de la politique de risque)  #
    # ------------------------------------------------------------------ #
    def rebalance(
        self,
        target_exposure: float,
        price: float,
        step: int,
        *,
        sigma_bar: float = 0.0,
        reason: str = "",
        symbol: str = "BTC/USDT",
        min_delta_value: float = 1.0,
    ) -> Trade | None:
        """Ajuste la position vers ``target_exposure`` × equity (long-only).

        Achète/vend la différence avec frais + slippage. Renvoie la jambe
        exécutée (ou ``None`` si l'écart est en dessous de ``min_delta_value``).
        """
        target_exposure = min(max(float(target_exposure), 0.0), 1.0)
        equity = self.mark_to_market(price)
        if equity <= 0 or price <= 0:
            return None
        target_value = target_exposure * equity
        current_value = self.position * price
        delta_value = target_value - current_value
        if abs(delta_value) < min_delta_value:
            return None
        slip = self.slippage_frac(sigma_bar)
        if delta_value > 0:
            return self._buy_value(delta_value, price, slip, step, reason, symbol)
        return self._sell_qty(min(-delta_value / price, self.position),
                              price, slip, step, reason, symbol)

    def _buy_value(self, value: float, price: float, slip: float,
                   step: int, reason: str, symbol: str) -> Trade | None:
        exec_price = price * (1.0 + slip)
        # Dépense bornée par le cash : qty·p_exec·(1+fee) ≤ min(value, cash)
        budget = min(value, self.cash)
        if budget <= 0:
            return None
        qty = budget / (exec_price * (1.0 + self.fee_rate))
        if qty <= 0:
            return None
        order_id = ""
        if self.exchange is not None:
            order = self.exchange.place_market_order(symbol, "buy", qty, exec_price)
            exec_price = order.price
            qty = order.amount
            order_id = order.order_id
        fee = qty * exec_price * self.fee_rate
        slip_cost = qty * price * slip
        # Coût moyen pondéré de la position agrandie — frais d'achat INCLUS dans
        # la base de coût, sinon le PnL des ventes (et le win-rate) est
        # optimiste d'un côté de frais (~10 bps par jambe).
        prev_cost = self.position * self.entry_price
        self.cash -= qty * exec_price + fee
        self.position += qty
        self.entry_price = (prev_cost + qty * exec_price + fee) / self.position
        self.fees_paid += fee
        self.slippage_paid += slip_cost
        trade = Trade(
            step=step, action=Action.BUY, price=exec_price, size=qty,
            fee=fee, slippage=slip_cost,
            exposure_after=self.exposure(price),
            reason=reason, exchange_order_id=order_id,
        )
        self.trades.append(trade)
        return trade

    def _sell_qty(self, qty: float, price: float, slip: float,
                  step: int, reason: str, symbol: str) -> Trade | None:
        qty = min(max(qty, 0.0), self.position)
        if qty <= 0:
            return None
        exec_price = price * (1.0 - slip)
        order_id = ""
        if self.exchange is not None:
            order = self.exchange.place_market_order(symbol, "sell", qty, exec_price)
            exec_price = order.price
            order_id = order.order_id
        proceeds = qty * exec_price
        fee = proceeds * self.fee_rate
        slip_cost = qty * price * slip
        cost_basis = qty * self.entry_price
        pnl = proceeds - fee - cost_basis
        self.cash += proceeds - fee
        self.position -= qty
        if self.position <= 1e-12:
            self.position = 0.0
            self.entry_price = 0.0
        self.fees_paid += fee
        self.slippage_paid += slip_cost
        trade = Trade(
            step=step, action=Action.SELL, price=exec_price, size=qty,
            pnl=pnl, pnl_pct=(pnl / cost_basis if cost_basis else 0.0),
            fee=fee, slippage=slip_cost,
            exposure_after=self.exposure(price),
            reason=reason, exchange_order_id=order_id,
        )
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------ #
    # API héritée (tout-ou-rien) — tests + clôture de fin de backtest    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def resolve_action(action: Action, has_position: bool) -> Action:
        """Map raw signal to executable action (long-only, no pyramiding)."""
        if not has_position:
            if action == Action.SELL:
                return Action.HOLD
            return action
        if action == Action.BUY:
            return Action.HOLD
        return action

    def execute(
        self,
        action: Action,
        price: float,
        step: int,
        scenario_id: int | None = None,
        confidence: float = 0.0,
        reason: str = "",
        position_frac: float = 0.95,
        symbol: str = "BTC/USDT",
    ) -> Trade | None:
        """Chemin binaire hérité : BUY = tout (95 %), SELL = tout. Sans slippage
        (compatibilité avec les tests historiques) — le chemin de la politique de
        risque passe par :meth:`rebalance`, lui avec slippage."""
        action = self.resolve_action(action, self.position > 0)
        if action == Action.HOLD:
            return None
        if action == Action.BUY:
            value = position_frac * self.mark_to_market(price)
            trade = self._buy_value(value, price, 0.0, step, reason, symbol)
        else:
            trade = self._sell_qty(self.position, price, 0.0, step, reason, symbol)
        if trade is not None:
            trade.scenario_id = scenario_id
            trade.confidence = confidence
        return trade

    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.action == Action.SELL]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed)
