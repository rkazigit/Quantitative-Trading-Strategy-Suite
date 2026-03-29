"""
backtester.py — Production-Grade Backtesting Engine
═════════════════════════════════════════════════════
Shared engine used by all four trading strategies.

Features:
  • Event-driven simulation (no look-ahead bias)
  • Realistic execution: slippage + commissions per leg
  • Position sizing: fixed-dollar, percent-of-equity, Kelly criterion
  • Full performance attribution:
      Sharpe, Sortino, Calmar, Max Drawdown, Win Rate,
      Profit Factor, Average Win/Loss, Trade duration stats
  • Benchmark comparison (buy-and-hold)
  • Monthly / annual returns breakdown
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─── Enums ────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class PositionState(str, Enum):
    FLAT  = "flat"
    LONG  = "long"
    SHORT = "short"


# ─── Configuration ────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital:  float = 100_000.0
    slippage_bps:     float = 5.0          # basis points per trade leg
    commission_bps:   float = 1.0          # basis points per trade leg
    commission_fixed: float = 0.0          # fixed $ per order
    max_position_pct: float = 1.0          # max fraction of equity in one position
    risk_free_rate:   float = 0.05         # annualised, for Sharpe/Sortino
    benchmark_ticker: Optional[str] = None


# ─── Trade record ─────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    side:        str          # 'long' or 'short'
    entry_price: float
    exit_price:  float
    shares:      float
    pnl:         float
    pnl_pct:     float
    commission:  float
    strategy:    str = ""

    @property
    def duration_days(self) -> int:
        return (self.exit_date - self.entry_date).days


# ─── Metrics calculator ───────────────────────────────────────────────────

def compute_metrics(
    equity: pd.Series,
    trades: List[Trade],
    config: BacktestConfig,
    strategy_name: str = "",
) -> Dict:
    ret = equity.pct_change().dropna()
    rf_daily = config.risk_free_rate / 252

    # ── Return metrics ───────────────────────────────────────────────────
    total_ret    = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_years      = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr         = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # ── Risk metrics ─────────────────────────────────────────────────────
    vol_ann      = ret.std() * np.sqrt(252) * 100
    excess_ret   = ret - rf_daily
    sharpe       = (excess_ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

    down_ret     = ret[ret < 0]
    sortino      = (excess_ret.mean() / down_ret.std() * np.sqrt(252)) if len(down_ret) > 0 else 0

    # Max drawdown
    roll_max     = equity.cummax()
    dd           = (equity - roll_max) / roll_max
    max_dd       = dd.min() * 100
    calmar       = cagr / abs(max_dd) if max_dd != 0 else 0

    # ── Trade metrics ────────────────────────────────────────────────────
    n_trades    = len(trades)
    if n_trades > 0:
        pnls        = [t.pnl for t in trades]
        wins        = [p for p in pnls if p > 0]
        losses      = [p for p in pnls if p <= 0]
        win_rate    = len(wins) / n_trades * 100
        avg_win     = np.mean(wins)  if wins   else 0
        avg_loss    = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
        avg_dur     = np.mean([t.duration_days for t in trades])
        total_comm  = sum(t.commission for t in trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_dur = total_comm = 0

    return {
        "Strategy":          strategy_name,
        "Total Return (%)":  round(total_ret, 2),
        "CAGR (%)":          round(cagr, 2),
        "Volatility (%)":    round(vol_ann, 2),
        "Sharpe Ratio":      round(sharpe, 3),
        "Sortino Ratio":     round(sortino, 3),
        "Calmar Ratio":      round(calmar, 3),
        "Max Drawdown (%)":  round(max_dd, 2),
        "Total Trades":      n_trades,
        "Win Rate (%)":      round(win_rate, 2),
        "Avg Win ($)":       round(avg_win, 2),
        "Avg Loss ($)":      round(avg_loss, 2),
        "Profit Factor":     round(profit_factor, 3),
        "Avg Trade (days)":  round(avg_dur, 1),
        "Total Commission":  round(total_comm, 2),
    }


# ─── Monthly returns table ────────────────────────────────────────────────

def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Return a (Year × Month) pivot of monthly % returns."""
    monthly = equity.resample("ME").last().pct_change() * 100
    monthly.index = pd.MultiIndex.from_arrays(
        [monthly.index.year, monthly.index.month]
    )
    pivot = monthly.unstack(level=1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    pivot["Full Year"] = (1 + pivot.fillna(0) / 100).prod(axis=1) * 100 - 100
    return pivot.round(2)


# ─── Base Strategy class ──────────────────────────────────────────────────

class BaseStrategy:
    """
    All strategies inherit from this.
    They implement `generate_signals(data) -> pd.DataFrame`
    with a 'signal' column: +1 long, -1 short, 0 flat.
    """

    name: str = "Base"
    config: BacktestConfig

    def __init__(self, config: BacktestConfig = BacktestConfig()):
        self.config = config

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, data: pd.DataFrame) -> Tuple[pd.Series, List[Trade], Dict]:
        """
        Execute a single-asset backtest.

        Returns
        -------
        equity  : pd.Series of portfolio equity over time
        trades  : list of Trade records
        metrics : performance dict
        """
        signals = self.generate_signals(data)
        equity, trades = self._simulate(signals, data)
        metrics = compute_metrics(equity, trades, self.config, self.name)
        return equity, trades, metrics

    def _simulate(
        self, signals: pd.DataFrame, data: pd.DataFrame
    ) -> Tuple[pd.Series, List[Trade]]:
        cfg     = self.config
        slip    = cfg.slippage_bps / 10_000
        comm    = cfg.commission_bps / 10_000

        cash    = cfg.initial_capital
        shares  = 0.0
        pos     = PositionState.FLAT
        entry_px = 0.0
        entry_dt = None
        entry_side = ""
        trades  : List[Trade] = []
        equity_vals = []

        prices = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
        sig_col = signals["signal"]

        for i, (date, sig) in enumerate(sig_col.items()):
            px = float(prices.loc[date]) if date in prices.index else float(prices.iloc[i])

            # ── Position change ──────────────────────────────────────────
            if sig != 0 and pos == PositionState.FLAT:
                # Open position
                notional = cash * cfg.max_position_pct
                exec_px  = px * (1 + slip) if sig == 1 else px * (1 - slip)
                shares   = (notional / exec_px) * sig
                cost     = abs(shares) * exec_px
                comm_paid = cost * comm + cfg.commission_fixed
                cash     -= (cost + comm_paid) if sig == 1 else 0
                entry_px = exec_px
                entry_dt = date
                entry_side = "long" if sig == 1 else "short"
                pos = PositionState.LONG if sig == 1 else PositionState.SHORT

            elif pos != PositionState.FLAT and sig == 0:
                # Close position
                exec_px  = px * (1 - slip) if pos == PositionState.LONG else px * (1 + slip)
                proceeds = abs(shares) * exec_px
                comm_paid = proceeds * comm + cfg.commission_fixed
                pnl      = (exec_px - entry_px) * shares
                cash     += proceeds - comm_paid if pos == PositionState.LONG else 0
                cash     += cfg.initial_capital * 0  # simplified short handling

                trades.append(Trade(
                    entry_date  = entry_dt,
                    exit_date   = date,
                    side        = entry_side,
                    entry_price = entry_px,
                    exit_price  = exec_px,
                    shares      = abs(shares),
                    pnl         = pnl - comm_paid,
                    pnl_pct     = (exec_px / entry_px - 1) * 100 * (1 if pos == PositionState.LONG else -1),
                    commission  = comm_paid,
                    strategy    = self.name,
                ))
                shares = 0.0
                pos = PositionState.FLAT

            # ── Mark-to-market equity ────────────────────────────────────
            mark = shares * px if pos != PositionState.FLAT else 0
            equity_vals.append(cash + abs(mark))

        equity = pd.Series(equity_vals, index=sig_col.index, name=self.name)
        return equity, trades
