"""
dual_class_arb.py — Dual-Class Share Arbitrage Strategy
═════════════════════════════════════════════════════════

Exploits price discrepancies between dual-listed share classes of
the same company, e.g.:
  • GOOGL (Class A, voting)  vs  GOOG  (Class C, non-voting)
  • BRK.A vs BRK.B
  • MKL.A vs MKL (Markel)

Theory:
  Both share classes represent ownership in the same underlying business.
  Any persistent price spread beyond a theoretical ratio is an arbitrage.

  Spread = Price_A − ratio × Price_C
  Z-score = (Spread − MA_spread) / std_spread

  • Z > +z_entry → GOOG expensive → SHORT GOOG, LONG GOOGL
  • Z < -z_entry → GOOGL expensive → LONG GOOG, SHORT GOOGL
  • |Z| < z_exit → Close position

Risk considerations:
  • Structural spread can persist (voting premium)
  • Funding cost of short leg
  • Regulatory risk

This implementation also computes:
  • Cointegration test on the spread
  • Rolling hedge ratio via OLS
  • Voting premium analysis
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from typing import Tuple, List, Dict

from backtesting.backtester import (
    BacktestConfig, BaseStrategy, Trade, compute_metrics, monthly_returns
)

warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────

DUAL_CLASS_PAIRS = [
    ("GOOGL", "GOOG",  "Alphabet A/C"),
    ("BRK-A", "BRK-B", "Berkshire A/B"),
]


class DualClassArbConfig:
    # Pair to trade
    ticker_a:     str   = "GOOGL"
    ticker_b:     str   = "GOOG"
    pair_name:    str   = "Alphabet A/C"

    # Signal params
    z_entry:      float = 2.0    # z-score to enter
    z_exit:       float = 0.5    # z-score to exit
    z_stop:       float = 4.0    # hard stop (spread blowing out)
    lookback:     int   = 60     # rolling window for z-score

    # Data
    start:        str   = "2019-01-01"
    end:          str   = "2024-12-31"


# ─── Data loading ─────────────────────────────────────────────────────────

def load_dual_class_data(cfg: DualClassArbConfig) -> pd.DataFrame:
    print(f"  Fetching {cfg.ticker_a} and {cfg.ticker_b}...")
    raw = yf.download(
        [cfg.ticker_a, cfg.ticker_b],
        start=cfg.start, end=cfg.end,
        auto_adjust=True, progress=False
    )
    prices = raw["Close"][[cfg.ticker_a, cfg.ticker_b]].dropna()
    prices.columns = ["A", "B"]
    print(f"  Loaded {len(prices)} trading days")
    return prices


# ─── Cointegration & hedge ratio ──────────────────────────────────────────

def analyse_pair(prices: pd.DataFrame) -> Dict:
    """Run cointegration test and compute rolling hedge ratio."""
    A, B = prices["A"], prices["B"]

    # Static OLS hedge ratio
    model  = OLS(A, add_constant(B)).fit()
    hedge  = model.params["B"]
    spread = A - hedge * B

    # Engle-Granger cointegration
    _, pval, _ = coint(A, B)

    # ADF on spread
    adf_stat, adf_pval, *_ = adfuller(spread.dropna())

    # Theoretical ratio (simple price ratio over full history)
    ratio_mean = (A / B).mean()
    ratio_std  = (A / B).std()

    print(f"\n  ── Pair Analysis ──────────────────────────────")
    print(f"  OLS Hedge Ratio:      {hedge:.4f}")
    print(f"  Price Ratio (mean):   {ratio_mean:.4f} ± {ratio_std:.4f}")
    print(f"  Cointegration p-val:  {pval:.4f} {'✓' if pval < 0.05 else '✗'}")
    print(f"  ADF on spread p-val:  {adf_pval:.4f} {'✓' if adf_pval < 0.05 else '✗'}")
    print(f"  Spread mean:          {spread.mean():.4f}")
    print(f"  Spread std:           {spread.std():.4f}")

    return {
        "hedge_ratio":  hedge,
        "coint_pval":   pval,
        "adf_pval":     adf_pval,
        "spread":       spread,
        "ratio":        A / B,
    }


# ─── Strategy ─────────────────────────────────────────────────────────────

class DualClassArbStrategy(BaseStrategy):
    name = "Dual-Class Arbitrage"

    def __init__(self, arb_cfg: DualClassArbConfig = DualClassArbConfig(),
                 config: BacktestConfig = BacktestConfig()):
        super().__init__(config)
        self.arb_cfg = arb_cfg

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Signals on the spread = A − hedge × B.
        +1  → long spread  (long A, short B)
        -1  → short spread (short A, long B)
         0  → flat
        """
        cfg = self.arb_cfg
        A, B = prices["A"], prices["B"]

        # Rolling OLS hedge ratio
        hedge_ratios = pd.Series(index=prices.index, dtype=float)
        for i in range(cfg.lookback, len(prices)):
            window_A = A.iloc[i - cfg.lookback: i]
            window_B = B.iloc[i - cfg.lookback: i]
            m = OLS(window_A, add_constant(window_B)).fit()
            hedge_ratios.iloc[i] = m.params.iloc[1]

        hedge_ratios = hedge_ratios.ffill().bfill()
        spread = A - hedge_ratios * B

        roll_mean = spread.rolling(cfg.lookback).mean()
        roll_std  = spread.rolling(cfg.lookback).std()
        zscore    = (spread - roll_mean) / roll_std

        # Signal generation
        position = pd.Series(0, index=prices.index)
        pos = 0
        for i in range(cfg.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue
            if pos == 0:
                if z > cfg.z_entry:
                    pos = -1   # Short spread (A expensive)
                elif z < -cfg.z_entry:
                    pos = 1    # Long spread  (B expensive)
            elif pos == 1:
                if z > -cfg.z_exit or abs(z) > cfg.z_stop:
                    pos = 0
            elif pos == -1:
                if z < cfg.z_exit or abs(z) > cfg.z_stop:
                    pos = 0
            position.iloc[i] = pos

        out = pd.DataFrame({
            "A":           A,
            "B":           B,
            "hedge":       hedge_ratios,
            "spread":      spread,
            "zscore":      zscore,
            "signal":      position,
            "roll_mean":   roll_mean,
            "roll_std":    roll_std,
        })
        return out

    def run_pair(self) -> Dict:
        """Full pipeline: load → analyse → signal → backtest."""
        cfg = self.arb_cfg
        prices = load_dual_class_data(cfg)
        analysis = analyse_pair(prices)
        signals  = self.generate_signals(prices)

        # Simplified equity sim for pair (long A + short B)
        slip  = self.config.slippage_bps / 10_000
        comm  = self.config.commission_bps / 10_000
        cap   = self.config.initial_capital

        equity = pd.Series(cap, index=prices.index, dtype=float)
        pos     = 0
        pos_val = 0.0
        entry_A = entry_B = hedge_at_entry = 0.0
        trades: List[Trade] = []

        for i in range(1, len(signals)):
            date = signals.index[i]
            sig  = signals["signal"].iloc[i]
            A_px = float(prices["A"].iloc[i])
            B_px = float(prices["B"].iloc[i])
            h    = float(signals["hedge"].iloc[i])

            if sig != pos:
                if pos != 0:
                    # Close
                    pnl = pos * ((A_px - entry_A) - h * (B_px - entry_B)) * (cap / 2 / entry_A)
                    trades.append(Trade(
                        entry_date  = signals.index[i-1],
                        exit_date   = date,
                        side        = "long" if pos == 1 else "short",
                        entry_price = entry_A,
                        exit_price  = A_px,
                        shares      = cap / 2 / entry_A,
                        pnl         = pnl,
                        pnl_pct     = pnl / cap * 100,
                        commission  = cap * comm * 2,
                        strategy    = self.name,
                    ))
                    cap += pnl - cap * comm * 2

                if sig != 0:
                    entry_A = A_px * (1 + sig * slip)
                    entry_B = B_px * (1 - sig * slip)
                    hedge_at_entry = h

                pos = sig

            # MTM
            if pos != 0:
                pnl_mtm = pos * ((A_px - entry_A) - hedge_at_entry * (B_px - entry_B)) * (cap / 2 / entry_A)
                equity.iloc[i] = cap + pnl_mtm
            else:
                equity.iloc[i] = cap

        metrics = compute_metrics(equity, trades, self.config, self.name)
        return {
            "prices":   prices,
            "signals":  signals,
            "equity":   equity,
            "trades":   trades,
            "metrics":  metrics,
            "analysis": analysis,
        }


def run(config: BacktestConfig = BacktestConfig()) -> Dict:
    cfg = DualClassArbConfig()
    strategy = DualClassArbStrategy(cfg, config)
    return strategy.run_pair()
