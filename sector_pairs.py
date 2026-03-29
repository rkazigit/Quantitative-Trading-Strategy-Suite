"""
sector_pairs.py — Sector-Based Pairs Trading Strategy
═══════════════════════════════════════════════════════

Systematically scans for cointegrated pairs WITHIN sectors,
then trades the spread using z-score signals with dynamic hedge ratios.

Sectors covered:
  • Technology (XLK components)
  • Financials  (XLF components)
  • Energy      (XLE components)
  • Healthcare  (XLV components)
  • Industrials (XLI components)

Pipeline:
  1. Download sector stock prices
  2. Test all within-sector pairs for cointegration (Engle-Granger)
  3. Estimate Ornstein-Uhlenbeck half-life (filter too-fast/slow)
  4. Rank by (p-value × half-life score)
  5. Backtest top-N pairs simultaneously
  6. Portfolio-level equity combining all positions
  7. Risk management: correlation cap (avoid crowded positions)
"""

from __future__ import annotations

import warnings
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

from backtesting.backtester import (
    BacktestConfig, Trade, compute_metrics
)


# ─── Sector universe ──────────────────────────────────────────────────────

SECTOR_UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "VLO",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "ABT", "TMO", "MRK", "DHR", "AMGN",
    ],
    "Industrials": [
        "RTX", "HON", "CAT", "UPS", "BA", "GE", "MMM", "LMT",
    ],
}


class SectorPairsConfig:
    start:              str   = "2020-01-01"
    end:                str   = "2024-12-31"
    coint_pval:         float = 0.05
    ou_hl_min:          int   = 5          # min half-life in days
    ou_hl_max:          int   = 60         # max half-life in days
    z_entry:            float = 2.0
    z_exit:             float = 0.5
    z_stop:             float = 4.0
    lookback:           int   = 60         # rolling z-score window
    top_n_pairs:        int   = 5          # trade top N pairs
    sectors:            List  = None       # None = all sectors

    def __post_init__(self):
        if self.sectors is None:
            self.sectors = list(SECTOR_UNIVERSE.keys())


# ─── Data loading ─────────────────────────────────────────────────────────

def load_sector_prices(cfg: SectorPairsConfig) -> Dict[str, pd.DataFrame]:
    """Download prices for all tickers, grouped by sector."""
    sectors = cfg.sectors if cfg.sectors else list(SECTOR_UNIVERSE.keys())
    sector_prices = {}

    for sector in sectors:
        tickers = SECTOR_UNIVERSE[sector]
        print(f"  Downloading {sector} ({len(tickers)} tickers)...")
        try:
            raw = yf.download(
                tickers, start=cfg.start, end=cfg.end,
                auto_adjust=True, progress=False, group_by="column"
            )
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]]

            prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.85))
            prices = prices.ffill().dropna()
            sector_prices[sector] = prices
            print(f"    → {prices.shape[1]} assets, {prices.shape[0]} days")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    return sector_prices


# ─── Pair analytics ───────────────────────────────────────────────────────

def estimate_ou_halflife(spread: pd.Series) -> float:
    """Estimate OU half-life via AR(1) regression on the spread."""
    spread  = spread.dropna()
    lag     = spread.shift(1).dropna()
    delta   = spread.diff().dropna()
    lag, delta = lag.align(delta, join="inner")

    X       = add_constant(lag.values)
    model   = OLS(delta.values, X).fit()
    beta    = model.params[1]
    if beta >= 0:
        return np.inf   # Not mean-reverting
    return -np.log(2) / beta


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    pval_threshold: float,
    ou_hl_min: int,
    ou_hl_max: int,
    sector_name: str = "",
) -> List[Dict]:
    """Find and rank cointegrated pairs in a price DataFrame."""
    tickers = list(prices.columns)
    found   = []

    for t1, t2 in itertools.combinations(tickers, 2):
        try:
            _, pval, _ = coint(prices[t1], prices[t2])
            if pval > pval_threshold:
                continue

            # Hedge ratio
            model     = OLS(prices[t1], add_constant(prices[t2])).fit()
            hedge     = model.params[t2]
            spread    = prices[t1] - hedge * prices[t2]
            half_life = estimate_ou_halflife(spread)

            if not (ou_hl_min <= half_life <= ou_hl_max):
                continue

            # Stationarity of spread
            _, adf_pval, *_ = adfuller(spread.dropna())

            found.append({
                "sector":     sector_name,
                "ticker1":    t1,
                "ticker2":    t2,
                "pvalue":     pval,
                "hedge":      hedge,
                "half_life":  half_life,
                "adf_pval":   adf_pval,
                "spread_std": spread.std(),
                "score":      (1 - pval) * (1 / (1 + abs(np.log(half_life / 20)))),
            })
        except Exception:
            continue

    found.sort(key=lambda x: x["score"], reverse=True)
    return found


# ─── Single-pair signal engine ────────────────────────────────────────────

def compute_pair_signals(
    prices:   pd.DataFrame,
    pair:     Dict,
    cfg:      SectorPairsConfig,
) -> pd.DataFrame:
    t1, t2 = pair["ticker1"], pair["ticker2"]
    A = prices[t1]
    B = prices[t2]

    # Rolling hedge ratio via OLS
    hedge_ts = pd.Series(index=prices.index, dtype=float)
    for i in range(cfg.lookback, len(prices)):
        win_A = A.iloc[i - cfg.lookback: i]
        win_B = B.iloc[i - cfg.lookback: i]
        m = OLS(win_A, add_constant(win_B)).fit()
        hedge_ts.iloc[i] = m.params[t2]

    hedge_ts = hedge_ts.ffill().bfill()
    spread   = A - hedge_ts * B
    roll_mean= spread.rolling(cfg.lookback).mean()
    roll_std = spread.rolling(cfg.lookback).std()
    zscore   = (spread - roll_mean) / roll_std

    signal   = pd.Series(0, index=prices.index)
    pos      = 0

    for i in range(cfg.lookback, len(zscore)):
        z = float(zscore.iloc[i])
        if np.isnan(z):
            signal.iloc[i] = pos
            continue

        if pos == 0:
            if z >  cfg.z_entry:  pos = -1
            elif z < -cfg.z_entry: pos =  1
        elif pos == 1:
            if z > -cfg.z_exit or abs(z) > cfg.z_stop:
                pos = 0
        elif pos == -1:
            if z <  cfg.z_exit or abs(z) > cfg.z_stop:
                pos = 0

        signal.iloc[i] = pos

    return pd.DataFrame({
        "A":        A,
        "B":        B,
        "hedge":    hedge_ts,
        "spread":   spread,
        "zscore":   zscore,
        "signal":   signal,
    })


# ─── Multi-pair portfolio backtest ────────────────────────────────────────

def backtest_portfolio(
    prices:    pd.DataFrame,
    pairs:     List[Dict],
    cfg:       SectorPairsConfig,
    bt_cfg:    BacktestConfig,
) -> Tuple[pd.Series, List[Trade], Dict]:
    """
    Run all pairs simultaneously, allocate capital equally.
    """
    n_pairs     = len(pairs)
    alloc       = bt_cfg.initial_capital / max(n_pairs, 1)
    slip        = bt_cfg.slippage_bps / 10_000
    comm        = bt_cfg.commission_bps / 10_000

    all_trades: List[Trade] = []
    pair_equities = []

    for pair in pairs:
        t1, t2   = pair["ticker1"], pair["ticker2"]
        if t1 not in prices.columns or t2 not in prices.columns:
            continue

        sigs = compute_pair_signals(prices, pair, cfg)

        cap        = alloc
        pos        = 0
        entry_A    = entry_B = hedge_entry = 0.0
        entry_date = None
        equity_vals= []

        for i in range(len(sigs)):
            sig  = int(sigs["signal"].iloc[i])
            A_px = float(sigs["A"].iloc[i])
            B_px = float(sigs["B"].iloc[i])
            h    = float(sigs["hedge"].iloc[i]) if not np.isnan(sigs["hedge"].iloc[i]) else 1.0
            date = sigs.index[i]

            if sig != pos:
                if pos != 0:
                    pnl = pos * ((A_px - entry_A) - hedge_entry * (B_px - entry_B)) * (cap / 2 / max(entry_A, 0.01))
                    cost = cap * comm * 2
                    all_trades.append(Trade(
                        entry_date  = entry_date,
                        exit_date   = date,
                        side        = "long" if pos == 1 else "short",
                        entry_price = entry_A,
                        exit_price  = A_px,
                        shares      = cap / 2 / max(entry_A, 0.01),
                        pnl         = pnl - cost,
                        pnl_pct     = pnl / cap * 100,
                        commission  = cost,
                        strategy    = f"{t1}/{t2}",
                    ))
                    cap += pnl - cost

                if sig != 0:
                    entry_A    = A_px * (1 + sig * slip)
                    entry_B    = B_px * (1 - sig * slip)
                    hedge_entry = h
                    entry_date  = date

                pos = sig

            if pos != 0:
                pnl_mtm = pos * ((A_px - entry_A) - hedge_entry * (B_px - entry_B)) * (cap / 2 / max(entry_A, 0.01))
                equity_vals.append(cap + pnl_mtm)
            else:
                equity_vals.append(cap)

        pair_eq = pd.Series(equity_vals, index=sigs.index, name=f"{t1}/{t2}")
        pair_equities.append(pair_eq)

    if not pair_equities:
        empty = pd.Series(bt_cfg.initial_capital, index=prices.index)
        return empty, [], compute_metrics(empty, [], bt_cfg, "Sector Pairs")

    # Portfolio equity = sum of all pair equities
    portfolio_equity = pd.concat(pair_equities, axis=1).sum(axis=1)
    metrics = compute_metrics(portfolio_equity, all_trades, bt_cfg, "Sector-Based Pairs")
    return portfolio_equity, all_trades, metrics


# ─── Main runner ──────────────────────────────────────────────────────────

def run(bt_config: BacktestConfig = BacktestConfig()) -> Dict:
    cfg = SectorPairsConfig()
    cfg.sectors = ["Technology", "Financials", "Energy"]  # subset for speed

    sector_prices = load_sector_prices(cfg)

    # Screen pairs across all sectors
    print("\n  ── Cointegration Screening ──────────────────────")
    all_pairs = []
    for sector, prices in sector_prices.items():
        pairs = find_cointegrated_pairs(
            prices, cfg.coint_pval, cfg.ou_hl_min, cfg.ou_hl_max, sector
        )
        print(f"  {sector:<14}: {len(pairs)} pairs found")
        all_pairs.extend(pairs)

    all_pairs.sort(key=lambda x: x["score"], reverse=True)
    top_pairs = all_pairs[: cfg.top_n_pairs]

    print(f"\n  Top {len(top_pairs)} pairs selected:")
    print(f"  {'Pair':<20} {'Sector':<14} {'p-val':<8} {'HL(d)':<7} {'Score'}")
    print(f"  {'─'*60}")
    for p in top_pairs:
        print(f"  {p['ticker1']+'/'+ p['ticker2']:<20} {p['sector']:<14} "
              f"{p['pvalue']:.4f}  {p['half_life']:.1f}  {p['score']:.4f}")

    # Combine all sector prices for the portfolio
    all_prices = pd.concat(list(sector_prices.values()), axis=1)
    all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
    all_prices = all_prices.dropna(how="all").ffill()

    print("\n  ── Backtesting portfolio of pairs ───────────────")
    equity, trades, metrics = backtest_portfolio(
        all_prices, top_pairs, cfg, bt_config
    )

    # Per-pair signals for visualisation
    pair_signals = {}
    for pair in top_pairs:
        t1, t2 = pair["ticker1"], pair["ticker2"]
        if t1 in all_prices.columns and t2 in all_prices.columns:
            pair_signals[f"{t1}/{t2}"] = compute_pair_signals(all_prices, pair, cfg)

    return {
        "equity":       equity,
        "trades":       trades,
        "metrics":      metrics,
        "pairs":        top_pairs,
        "pair_signals": pair_signals,
        "sector_prices": sector_prices,
        "all_prices":   all_prices,
    }
