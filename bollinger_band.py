"""
bollinger_band.py — Bollinger Band Mean-Reversion Strategy
═══════════════════════════════════════════════════════════

Classic Bollinger Bands:
    Middle Band = SMA(Close, N)
    Upper  Band = Middle + k × σ(Close, N)
    Lower  Band = Middle - k × σ(Close, N)

Trading rules implemented:
  1. Mean-Reversion (default):
       Buy  when price touches lower band → expect reversion to middle
       Sell when price touches upper band → expect reversion to middle

  2. Breakout variant:
       Buy  when price closes ABOVE upper band (momentum)
       Sell when price closes BELOW lower band (momentum)

  3. Squeeze variant:
       Detects low-volatility compression (BandWidth < threshold)
       Trades the subsequent expansion

Additional features:
  • %B oscillator tracking
  • Bandwidth compression detection
  • RSI filter to avoid false signals in trending markets
  • Walk-forward optimisation of (N, k) parameters
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Dict

from backtesting.backtester import (
    BacktestConfig, BaseStrategy, Trade, compute_metrics, monthly_returns
)

warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────

class BollingerConfig:
    ticker:        str   = "SPY"
    start:         str   = "2019-01-01"
    end:           str   = "2024-12-31"
    window:        int   = 20           # SMA/std window
    n_std:         float = 2.0          # band width multiplier
    rsi_window:    int   = 14           # RSI filter period
    rsi_oversold:  float = 40.0         # RSI threshold for longs
    rsi_overbought:float = 60.0         # RSI threshold for shorts
    use_rsi_filter:bool  = True         # only trade with RSI confirmation
    variant:       str   = "mean_reversion"  # or "breakout" or "squeeze"


# ─── Indicator calculations ───────────────────────────────────────────────

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs       = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_bollinger(prices: pd.Series, window: int, n_std: float) -> pd.DataFrame:
    sma    = prices.rolling(window).mean()
    std    = prices.rolling(window).std(ddof=0)
    upper  = sma + n_std * std
    lower  = sma - n_std * std
    pct_b  = (prices - lower) / (upper - lower)   # %B oscillator
    bw     = (upper - lower) / sma * 100           # BandWidth %

    return pd.DataFrame({
        "close":  prices,
        "sma":    sma,
        "upper":  upper,
        "lower":  lower,
        "std":    std,
        "pct_b":  pct_b,
        "bandwidth": bw,
    })


def detect_squeeze(bw: pd.Series, percentile: float = 20.0) -> pd.Series:
    """True when bandwidth is in its lowest N-th percentile (compression)."""
    rolling_lo = bw.rolling(126).quantile(percentile / 100)
    return bw <= rolling_lo


# ─── Strategy ─────────────────────────────────────────────────────────────

class BollingerBandStrategy(BaseStrategy):
    name = "Bollinger Bands"

    def __init__(self, bb_cfg: BollingerConfig = BollingerConfig(),
                 config: BacktestConfig = BacktestConfig()):
        super().__init__(config)
        self.bb_cfg = bb_cfg

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg    = self.bb_cfg
        prices = data["Close"] if "Close" in data.columns else data.iloc[:, 0]

        bb  = compute_bollinger(prices, cfg.window, cfg.n_std)
        rsi = compute_rsi(prices, cfg.rsi_window)
        squeeze = detect_squeeze(bb["bandwidth"])

        signal  = pd.Series(0, index=prices.index)
        position = 0

        for i in range(max(cfg.window, cfg.rsi_window) + 1, len(prices)):
            px  = float(prices.iloc[i])
            up  = float(bb["upper"].iloc[i])
            lo  = float(bb["lower"].iloc[i])
            mid = float(bb["sma"].iloc[i])
            r   = float(rsi.iloc[i])
            bwi = float(bb["bandwidth"].iloc[i])
            sq  = bool(squeeze.iloc[i])

            if np.isnan(up) or np.isnan(lo) or np.isnan(r):
                signal.iloc[i] = position
                continue

            if cfg.variant == "mean_reversion":
                if position == 0:
                    long_ok  = (not cfg.use_rsi_filter) or (r < cfg.rsi_oversold)
                    short_ok = (not cfg.use_rsi_filter) or (r > cfg.rsi_overbought)
                    if px <= lo and long_ok:
                        position = 1
                    elif px >= up and short_ok:
                        position = -1
                elif position == 1:
                    # Exit long when price crosses back above middle band
                    if px >= mid:
                        position = 0
                elif position == -1:
                    # Exit short when price crosses back below middle band
                    if px <= mid:
                        position = 0

            elif cfg.variant == "breakout":
                if position == 0:
                    if px > up:
                        position = 1   # Momentum buy on breakout above
                    elif px < lo:
                        position = -1  # Momentum short on breakout below
                elif position == 1:
                    if px < mid:
                        position = 0
                elif position == -1:
                    if px > mid:
                        position = 0

            elif cfg.variant == "squeeze":
                # Wait for squeeze → enter breakout direction on expansion
                prev_sq = bool(squeeze.iloc[i - 1])
                if prev_sq and not sq:
                    # Squeeze just fired
                    momentum = prices.iloc[i] - prices.iloc[i - 5]
                    if position == 0:
                        position = 1 if momentum > 0 else -1
                elif position != 0 and abs(float(bb["pct_b"].iloc[i])) < 0.5:
                    position = 0

            signal.iloc[i] = position

        bb["signal"]  = signal
        bb["rsi"]     = rsi
        bb["squeeze"] = squeeze
        return bb


# ─── Walk-forward optimisation ────────────────────────────────────────────

def walk_forward_optimise(
    data:       pd.DataFrame,
    config:     BacktestConfig,
    in_window:  int = 252,
    out_window: int = 63,
) -> pd.DataFrame:
    """
    Roll through the data, optimise (window, n_std) on in-sample,
    apply best params out-of-sample.

    Returns DataFrame of out-of-sample equity contributions.
    """
    windows = [10, 15, 20, 30]
    n_stds  = [1.5, 2.0, 2.5]
    results = []

    n = len(data)
    for start in range(0, n - in_window - out_window, out_window):
        in_data  = data.iloc[start: start + in_window]
        out_data = data.iloc[start + in_window: start + in_window + out_window]

        # Grid search on in-sample
        best_sharpe, best_w, best_k = -999, 20, 2.0
        for w in windows:
            for k in n_stds:
                cfg = BollingerConfig()
                cfg.window  = w
                cfg.n_std   = k
                cfg.use_rsi_filter = False
                strat = BollingerBandStrategy(cfg, config)
                eq, _, m = strat.run(in_data)
                if m["Sharpe Ratio"] > best_sharpe:
                    best_sharpe = m["Sharpe Ratio"]
                    best_w, best_k = w, k

        # Apply best params out-of-sample
        cfg = BollingerConfig()
        cfg.window = best_w
        cfg.n_std  = best_k
        strat  = BollingerBandStrategy(cfg, config)
        eq, _, m = strat.run(out_data)
        results.append({
            "period_start": out_data.index[0],
            "best_window":  best_w,
            "best_n_std":   best_k,
            "sharpe":       m["Sharpe Ratio"],
            "total_return": m["Total Return (%)"],
        })

    return pd.DataFrame(results)


# ─── Convenience runner ───────────────────────────────────────────────────

def load_data(cfg: BollingerConfig) -> pd.DataFrame:
    print(f"  Fetching {cfg.ticker}...")
    data = yf.download(cfg.ticker, start=cfg.start, end=cfg.end,
                       auto_adjust=True, progress=False)
    print(f"  Loaded {len(data)} rows")
    return data


def run(config: BacktestConfig = BacktestConfig()) -> Dict:
    cfg  = BollingerConfig()
    data = load_data(cfg)

    results = {}
    for variant in ["mean_reversion", "breakout", "squeeze"]:
        bb_cfg = BollingerConfig()
        bb_cfg.variant = variant
        strat  = BollingerBandStrategy(bb_cfg, config)
        strat.name = f"Bollinger ({variant.replace('_',' ').title()})"
        eq, trades, metrics = strat.run(data)
        signals = strat.generate_signals(data)
        results[variant] = {
            "equity":   eq,
            "trades":   trades,
            "metrics":  metrics,
            "signals":  signals,
            "data":     data,
        }

    return results
