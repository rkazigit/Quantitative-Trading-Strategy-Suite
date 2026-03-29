"""
charts.py — Multi-Strategy Trading Dashboard
═══════════════════════════════════════════════
Generates publication-quality charts for all four strategies:
  1. Equity curves + drawdown
  2. Monthly returns heatmap
  3. Trade distribution
  4. Strategy-specific signal charts
  5. Comparative performance radar
  6. Rolling Sharpe
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from typing import Dict, List, Optional

from backtesting.backtester import Trade, compute_metrics, monthly_returns, BacktestConfig

warnings.filterwarnings("ignore")

# ─── Theme ────────────────────────────────────────────────────────────────
BG, PANEL, PANEL2 = "#080c14", "#0f1923", "#162030"
ACCENT  = "#00c8ff"
ACCENT2 = "#ff5722"
GREEN   = "#00e676"
RED     = "#ff3d57"
YELLOW  = "#ffca28"
PURPLE  = "#b388ff"
TEAL    = "#1de9b6"
MUTED   = "#37474f"
TEXT    = "#eceff1"
SUBTEXT = "#78909c"
GRID    = "#1a2332"

PALETTE = [ACCENT, GREEN, YELLOW, PURPLE, TEAL, ACCENT2, RED]

CMAP_RET = LinearSegmentedColormap.from_list(
    "ret", ["#ff3d57", "#0f1923", "#00e676"], N=256
)
CMAP_HOT = LinearSegmentedColormap.from_list(
    "hot", ["#080c14", "#1a3a5c", "#00c8ff", "#ffca28", "#ff5722"], N=256
)


def _style(ax, title="", xlabel="", ylabel="", legend=False):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(MUTED)
    if xlabel: ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=8)
    if title:  ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.9, linestyle="--")
    if legend:
        ax.legend(facecolor=PANEL2, edgecolor=MUTED, labelcolor=TEXT,
                  fontsize=7.5, framealpha=0.9)


def _fmt_dollar(x, _):  return f"${x:,.0f}"
def _fmt_pct(x, _):     return f"{x:.1f}%"


# ─── 1. Master equity + drawdown dashboard ────────────────────────────────

def plot_equity_dashboard(
    equities:   Dict[str, pd.Series],   # name → equity series
    trades_map: Dict[str, List[Trade]],
    config:     BacktestConfig,
    filename:   str = "equity_dashboard.png",
):
    n   = len(equities)
    fig = plt.figure(figsize=(20, 5 * n + 3), facecolor=BG)
    gs  = gridspec.GridSpec(
        n + 1, 3, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97, top=0.94, bottom=0.04
    )

    fig.suptitle("Trading Strategy Backtests — Equity & Drawdown",
                 color=TEXT, fontsize=14, fontweight="bold")

    for row, (name, equity) in enumerate(equities.items()):
        # ── Equity ──────────────────────────────────────────────────────
        ax_eq  = fig.add_subplot(gs[row, :2])
        color  = PALETTE[row % len(PALETTE)]

        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max * 100

        ax_eq.plot(equity.index, equity, color=color, lw=1.4, label=name)
        ax_eq.fill_between(equity.index, config.initial_capital, equity,
                           where=(equity >= config.initial_capital),
                           alpha=0.10, color=GREEN)
        ax_eq.fill_between(equity.index, config.initial_capital, equity,
                           where=(equity < config.initial_capital),
                           alpha=0.10, color=RED)
        ax_eq.axhline(config.initial_capital, color=MUTED, lw=0.7, ls="--")

        # Mark trades
        trades = trades_map.get(name, [])
        for t in trades:
            clr = GREEN if t.pnl > 0 else RED
            if t.entry_date in equity.index:
                ax_eq.axvline(t.entry_date, color=clr, lw=0.4, alpha=0.3)

        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_dollar))
        _style(ax_eq, f"{name} — Equity Curve", legend=True)
        ax_eq.legend(facecolor=PANEL2, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)

        # ── Drawdown ────────────────────────────────────────────────────
        ax_dd = ax_eq.twinx()
        ax_dd.fill_between(equity.index, dd, 0, alpha=0.30, color=RED)
        ax_dd.plot(equity.index, dd, color=RED, lw=0.7, alpha=0.7)
        ax_dd.set_ylabel("Drawdown %", color=RED, fontsize=7)
        ax_dd.tick_params(colors=RED, labelsize=7)
        ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pct))
        ax_dd.set_ylim(dd.min() * 1.5, 5)
        for s in ax_dd.spines.values():
            s.set_edgecolor(MUTED)

        # ── Metrics card ────────────────────────────────────────────────
        ax_m = fig.add_subplot(gs[row, 2])
        ax_m.set_facecolor(PANEL2)
        ax_m.axis("off")
        ax_m.set_title(f"Metrics", color=TEXT, fontsize=9, fontweight="bold", pad=5)

        metrics = compute_metrics(equity, trades, config, name)
        metric_items = [
            ("Total Return", f"{metrics['Total Return (%)']:+.2f}%",
             GREEN if metrics['Total Return (%)'] > 0 else RED),
            ("CAGR",         f"{metrics['CAGR (%)']:+.2f}%",
             GREEN if metrics['CAGR (%)'] > 0 else RED),
            ("Sharpe",       f"{metrics['Sharpe Ratio']:.3f}",
             GREEN if metrics['Sharpe Ratio'] > 1 else YELLOW if metrics['Sharpe Ratio'] > 0 else RED),
            ("Sortino",      f"{metrics['Sortino Ratio']:.3f}", SUBTEXT),
            ("Max DD",       f"{metrics['Max Drawdown (%)']:.2f}%", RED),
            ("Calmar",       f"{metrics['Calmar Ratio']:.3f}", SUBTEXT),
            ("Trades",       f"{metrics['Total Trades']}", TEXT),
            ("Win Rate",     f"{metrics['Win Rate (%)']:.1f}%",
             GREEN if metrics['Win Rate (%)'] > 50 else YELLOW),
            ("Profit Factor",f"{metrics['Profit Factor']:.2f}",
             GREEN if metrics['Profit Factor'] > 1 else RED),
            ("Volatility",   f"{metrics['Volatility (%)']:.2f}%", SUBTEXT),
        ]
        for k, (label, val, clr) in enumerate(metric_items):
            y = 0.92 - k * 0.088
            ax_m.text(0.02, y, label, transform=ax_m.transAxes,
                      color=SUBTEXT, fontsize=8.5)
            ax_m.text(0.98, y, val, transform=ax_m.transAxes,
                      color=clr, fontsize=8.5, ha="right", fontweight="bold")

    # ── Bottom row: rolling Sharpe comparison ─────────────────────────
    ax_rs = fig.add_subplot(gs[n, :])
    ax_rs.set_facecolor(PANEL)
    for i, (name, equity) in enumerate(equities.items()):
        ret = equity.pct_change().dropna()
        rolling_sh = ret.rolling(63).mean() / ret.rolling(63).std() * np.sqrt(252)
        ax_rs.plot(rolling_sh.index, rolling_sh,
                   color=PALETTE[i % len(PALETTE)], lw=1.2, label=name, alpha=0.85)
    ax_rs.axhline(0, color=MUTED, lw=0.7)
    ax_rs.axhline(1, color=GREEN, lw=0.5, ls="--", alpha=0.5)
    _style(ax_rs, "Rolling 63-Day Sharpe Ratio", legend=True)

    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved → {filename}")
    return filename


# ─── 2. Monthly returns heatmaps ─────────────────────────────────────────

def plot_monthly_returns(
    equities:  Dict[str, pd.Series],
    filename:  str = "monthly_returns.png",
):
    n   = len(equities)
    fig = plt.figure(figsize=(18, 4 * n), facecolor=BG)
    gs  = gridspec.GridSpec(n, 1, figure=fig, hspace=0.6,
                            left=0.08, right=0.98, top=0.94, bottom=0.04)
    fig.suptitle("Monthly Returns Heatmap (%)",
                 color=TEXT, fontsize=13, fontweight="bold")

    for row, (name, equity) in enumerate(equities.items()):
        ax = fig.add_subplot(gs[row])
        ax.set_facecolor(PANEL)

        mr = monthly_returns(equity)
        # Only numeric columns for heatmap
        num_cols = [c for c in mr.columns if c != "Full Year"]
        data     = mr[num_cols].values

        vmax = max(abs(np.nanpercentile(data, 95)), 2.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(data, aspect="auto", cmap=CMAP_RET, norm=norm)

        ax.set_xticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, color=SUBTEXT, fontsize=8)
        ax.set_yticks(range(len(mr.index)))
        ax.set_yticklabels(mr.index, color=SUBTEXT, fontsize=8)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    clr = TEXT if abs(v) < vmax * 0.6 else BG
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=6.5, color=clr, fontweight="bold")

        # Full year column
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        fy = mr["Full Year"].values
        ax2.set_yticks(range(len(fy)))
        ax2.set_yticklabels(
            [f"{v:+.1f}%" for v in fy],
            fontsize=7.5,
            color=[GREEN if v >= 0 else RED for v in fy]
        )
        for s in ax2.spines.values(): s.set_edgecolor(MUTED)

        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                     label="Return (%)").ax.yaxis.label.set_color(SUBTEXT)
        ax.set_title(f"{name}", color=PALETTE[row % len(PALETTE)],
                     fontsize=10, fontweight="bold", pad=5)
        for s in ax.spines.values(): s.set_edgecolor(MUTED)

    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved → {filename}")
    return filename


# ─── 3. Bollinger signal chart ────────────────────────────────────────────

def plot_bollinger_signals(signals: pd.DataFrame, price_col="close",
                            filename="bollinger_signals.png"):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1]},
                              facecolor=BG)
    fig.suptitle("Bollinger Band Strategy — Signals & Indicators",
                 color=TEXT, fontsize=13, fontweight="bold")

    ax1, ax2, ax3 = axes

    # ── Price + bands ────────────────────────────────────────────────────
    ax1.plot(signals.index, signals[price_col], color=TEXT, lw=1.0, label="Price")
    ax1.plot(signals.index, signals["sma"],     color=YELLOW, lw=0.9, ls="--", label="SMA-20")
    ax1.plot(signals.index, signals["upper"],   color=RED,   lw=0.8, alpha=0.7, label="Upper BB")
    ax1.plot(signals.index, signals["lower"],   color=GREEN, lw=0.8, alpha=0.7, label="Lower BB")
    ax1.fill_between(signals.index, signals["upper"], signals["lower"],
                     alpha=0.06, color=ACCENT)

    # Trade signals
    longs  = signals[signals["signal"].diff() == 1].index
    shorts = signals[signals["signal"].diff() == -1].index
    exits  = signals[signals["signal"].diff().abs() > 0].index

    ax1.scatter(longs,  signals.loc[longs,  price_col], marker="^",
                color=GREEN, s=60, zorder=5, label="Long")
    ax1.scatter(shorts, signals.loc[shorts, price_col], marker="v",
                color=RED,   s=60, zorder=5, label="Short")

    # Shade positions
    pos = signals["signal"]
    ax1.fill_between(signals.index, signals["lower"], signals["upper"],
                     where=(pos == 1),  alpha=0.07, color=GREEN)
    ax1.fill_between(signals.index, signals["lower"], signals["upper"],
                     where=(pos == -1), alpha=0.07, color=RED)

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_dollar))
    _style(ax1, "", legend=True)

    # ── %B oscillator ────────────────────────────────────────────────────
    ax2.plot(signals.index, signals["pct_b"], color=ACCENT, lw=0.9)
    ax2.axhline(1.0, color=RED,   lw=0.6, ls="--", alpha=0.7)
    ax2.axhline(0.0, color=GREEN, lw=0.6, ls="--", alpha=0.7)
    ax2.axhline(0.5, color=MUTED, lw=0.5, ls=":")
    ax2.fill_between(signals.index, signals["pct_b"], 0.5,
                     where=(signals["pct_b"] > 0.5), alpha=0.15, color=RED)
    ax2.fill_between(signals.index, signals["pct_b"], 0.5,
                     where=(signals["pct_b"] < 0.5), alpha=0.15, color=GREEN)
    _style(ax2, "%B Oscillator", ylabel="%B")

    # ── Bandwidth ────────────────────────────────────────────────────────
    ax3.plot(signals.index, signals["bandwidth"], color=PURPLE, lw=0.9)
    if "squeeze" in signals.columns:
        sq_mask = signals["squeeze"]
        ax3.fill_between(signals.index, signals["bandwidth"],
                         where=sq_mask, alpha=0.25, color=YELLOW,
                         label="Squeeze zone")
    _style(ax3, "Bandwidth (Volatility Compression)", ylabel="BW %", legend=True)

    for ax in axes:
        ax.set_facecolor(PANEL)

    fig.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved → {filename}")
    return filename


# ─── 4. Sector pairs heatmap ─────────────────────────────────────────────

def plot_sector_pairs_overview(pairs: List[Dict], pair_signals: Dict,
                                filename="sector_pairs_overview.png"):
    n = len(pair_signals)
    if n == 0:
        return

    fig = plt.figure(figsize=(18, 4 * n), facecolor=BG)
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.55, wspace=0.3,
                            left=0.06, right=0.97, top=0.94, bottom=0.04)
    fig.suptitle("Sector-Based Pairs Trading — Spread & Z-Score",
                 color=TEXT, fontsize=13, fontweight="bold")

    for row, (pair_label, sig) in enumerate(pair_signals.items()):
        # ── Spread ──────────────────────────────────────────────────────
        ax_s = fig.add_subplot(gs[row, :2])
        ax_s.plot(sig.index, sig["spread"], color=ACCENT, lw=0.9, label="Spread")
        ax_s.plot(sig.index, sig["spread"].rolling(60).mean(), color=YELLOW,
                  lw=0.8, ls="--", label="Rolling Mean")
        rolling_std = sig["spread"].rolling(60).std()
        roll_mean   = sig["spread"].rolling(60).mean()
        upper_bb    = roll_mean + 2 * rolling_std
        lower_bb    = roll_mean - 2 * rolling_std
        ax_s.fill_between(sig.index, lower_bb, upper_bb,
                          alpha=0.07, color=ACCENT)

        # Position shading
        ax_s.fill_between(sig.index, lower_bb, upper_bb,
                          where=(sig["signal"] == 1),  alpha=0.12, color=GREEN)
        ax_s.fill_between(sig.index, lower_bb, upper_bb,
                          where=(sig["signal"] == -1), alpha=0.12, color=RED)

        _style(ax_s, f"{pair_label} — Spread", legend=True)

        # ── Z-score ─────────────────────────────────────────────────────
        ax_z = fig.add_subplot(gs[row, 2])
        zs = sig["zscore"].dropna()
        ax_z.plot(zs.index, zs, color=PURPLE, lw=0.9)
        ax_z.axhline( 2, color=RED,   lw=0.6, ls="--")
        ax_z.axhline(-2, color=GREEN, lw=0.6, ls="--")
        ax_z.axhline( 0, color=MUTED, lw=0.5)
        ax_z.fill_between(zs.index, zs, 0,
                          where=(sig["signal"].loc[zs.index] == 1),  alpha=0.18, color=GREEN)
        ax_z.fill_between(zs.index, zs, 0,
                          where=(sig["signal"].loc[zs.index] == -1), alpha=0.18, color=RED)
        _style(ax_z, "Z-Score", ylabel="Z")

    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved → {filename}")
    return filename


# ─── 5. RL training curve ────────────────────────────────────────────────

def plot_rl_training(ql_returns: List[float], dqn_returns: List[float],
                     filename="rl_training.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle("Reinforcement Learning — Training Curves",
                 color=TEXT, fontsize=13, fontweight="bold")

    def _smooth(x, w=5):
        return pd.Series(x).rolling(w, min_periods=1).mean().values

    for ax, returns, title, color in [
        (ax1, ql_returns, "Q-Learning Training", ACCENT),
        (ax2, dqn_returns, "Deep Q-Network Training", GREEN),
    ]:
        ax.set_facecolor(PANEL)
        eps = range(1, len(returns) + 1)
        ax.bar(eps, returns, color=color, alpha=0.25)
        ax.plot(eps, _smooth(returns), color=color, lw=2.0, label="Smoothed reward")
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xlabel("Episode", color=SUBTEXT, fontsize=9)
        ax.set_ylabel("Total Episode Reward", color=SUBTEXT, fontsize=9)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold")
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        ax.grid(color=GRID, linewidth=0.4, alpha=0.8, linestyle="--")
        for s in ax.spines.values(): s.set_edgecolor(MUTED)

    fig.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved → {filename}")
    return filename


# ─── 6. Comparative radar ────────────────────────────────────────────────

def plot_comparison_radar(metrics_list: List[Dict], filename="comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
    fig.suptitle("Strategy Comparison", color=TEXT, fontsize=14, fontweight="bold")

    # ── Bar chart comparison ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    keys    = ["Total Return (%)", "Sharpe Ratio", "Sortino Ratio",
               "Calmar Ratio", "Win Rate (%)", "Max Drawdown (%)"]
    names   = [m["Strategy"] for m in metrics_list]
    x       = np.arange(len(keys))
    width   = 0.8 / len(names)

    for i, (m, color) in enumerate(zip(metrics_list, PALETTE)):
        vals = [m.get(k, 0) for k in keys]
        bars = ax.bar(x + i * width, vals, width, label=m["Strategy"],
                      color=color, alpha=0.8)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([k.replace(" (%)", "").replace(" Ratio", "") for k in keys],
                       rotation=25, ha="right", color=SUBTEXT, fontsize=8)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.legend(facecolor=PANEL2, edgecolor=MUTED, labelcolor=TEXT, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.grid(color=GRID, lw=0.4, alpha=0.8, axis="y", ls="--")
    for s in ax.spines.values(): s.set_edgecolor(MUTED)
    ax.set_title("Key Metrics Comparison", color=TEXT, fontsize=11, fontweight="bold")

    # ── Summary table ────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL2)
    ax2.axis("off")
    ax2.set_title("Full Performance Table", color=TEXT, fontsize=11, fontweight="bold", pad=8)

    col_keys = ["Strategy", "Total Return (%)", "CAGR (%)", "Sharpe Ratio",
                "Sortino Ratio", "Max Drawdown (%)", "Total Trades",
                "Win Rate (%)", "Profit Factor"]
    rows = [[str(m.get(k, "—")) for k in col_keys] for m in metrics_list]
    col_labels = [k.replace(" (%)", "\n(%)").replace(" Ratio", "\nRatio") for k in col_keys]

    tbl = ax2.table(
        cellText   = rows,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
        bbox       = [0.0, 0.0, 1.0, 0.95],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for (row, col), cell in tbl.get_celld().items():
        bg = "#1a3a5c" if row == 0 else (PANEL if row % 2 == 0 else PANEL2)
        cell.set_facecolor(bg)
        cell.set_edgecolor(MUTED)
        if row == 0:
            cell.set_text_props(color=ACCENT, fontweight="bold")
        elif col > 0 and row > 0:
            try:
                v = float(rows[row-1][col])
                pos_cols = {1, 2, 3, 4, 7, 8}
                neg_cols = {5}
                if col in pos_cols:
                    cell.set_text_props(color=GREEN if v > 0 else RED)
                elif col in neg_cols:
                    cell.set_text_props(color=RED if v < 0 else GREEN)
                else:
                    cell.set_text_props(color=TEXT)
            except (ValueError, IndexError):
                cell.set_text_props(color=TEXT)
        else:
            cell.set_text_props(color=TEXT)

    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved → {filename}")
    return filename
