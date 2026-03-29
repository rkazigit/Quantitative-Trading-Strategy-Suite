"""
main.py — Trading Strategy Suite — Master Runner
══════════════════════════════════════════════════

Runs all four strategies sequentially and generates a comprehensive report:
  1. Dual-Class Arbitrage     (GOOGL vs GOOG)
  2. Bollinger Band Strategy  (SPY, 3 variants)
  3. Reinforcement Learning   (QQQ, Q-Learning + DQN)
  4. Sector-Based Pairs       (Tech/Fin/Energy universe)

Usage:
    python main.py                     # Run all strategies
    python main.py --strategy bb       # Bollinger only
    python main.py --strategy arb      # Dual class only
    python main.py --strategy rl       # RL only
    python main.py --strategy pairs    # Sector pairs only
    python main.py --no-charts         # Skip visualisations
    python main.py --capital 50000     # Custom initial capital
"""

import argparse
import sys
import time
import os
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from backtesting.backtester import BacktestConfig


# ─── ANSI colours ─────────────────────────────────────────────────────────
def c(t, code): return f"\033[{code}m{t}\033[0m"
def cyan(t):   return c(t, "96")
def green(t):  return c(t, "92")
def yellow(t): return c(t, "93")
def red(t):    return c(t, "91")
def bold(t):   return c(t, "1")
def dim(t):    return c(t, "2")
def magenta(t):return c(t, "95")


def banner():
    print("\n" + cyan("═" * 65))
    print(f"  {bold('QUANTITATIVE TRADING STRATEGY SUITE')}")
    print(f"  {dim('Dual-Class Arb · Bollinger Bands · RL Agent · Sector Pairs')}")
    print(cyan("═" * 65) + "\n")


def section(title: str, emoji: str = "▶"):
    print(f"\n{yellow('─' * 65)}")
    print(f"  {emoji}  {bold(title)}")
    print(yellow("─" * 65))


def print_metrics(metrics: dict):
    items = [
        ("Total Return",  f"{metrics.get('Total Return (%)', 0):+.2f}%",
         metrics.get('Total Return (%)', 0) > 0),
        ("CAGR",          f"{metrics.get('CAGR (%)', 0):+.2f}%",
         metrics.get('CAGR (%)', 0) > 0),
        ("Sharpe Ratio",  f"{metrics.get('Sharpe Ratio', 0):.3f}",
         metrics.get('Sharpe Ratio', 0) > 1),
        ("Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.3f}", None),
        ("Max Drawdown",  f"{metrics.get('Max Drawdown (%)', 0):.2f}%", False),
        ("Win Rate",      f"{metrics.get('Win Rate (%)', 0):.1f}%",
         metrics.get('Win Rate (%)', 0) > 50),
        ("Profit Factor", f"{metrics.get('Profit Factor', 0):.3f}",
         metrics.get('Profit Factor', 0) > 1),
        ("Total Trades",  str(metrics.get('Total Trades', 0)), None),
    ]
    for label, val, good in items:
        if good is True:   clr = green
        elif good is False: clr = red
        else:               clr = dim
        print(f"    {dim(label+':'+'.'*(22-len(label)))} {clr(val)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="all",
                   choices=["all", "arb", "bb", "rl", "pairs"])
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--capital",   type=float, default=100_000)
    return p.parse_args()


def main():
    args   = parse_args()
    banner()

    config = BacktestConfig(
        initial_capital  = args.capital,
        slippage_bps     = 5.0,
        commission_bps   = 1.0,
        risk_free_rate   = 0.05,
    )

    all_equities = {}
    all_trades   = {}
    all_metrics  = []
    chart_data   = {}

    run_all = args.strategy == "all"

    # ══════════════════════════════════════════════════════════════════════
    # 1. DUAL-CLASS ARBITRAGE
    # ══════════════════════════════════════════════════════════════════════
    if run_all or args.strategy == "arb":
        section("DUAL-CLASS ARBITRAGE  (GOOGL vs GOOG)", "1️⃣")
        from strategies.dual_class_arb import run as run_arb
        t0  = time.time()
        arb = run_arb(config)
        print(f"\n  Results ({time.time()-t0:.1f}s):")
        print_metrics(arb["metrics"])

        name = "Dual-Class Arb"
        all_equities[name] = arb["equity"]
        all_trades[name]   = arb["trades"]
        all_metrics.append(arb["metrics"])
        chart_data["arb"]  = arb

    # ══════════════════════════════════════════════════════════════════════
    # 2. BOLLINGER BAND STRATEGY
    # ══════════════════════════════════════════════════════════════════════
    if run_all or args.strategy == "bb":
        section("BOLLINGER BAND STRATEGY  (SPY)", "2️⃣")
        from strategies.bollinger_band import run as run_bb
        t0 = time.time()
        bb = run_bb(config)
        print(f"\n  Results ({time.time()-t0:.1f}s):")

        for variant, res in bb.items():
            name = f"BB {variant.replace('_', ' ').title()}"
            print(f"\n  {cyan(name)}:")
            print_metrics(res["metrics"])
            if variant == "mean_reversion":  # Use one variant for master comparison
                all_equities[name] = res["equity"]
                all_trades[name]   = res["trades"]
                all_metrics.append(res["metrics"])

        chart_data["bb"] = bb

    # ══════════════════════════════════════════════════════════════════════
    # 3. REINFORCEMENT LEARNING
    # ══════════════════════════════════════════════════════════════════════
    if run_all or args.strategy == "rl":
        section("REINFORCEMENT LEARNING AGENT  (QQQ)", "3️⃣")
        print(f"  {dim('Training Q-Learning and DQN agents (may take 2-3 min)...')}")
        from strategies.rl_agent import run as run_rl
        t0 = time.time()
        rl = run_rl(config)
        print(f"\n  Results ({time.time()-t0:.1f}s):")

        for key, label in [("qlearning", "Q-Learning"), ("dqn", "Deep Q-Network")]:
            if key in rl:
                print(f"\n  {cyan(label)} (out-of-sample):")
                print_metrics(rl[key]["metrics"])
                all_equities[label] = rl[key]["equity"]
                all_trades[label]   = rl[key]["trades"]
                all_metrics.append(rl[key]["metrics"])

        chart_data["rl"] = rl

    # ══════════════════════════════════════════════════════════════════════
    # 4. SECTOR-BASED PAIRS TRADING
    # ══════════════════════════════════════════════════════════════════════
    if run_all or args.strategy == "pairs":
        section("SECTOR-BASED PAIRS TRADING", "4️⃣")
        from strategies.sector_pairs import run as run_pairs
        t0    = time.time()
        pairs = run_pairs(config)
        print(f"\n  Results ({time.time()-t0:.1f}s):")
        print_metrics(pairs["metrics"])

        name = "Sector Pairs"
        all_equities[name] = pairs["equity"]
        all_trades[name]   = pairs["trades"]
        all_metrics.append(pairs["metrics"])
        chart_data["pairs"] = pairs

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    if len(all_metrics) > 1:
        section("STRATEGY COMPARISON SUMMARY", "📊")
        print(f"\n  {'Strategy':<30} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>9} {'WinRate%':>10}")
        print(f"  {'─'*62}")
        for m in all_metrics:
            tr = m.get('Total Return (%)', 0)
            sh = m.get('Sharpe Ratio', 0)
            dd = m.get('Max Drawdown (%)', 0)
            wr = m.get('Win Rate (%)', 0)
            tr_s = green(f"{tr:+.2f}%") if tr > 0 else red(f"{tr:+.2f}%")
            print(f"  {m['Strategy']:<30} {tr_s:>9}  {sh:>7.3f}  {dd:>8.2f}%  {wr:>8.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════
    if not args.no_charts and all_equities:
        section("GENERATING CHARTS", "📈")
        from visualizations.charts import (
            plot_equity_dashboard, plot_monthly_returns,
            plot_bollinger_signals, plot_sector_pairs_overview,
            plot_rl_training, plot_comparison_radar,
        )

        print("  Building equity dashboard...")
        plot_equity_dashboard(all_equities, all_trades, config)

        print("  Building monthly returns heatmap...")
        plot_monthly_returns(all_equities)

        if "bb" in chart_data:
            print("  Building Bollinger signal chart...")
            mr_signals = chart_data["bb"]["mean_reversion"]["signals"]
            plot_bollinger_signals(mr_signals)

        if "pairs" in chart_data and chart_data["pairs"].get("pair_signals"):
            print("  Building sector pairs chart...")
            plot_sector_pairs_overview(
                chart_data["pairs"]["pairs"],
                chart_data["pairs"]["pair_signals"],
            )

        if "rl" in chart_data:
            print("  Building RL training curves...")
            ql_rets  = chart_data["rl"].get("qlearning", {}).get("train_returns", [])
            dqn_rets = chart_data["rl"].get("dqn", {}).get("train_returns", [])
            if ql_rets or dqn_rets:
                plot_rl_training(ql_rets, dqn_rets)

        if len(all_metrics) > 1:
            print("  Building comparison chart...")
            plot_comparison_radar(all_metrics)

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{cyan('═' * 65)}")
    print(f"  {bold('All strategies complete.')}")
    print(f"  Charts saved as PNG files in the current directory.")
    print(cyan("═" * 65) + "\n")


if __name__ == "__main__":
    main()
