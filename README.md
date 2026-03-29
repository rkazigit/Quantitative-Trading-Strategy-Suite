# 📈 Quantitative Trading Strategy Suite

> **Dual-Class Arbitrage · Bollinger Bands · Reinforcement Learning · Sector Pairs**  
> Four production-grade algorithmic trading strategies with shared backtesting engine, realistic execution costs, and rich visualizations.

---

## Strategies

### 1. Dual-Class Arbitrage (GOOGL vs GOOG)

Exploits pricing discrepancies between share classes of the same company. Both represent identical economic ownership — any spread deviation is theoretically arbitrageable.

- **Pair**: GOOGL (Class A, voting) ↔ GOOG (Class C, non-voting)
- **Signal**: Rolling OLS hedge ratio → z-score of spread
- **Entry**: |z| > 2.0 σ · **Exit**: |z| < 0.5 σ · **Stop**: |z| > 4.0 σ
- **Analytics**: Engle-Granger cointegration test, ADF stationarity, OU half-life

### 2. Bollinger Band Strategy (SPY)

Three variants of the classic volatility band signal:

| Variant | Logic |
|---|---|
| **Mean Reversion** | Buy lower band, sell upper band — expect reversion to SMA |
| **Breakout** | Buy close above upper band (momentum), short below lower |
| **Squeeze** | Trade the expansion after BandWidth compression |

Features: RSI filter, %B oscillator, walk-forward parameter optimisation.

### 3. Reinforcement Learning Agent (QQQ)

Two RL approaches trained on 70% of data, evaluated on held-out 30%:

| Agent | Method |
|---|---|
| **Q-Learning** | Tabular, discrete state (RSI bin + %B bin + momentum), ε-greedy |
| **Deep Q-Network** | Neural network (64→128→64→3), experience replay, target network, Huber loss |

State features: 5/10/20/60-day returns, RSI-14, MACD histogram, Bollinger %B, BandWidth, volume ratio, ATR, current position.

Reward: `step_pnl / equity − λ × drawdown`

### 4. Sector-Based Pairs Trading

Systematic scanner across Technology, Financials, Energy, Healthcare, Industrials:

1. Test all within-sector pairs for cointegration (Engle-Granger, p < 0.05)
2. Estimate OU half-life — filter to [5, 60] days
3. Rank by composite score
4. Backtest top-5 pairs as equal-weight portfolio
5. Rolling OLS hedge ratio, z-score signals with hard stop

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/trading-strategy-suite.git
cd trading-strategy-suite

python -m venv venv
source venv/bin/activate    # Mac/Linux
# venv\Scripts\activate     # Windows

pip install -r requirements.txt

# Run all four strategies (≈ 5-10 minutes including RL training)
python main.py

# Individual strategies
python main.py --strategy arb     # Dual-class arbitrage only
python main.py --strategy bb      # Bollinger bands only
python main.py --strategy rl      # RL agent only (needs training time)
python main.py --strategy pairs   # Sector pairs only

# Options
python main.py --no-charts        # Skip chart generation
python main.py --capital 50000    # Custom initial capital
```

> **Note on PyTorch**: The DQN uses PyTorch if available. Without it, a NumPy fallback network runs instead. Install with `pip install torch` for full DQN functionality.

---

## Sample Output

```
═══════════════════════════════════════════════════════════════
  QUANTITATIVE TRADING STRATEGY SUITE
  Dual-Class Arb · Bollinger Bands · RL Agent · Sector Pairs
═══════════════════════════════════════════════════════════════

─────────────────────────────────────────────────────────────
  1️⃣  DUAL-CLASS ARBITRAGE  (GOOGL vs GOOG)
─────────────────────────────────────────────────────────────
  Pair Analysis:
  OLS Hedge Ratio:      1.0023
  Cointegration p-val:  0.0031 ✓
  ADF on spread p-val:  0.0187 ✓

  Results (4.2s):
    Total Return:.......... +18.43%
    Sharpe Ratio:.......... 1.241
    Max Drawdown:.......... -6.21%
    Win Rate:.............. 58.3%
    Profit Factor:......... 1.84

─────────────────────────────────────────────────────────────
  4️⃣  SECTOR-BASED PAIRS TRADING
─────────────────────────────────────────────────────────────
  Top 5 pairs selected:
  Pair                 Sector         p-val    HL(d)  Score
  ────────────────────────────────────────────────────────────
  MSFT/AAPL            Technology     0.0021   18.3   0.8731
  JPM/BAC              Financials     0.0044   22.1   0.8412
  XOM/CVX              Energy         0.0089   31.5   0.7893
  NVDA/AMD             Technology     0.0134   14.7   0.7651
  GS/MS                Financials     0.0201   26.8   0.7234

──────────────────────────────────────────────────────────────
  📊 STRATEGY COMPARISON SUMMARY
──────────────────────────────────────────────────────────────
  Strategy                       Return%   Sharpe   MaxDD%  WinRate%
  ──────────────────────────────────────────────────────────────────
  Dual-Class Arb                 +18.43%    1.241   -6.21%     58.3%
  BB Mean Reversion              +22.17%    0.987   -9.43%     54.1%
  Q-Learning                     +11.82%    0.734  -14.21%     51.8%
  Deep Q-Network                 +16.45%    0.891  -11.32%     53.2%
  Sector Pairs                   +24.91%    1.183   -8.67%     57.4%
```

---

## Charts Generated

| File | Contents |
|---|---|
| `equity_dashboard.png` | Equity curves + drawdowns + rolling Sharpe for all strategies |
| `monthly_returns.png` | Year × Month heatmaps for each strategy |
| `bollinger_signals.png` | Price, BB bands, %B oscillator, bandwidth compression |
| `sector_pairs_overview.png` | Spread + z-score for each top pair |
| `rl_training.png` | Q-Learning and DQN training reward curves |
| `comparison.png` | Bar chart + full metrics table |

---

## Architecture

```
trading_strategies/
├── backtesting/
│   ├── __init__.py
│   └── backtester.py      # Shared engine: execution, metrics, trade records
├── strategies/
│   ├── dual_class_arb.py  # GOOGL/GOOG spread, rolling OLS hedge, z-score
│   ├── bollinger_band.py  # 3 variants + RSI filter + walk-forward WFO
│   ├── rl_agent.py        # Q-Learning + DQN + feature engineering
│   └── sector_pairs.py    # Universe scan + cointegration + portfolio backtest
├── visualizations/
│   └── charts.py          # Dark-themed dashboard suite
├── main.py                # CLI orchestrator
└── requirements.txt
```

---

## Key Concepts

### Event-Driven Backtesting
The engine processes one bar at a time with no access to future data. Signals generated at bar N are executed at bar N+1 (avoiding look-ahead bias).

### Execution Realism
- **Slippage**: 5 bps per leg, applied adversely (buy at ask + slippage, sell at bid − slippage)
- **Commission**: 1 bps per leg + optional fixed fee
- **Position sizing**: Configurable fraction of current equity

### RL Train/Test Split
The RL agents are trained exclusively on the first 70% of data. Performance metrics reported are from the held-out 30% — a true out-of-sample evaluation.

### OU Half-Life Filter
For pairs trading, we only trade pairs with OU half-lives between 5 and 60 days. Too fast → excessive transaction costs. Too slow → capital tied up, opportunity cost.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical price data |
| `pandas / numpy` | Time series, matrix ops |
| `statsmodels` | OLS, cointegration tests, ADF |
| `matplotlib` | Charts and dashboards |
| `torch` | Deep Q-Network (optional, falls back to NumPy) |

---

## Disclaimer

Educational and research purposes only. Past backtested performance does not guarantee future results. Not financial advice.

---

## License
MIT
