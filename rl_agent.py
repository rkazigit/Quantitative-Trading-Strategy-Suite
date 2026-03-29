"""
rl_agent.py — Reinforcement Learning Trading Agent
════════════════════════════════════════════════════

Implements two RL approaches:

1. Tabular Q-Learning (fast, interpretable baseline)
   - State: discretised (RSI bin, BB %B bin, momentum bin)
   - Actions: {0: hold, 1: buy, 2: sell}
   - Reward: risk-adjusted daily P&L

2. Deep Q-Network (DQN) with Experience Replay
   - State: raw feature vector (returns, RSI, MACD, volume ratio, etc.)
   - Neural network: 64 → 128 → 64 → 3 (PyTorch if available, else NumPy)
   - Target network for stability
   - ε-greedy exploration with exponential decay

Training protocol:
   • Train on 70% of data (in-sample)
   • Evaluate on 30% (out-of-sample) — no snooping
   • Episode = one complete pass through training data
   • Reward = (portfolio return) − (λ × drawdown penalty)

State features:
   [5-day return, 10-day return, 20-day return,
    RSI-14, MACD, MACD signal, %B, BandWidth,
    volume ratio (vs 20-day avg), ATR ratio,
    current position (-1/0/1)]
"""

from __future__ import annotations

import warnings
import random
import collections
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from backtesting.backtester import (
    BacktestConfig, Trade, compute_metrics
)

# Try to import PyTorch for DQN; fall back to NumPy network
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─── Config ───────────────────────────────────────────────────────────────

class RLConfig:
    ticker:         str   = "QQQ"
    start:          str   = "2018-01-01"
    end:            str   = "2024-12-31"
    train_pct:      float = 0.70       # train/test split
    # Q-Learning
    alpha:          float = 0.10       # learning rate
    gamma:          float = 0.95       # discount factor
    epsilon_start:  float = 1.00       # exploration start
    epsilon_end:    float = 0.01       # exploration end
    epsilon_decay:  float = 0.995
    n_episodes:     int   = 50
    # DQN
    hidden_size:    int   = 128
    batch_size:     int   = 64
    memory_size:    int   = 10_000
    target_update:  int   = 10         # steps between target net sync
    lr_dqn:         float = 1e-3
    n_dqn_episodes: int   = 30
    # Reward shaping
    drawdown_penalty:float = 0.1


# ─── Feature engineering ──────────────────────────────────────────────────

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute rich feature set for the RL state space."""
    c  = data["Close"]
    v  = data["Volume"] if "Volume" in data.columns else pd.Series(1, index=data.index)

    ret   = c.pct_change()
    feat  = pd.DataFrame(index=data.index)

    # Price momentum
    for w in [5, 10, 20, 60]:
        feat[f"ret_{w}d"] = c.pct_change(w)

    # RSI
    delta  = c.diff()
    up     = delta.clip(lower=0)
    dn     = (-delta).clip(lower=0)
    avg_u  = up.ewm(alpha=1/14, min_periods=14).mean()
    avg_d  = dn.ewm(alpha=1/14, min_periods=14).mean()
    feat["rsi"] = 100 - 100 / (1 + avg_u / avg_d.replace(0, np.nan))

    # MACD
    ema12          = c.ewm(span=12, min_periods=12).mean()
    ema26          = c.ewm(span=26, min_periods=26).mean()
    feat["macd"]   = (ema12 - ema26) / c
    feat["macd_sig"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_sig"]

    # Bollinger %B and BandWidth
    sma20           = c.rolling(20).mean()
    std20           = c.rolling(20).std(ddof=0)
    feat["pct_b"]   = (c - (sma20 - 2*std20)) / (4 * std20 + 1e-8)
    feat["bw"]      = (4 * std20) / sma20

    # Volatility (ATR proxy)
    feat["vol_20d"] = ret.rolling(20).std()

    # Volume
    vol_ma          = v.rolling(20).mean()
    feat["vol_ratio"] = v / (vol_ma + 1e-8)

    # Price distance from moving averages
    feat["dist_ma20"] = (c - sma20) / sma20
    feat["dist_ma50"] = (c - c.rolling(50).mean()) / c.rolling(50).mean()

    feat["position"] = 0.0   # will be filled during simulation

    return feat.dropna()


def discretise_state(feat_row: pd.Series, n_bins: int = 5) -> Tuple:
    """Map continuous features to discrete state tuple for Q-table."""
    def binn(val, lo=-1, hi=1):
        val = float(np.clip(val, lo, hi))
        return int((val - lo) / (hi - lo) * (n_bins - 1))

    rsi_norm = (float(feat_row.get("rsi", 50)) - 50) / 50
    return (
        binn(float(feat_row.get("ret_5d",  0)), -0.1, 0.1),
        binn(rsi_norm,                           -1.0, 1.0),
        binn(float(feat_row.get("pct_b",   0.5)), 0.0, 1.0),
        binn(float(feat_row.get("macd",    0)),  -0.05, 0.05),
        int(float(feat_row.get("position", 0)) + 1),  # -1,0,1 → 0,1,2
    )


# ─── Q-Learning Agent ─────────────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-Learning with ε-greedy exploration.
    Q-table: state_tuple → [hold, long, short]
    """
    name = "Q-Learning"

    def __init__(self, cfg: RLConfig):
        self.cfg     = cfg
        self.q_table : Dict[Tuple, np.ndarray] = {}
        self.epsilon = cfg.epsilon_start

    def _q(self, state: Tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        return self.q_table[state]

    def choose_action(self, state: Tuple, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(np.argmax(self._q(state)))

    def update(self, s, a, r, s2, done):
        q_sa   = self._q(s)[a]
        q_next = 0 if done else np.max(self._q(s2))
        self._q(s)[a] = q_sa + self.cfg.alpha * (
            r + self.cfg.gamma * q_next - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end,
                           self.epsilon * self.cfg.epsilon_decay)


# ─── DQN components ───────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class QNet(nn.Module):
        def __init__(self, n_input: int, n_hidden: int, n_actions: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_input, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(n_hidden, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden // 2),
                nn.ReLU(),
                nn.Linear(n_hidden // 2, n_actions),
            )
        def forward(self, x):
            return self.net(x)


class NumpyQNet:
    """Lightweight 2-layer network for environments without PyTorch."""
    def __init__(self, n_in: int, n_h: int, n_out: int = 3):
        lim1 = np.sqrt(2 / n_in)
        lim2 = np.sqrt(2 / n_h)
        self.W1 = np.random.randn(n_in, n_h) * lim1
        self.b1 = np.zeros(n_h)
        self.W2 = np.random.randn(n_h, n_out) * lim2
        self.b2 = np.zeros(n_out)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class DQNAgent:
    """
    Deep Q-Network with:
    • Experience replay buffer
    • Target network (soft update every target_update steps)
    • Huber loss
    """
    name = "Deep Q-Network"

    def __init__(self, n_features: int, cfg: RLConfig):
        self.cfg      = cfg
        self.epsilon  = cfg.epsilon_start
        self.memory   = collections.deque(maxlen=cfg.memory_size)
        self.step_cnt = 0
        self.n_feat   = n_features

        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.policy_net = QNet(n_features, cfg.hidden_size).to(self.device)
            self.target_net = QNet(n_features, cfg.hidden_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.AdamW(
                self.policy_net.parameters(), lr=cfg.lr_dqn, weight_decay=1e-4)
            self.loss_fn = nn.HuberLoss()
        else:
            self.policy_net = NumpyQNet(n_features, cfg.hidden_size)
            self.target_net = NumpyQNet(n_features, cfg.hidden_size)

    def _to_tensor(self, x):
        if TORCH_AVAILABLE:
            return torch.FloatTensor(x).to(self.device)
        return np.array(x, dtype=np.float32)

    def choose_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                q = self.policy_net(self._to_tensor(state).unsqueeze(0))
            return int(q.argmax().item())
        else:
            q = self.policy_net.predict(state.reshape(1, -1))
            return int(np.argmax(q))

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.cfg.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        if TORCH_AVAILABLE:
            s  = self._to_tensor(np.array(states))
            a  = torch.LongTensor(actions).to(self.device)
            r  = torch.FloatTensor(rewards).to(self.device)
            s2 = self._to_tensor(np.array(next_states))
            d  = torch.FloatTensor(dones).to(self.device)

            current_q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = self.target_net(s2).max(1)[0]
            target_q = r + self.cfg.gamma * next_q * (1 - d)

            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            return float(loss.item())
        else:
            # NumPy gradient descent
            states_arr = np.array(states)
            q_vals = self.policy_net.predict(states_arr)
            next_q = np.max(self.target_net.predict(np.array(next_states)), axis=1)
            targets = np.array(rewards) + self.cfg.gamma * next_q * (1 - np.array(dones))
            for i, a in enumerate(actions):
                q_vals[i, a] = targets[i]
            # Simple SGD step
            lr = self.cfg.lr_dqn
            h  = np.maximum(states_arr @ self.policy_net.W1 + self.policy_net.b1, 0)
            pred = h @ self.policy_net.W2 + self.policy_net.b2
            err  = pred - q_vals
            dW2  = h.T @ err / len(batch)
            db2  = err.mean(axis=0)
            self.policy_net.W2 -= lr * dW2
            self.policy_net.b2 -= lr * db2
            return float(np.mean(err ** 2))

    def update_target(self):
        if TORCH_AVAILABLE:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.target_net.W1 = self.policy_net.W1.copy()
            self.target_net.W2 = self.policy_net.W2.copy()

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end,
                           self.epsilon * self.cfg.epsilon_decay)


# ─── Training & simulation environment ───────────────────────────────────

class TradingEnv:
    """
    Lightweight gym-like environment.
    State: normalised feature vector + current position.
    Reward: step P&L − drawdown penalty.
    """

    def __init__(self, features: pd.DataFrame, prices: pd.Series,
                 cfg: RLConfig, initial_capital: float = 100_000):
        self.feat    = features.values.astype(np.float32)
        self.prices  = prices.values
        self.cfg     = cfg
        self.capital = initial_capital
        self.n_feat  = features.shape[1]
        self.reset()

    def reset(self):
        self.idx      = 0
        self.position = 0      # -1, 0, 1
        self.equity   = self.capital
        self.peak     = self.capital
        self.equity_history = [self.capital]
        return self._state()

    def _state(self) -> np.ndarray:
        f = self.feat[self.idx].copy()
        f[-1] = float(self.position)  # last feature = position
        return f

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """action: 0=hold, 1=long, 2=short"""
        if self.idx >= len(self.prices) - 2:
            return self._state(), 0.0, True

        px_now  = self.prices[self.idx]
        px_next = self.prices[self.idx + 1]
        ret     = (px_next - px_now) / px_now

        # Convert action to position
        new_pos = {0: self.position, 1: 1, 2: -1}[action]

        # Transaction cost if position changed
        tc = 0.001 if new_pos != self.position else 0.0  # 10 bps round-trip

        self.position = new_pos
        pnl = self.position * ret * self.equity - tc * self.equity
        self.equity += pnl
        self.peak    = max(self.peak, self.equity)
        dd           = (self.peak - self.equity) / self.peak

        reward = (pnl / self.equity) - self.cfg.drawdown_penalty * dd

        self.equity_history.append(self.equity)
        self.idx += 1
        done = self.idx >= len(self.prices) - 1
        return self._state(), float(reward), done


def train_q_learning(
    env: TradingEnv, agent: QLearningAgent, cfg: RLConfig
) -> List[float]:
    episode_returns = []
    for ep in range(cfg.n_episodes):
        state_vec = env.reset()
        state     = discretise_state(
            pd.Series(state_vec, index=[f"f{i}" for i in range(len(state_vec))]))
        done      = False
        ep_return = 0.0
        while not done:
            action = agent.choose_action(state, explore=True)
            next_vec, reward, done = env.step(action)
            next_state = discretise_state(
                pd.Series(next_vec, index=[f"f{i}" for i in range(len(next_vec))]))
            agent.update(state, action, reward, next_state, done)
            state     = next_state
            ep_return += reward
        agent.decay_epsilon()
        episode_returns.append(ep_return)
        if (ep + 1) % 10 == 0:
            print(f"    Q-Learning Ep {ep+1:3d}/{cfg.n_episodes}  "
                  f"ε={agent.epsilon:.3f}  total_reward={ep_return:.4f}")
    return episode_returns


def train_dqn(
    env: TradingEnv, agent: DQNAgent, cfg: RLConfig
) -> List[float]:
    episode_returns = []
    step = 0
    for ep in range(cfg.n_dqn_episodes):
        state = env.reset()
        done  = False
        ep_return = 0.0
        while not done:
            action  = agent.choose_action(state, explore=True)
            ns, r, done = env.step(action)
            agent.push(state, action, r, ns, float(done))
            loss    = agent.learn()
            state   = ns
            ep_return += r
            step += 1
            if step % cfg.target_update == 0:
                agent.update_target()
        agent.decay_epsilon()
        episode_returns.append(ep_return)
        if (ep + 1) % 10 == 0:
            print(f"    DQN Ep {ep+1:3d}/{cfg.n_dqn_episodes}  "
                  f"ε={agent.epsilon:.3f}  total_reward={ep_return:.4f}")
    return episode_returns


def evaluate_agent(env: TradingEnv, agent, agent_type: str,
                   index: pd.Index) -> Tuple[pd.Series, List[Trade]]:
    """Run greedy policy and collect equity + trades."""
    state = env.reset()
    done  = False
    pos   = 0
    trades: List[Trade] = []
    entry_px   = 0.0
    entry_idx  = 0
    entry_side = ""

    while not done:
        if agent_type == "qlearning":
            disc_s = discretise_state(
                pd.Series(state, index=[f"f{i}" for i in range(len(state))]))
            action = agent.choose_action(disc_s, explore=False)
        else:
            action = agent.choose_action(state, explore=False)

        new_pos = {0: pos, 1: 1, 2: -1}[action]

        # Log trade
        if new_pos != pos and pos != 0:
            px_exit = env.prices[env.idx]
            pnl = pos * (px_exit - entry_px) / entry_px * env.equity * 0.5
            trades.append(Trade(
                entry_date  = index[entry_idx],
                exit_date   = index[min(env.idx, len(index)-1)],
                side        = entry_side,
                entry_price = entry_px,
                exit_price  = px_exit,
                shares      = env.equity * 0.5 / max(entry_px, 1),
                pnl         = pnl,
                pnl_pct     = pos * (px_exit / entry_px - 1) * 100,
                commission  = env.equity * 0.001,
                strategy    = agent_type,
            ))
        if new_pos != 0 and pos == 0:
            entry_px   = env.prices[env.idx]
            entry_idx  = env.idx
            entry_side = "long" if new_pos == 1 else "short"

        pos = new_pos
        _, _, done = env.step(action)

    equity = pd.Series(
        env.equity_history[:len(index)],
        index=index[:len(env.equity_history)],
        name=agent_type,
    )
    return equity, trades


# ─── Main runner ──────────────────────────────────────────────────────────

def load_data(cfg: RLConfig) -> pd.DataFrame:
    print(f"  Fetching {cfg.ticker}...")
    data = yf.download(cfg.ticker, start=cfg.start, end=cfg.end,
                       auto_adjust=True, progress=False)
    print(f"  Loaded {len(data)} rows")
    return data


def run(bt_config: BacktestConfig = BacktestConfig()) -> Dict:
    cfg  = RLConfig()
    data = load_data(cfg)

    feat  = build_features(data)
    prices_aligned = data["Close"].loc[feat.index]

    # Train / test split
    split = int(len(feat) * cfg.train_pct)
    feat_train  = feat.iloc[:split]
    feat_test   = feat.iloc[split:]
    px_train    = prices_aligned.iloc[:split]
    px_test     = prices_aligned.iloc[split:]

    print(f"\n  Train: {feat_train.index[0].date()} → {feat_train.index[-1].date()} ({len(feat_train)}d)")
    print(f"  Test:  {feat_test.index[0].date()}  → {feat_test.index[-1].date()}  ({len(feat_test)}d)")

    results = {}

    # ── Q-Learning ────────────────────────────────────────────────────────
    print("\n  ── Training Q-Learning agent...")
    env_train = TradingEnv(feat_train, px_train.values, cfg, bt_config.initial_capital)
    q_agent   = QLearningAgent(cfg)
    q_returns = train_q_learning(env_train, q_agent, cfg)

    print("\n  ── Evaluating Q-Learning (out-of-sample)...")
    env_test  = TradingEnv(feat_test, px_test.values, cfg, bt_config.initial_capital)
    q_equity, q_trades = evaluate_agent(env_test, q_agent, "qlearning", feat_test.index)
    q_metrics = compute_metrics(q_equity, q_trades, bt_config, "Q-Learning")
    results["qlearning"] = {
        "equity": q_equity, "trades": q_trades, "metrics": q_metrics,
        "train_returns": q_returns, "agent": q_agent,
        "feat_test": feat_test, "px_test": px_test,
    }

    # ── DQN ───────────────────────────────────────────────────────────────
    print("\n  ── Training DQN agent...")
    env_train2 = TradingEnv(feat_train, px_train.values, cfg, bt_config.initial_capital)
    dqn_agent  = DQNAgent(feat_train.shape[1], cfg)
    dqn_returns = train_dqn(env_train2, dqn_agent, cfg)

    print("\n  ── Evaluating DQN (out-of-sample)...")
    env_test2 = TradingEnv(feat_test, px_test.values, cfg, bt_config.initial_capital)
    dqn_equity, dqn_trades = evaluate_agent(env_test2, dqn_agent, "dqn", feat_test.index)
    dqn_metrics = compute_metrics(dqn_equity, dqn_trades, bt_config, "Deep Q-Network")
    results["dqn"] = {
        "equity": dqn_equity, "trades": dqn_trades, "metrics": dqn_metrics,
        "train_returns": dqn_returns, "agent": dqn_agent,
    }

    # Buy-and-hold benchmark
    bh_equity = pd.Series(
        bt_config.initial_capital * px_test / px_test.iloc[0],
        name="Buy & Hold"
    )
    results["benchmark"] = {
        "equity": bh_equity,
        "metrics": compute_metrics(bh_equity, [], bt_config, "Buy & Hold"),
    }

    return results
