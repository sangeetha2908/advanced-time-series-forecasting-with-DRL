import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv
from baseline_strategy import sma_strategy

def sharpe(returns):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0.0
    return np.sqrt(252) * returns.mean() / returns.std()

def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()

df = pd.read_csv('synthetic_data.csv')

# Baseline
base = sma_strategy(df)
base_equity = base['equity_curve'].dropna()
base_returns = base_equity.pct_change().dropna()
print('Baseline Sharpe:', sharpe(base_returns))
print('Baseline Total Return:', base_equity.iloc[-1] - 1)

# PPO (if trained)
try:
    model = PPO.load('models/ppo_final')
    env = TradingEnv(df)
    obs = env.reset()
    equity = [env.net_worth]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(int(action))
        equity.append(env.net_worth)
    returns = np.diff(equity) / equity[:-1]
    print('PPO Sharpe:', sharpe(returns))
    print('PPO Total Return:', equity[-1] / 10000 - 1)
except Exception as e:
    print('PPO evaluation skipped (model missing).', e)
