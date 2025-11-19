from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('synthetic_prices.csv')
    env = DummyVecEnv([lambda: TradingEnv(df, window_size=10)])
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=[128, 64]),
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        batch_size=64
    )
    model.learn(total_timesteps=200000)
    model.save('ppo_trading')
    print('Training complete.')
