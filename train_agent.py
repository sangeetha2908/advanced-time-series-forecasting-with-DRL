import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from trading_env import TradingEnv

SAVE_DIR = 'models'
os.makedirs(SAVE_DIR, exist_ok=True)
df = pd.read_csv('synthetic_data.csv')

def make_env():
    return Monitor(TradingEnv(df))

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

eval_env = Monitor(TradingEnv(df))

model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log='tb_logs', seed=42)
eval_callback = EvalCallback(eval_env, best_model_save_path=SAVE_DIR, log_path=SAVE_DIR, eval_freq=10000, n_eval_episodes=5)
model.learn(total_timesteps=200000, callback=eval_callback)
model.save(os.path.join(SAVE_DIR, 'ppo_final'))
vec_env.save(os.path.join(SAVE_DIR, 'vec_normalize.pkl'))
print('Training finished and saved.')
