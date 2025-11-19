import gym
import numpy as np

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_steps = len(df) - 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
    def _get_state(self):
        row = self.df.iloc[self.current_step]
        return np.array([row.price, row.return, row.ma_fast, row.ma_slow, row.rsi, self.shares_held], dtype=np.float32)
    def step(self, action):
        price = self.df.iloc[self.current_step].price
        prev_worth = self.net_worth
        # BUY
        if action == 1 and self.balance >= price:
            self.shares_held += 1
            self.balance -= price
        # SELL
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += price
        self.net_worth = self.balance + self.shares_held * price
        reward = self.net_worth - prev_worth
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done, {}
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._get_state()
    def render(self, mode='human'):
        print(f'Step: {self.current_step} Price: {self.df.iloc[self.current_step].price} NetWorth: {self.net_worth}')
