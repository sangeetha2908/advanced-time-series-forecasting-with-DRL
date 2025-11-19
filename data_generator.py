import numpy as np
import pandas as pd

def generate_synthetic_data(n_steps=5000, seed=42):
    np.random.seed(seed)
    prices = [100]
    vol = 0.015
    long_term_mean = 100
    mean_revert_strength = 0.02
    for _ in range(n_steps):
        shock = np.random.randn() * vol
        mean_revert = mean_revert_strength * (long_term_mean - prices[-1])
        new_price = prices[-1] + mean_revert + shock
        prices.append(new_price)
    df = pd.DataFrame({'price': prices})
    df['return'] = df['price'].pct_change()
    df['ma_fast'] = df['price'].rolling(10).mean()
    df['ma_slow'] = df['price'].rolling(50).mean()
    df['rsi'] = 100 - (100 / (1 + df['return'].rolling(14).mean()))
    df = df.dropna()
    return df

if __name__ == '__main__':
    df = generate_synthetic_data()
    df.to_csv('synthetic_data.csv', index=False)
    print('Saved synthetic_data.csv', df.shape)
