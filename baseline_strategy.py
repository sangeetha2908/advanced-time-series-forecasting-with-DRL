import pandas as pd

def sma_strategy(df, fast=10, slow=50):
    df = df.copy()
    df['ma_fast'] = df['price'].rolling(fast).mean()
    df['ma_slow'] = df['price'].rolling(slow).mean()
    df['signal'] = (df['ma_fast'] > df['ma_slow']).astype(int)
    df['market_return'] = df['price'].pct_change()
    df['strategy_return'] = df['market_return'] * df['signal'].shift(1)
    df['equity_curve'] = (1 + df['strategy_return'].fillna(0)).cumprod()
    return df
