import pandas as pd
import numpy as np
import ta
from typing import Optional

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for the given DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Make a copy of the input DataFrame
    df = df.copy()
    
    # Calculate Donchian Channels
    df['DonchianHigh'] = df['High'].rolling(window=20).max()
    df['DonchianLow'] = df['Low'].rolling(window=20).min()
    df['DonchianMiddle'] = (df['DonchianHigh'] + df['DonchianLow']) / 2
    
    # Calculate ADX
    adx = ta.trend.ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    )
    df['ADX'] = adx.adx()
    
    # Calculate ATR
    atr = ta.volatility.AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    )
    df['ATR'] = atr.average_true_range()
    
    # Calculate RSI
    df['RSI'] = ta.momentum.RSIIndicator(
        close=df['Close'],
        window=14
    ).rsi()
    
    # Calculate MACD
    macd = ta.trend.MACD(
        close=df['Close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['MACD'] = macd.macd()
    df['MACDSignal'] = macd.macd_signal()
    df['MACDHist'] = macd.macd_diff()
    
    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(
        close=df['Close'],
        window=20,
        window_dev=2
    )
    df['BBUpper'] = bollinger.bollinger_hband()
    df['BBMiddle'] = bollinger.bollinger_mavg()
    df['BBLower'] = bollinger.bollinger_lband()
    
    # Calculate Moving Averages
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # Calculate Stochastics
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['StochK'] = stoch.stoch()
    df['StochD'] = stoch.stoch_signal()
    
    # Calculate OBV
    df['OBV'] = ta.volume.on_balance_volume(
        close=df['Close'],
        volume=df['Volume']
    )
    
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on technical indicators
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with added signal columns
    """
    # Make a copy of the input DataFrame
    df = df.copy()
    
    # Initialize signal columns
    df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
    
    # Generate signals based on Donchian breakout and ADX
    for i in range(1, len(df)):
        # Check for bullish breakout
        if (df['Close'].iloc[i] > df['DonchianHigh'].iloc[i-1] and
            df['Close'].iloc[i-1] <= df['DonchianHigh'].iloc[i-1] and
            df['ADX'].iloc[i] >= 25):
            df['Signal'].iloc[i] = 1
        
        # Check for bearish breakout
        elif (df['Close'].iloc[i] < df['DonchianLow'].iloc[i-1] and
              df['Close'].iloc[i-1] >= df['DonchianLow'].iloc[i-1] and
              df['ADX'].iloc[i] >= 25):
            df['Signal'].iloc[i] = -1
    
    return df

if __name__ == '__main__':
    # Test indicator calculation
    import yfinance as yf
    
    # Download sample data
    data = yf.download('BTC-USD', start='2023-01-01', end='2023-12-31', interval='1h')
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Generate signals
    data = generate_signals(data)
    
    # Print results
    print("\nSample data with indicators:")
    print(data.tail()) 