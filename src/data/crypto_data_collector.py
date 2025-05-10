import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, Any

from ..config import (
    BINANCE_API_KEY,
    SYMBOL,
    INTERVAL,
    DATA_DIR
)

def fetch_ohlc_data(
    symbol: str = SYMBOL,
    start_date: str = None,
    end_date: str = None,
    interval: str = INTERVAL,
    limit: int = 1000
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Binance API
    
    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Time interval (1m, 5m, 15m, 30m, 1h, 1d)
        limit: Number of data points to retrieve per request
        
    Returns:
        DataFrame with columns: OpenTime, Open, High, Low, Close, Volume
    """
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    
    # Convert dates to timestamps
    if start_date:
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        start_ts = None
    
    if end_date:
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        end_ts = None
    
    # Initialize empty list for all data
    all_data = []
    
    # Set initial parameters
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    if start_ts:
        params["startTime"] = start_ts
    
    # Add API key to headers
    headers = {
        "X-MBX-APIKEY": BINANCE_API_KEY
    }
    
    try:
        while True:
            response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers)
            data = response.json()
            
            if isinstance(data, list):
                all_data.extend(data)
                
                # Update start time for next request
                if len(data) > 0:
                    last_timestamp = data[-1][0]
                    params["startTime"] = last_timestamp + 1
                else:
                    break
                
                # Check if we've reached the end date
                if end_ts and params["startTime"] >= end_ts:
                    break
                
                # Add delay to avoid rate limits
                time.sleep(0.1)
            else:
                print(f"Error fetching data for {symbol}: {data.get('msg', 'Unknown error')}")
                return None
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                "OpenTime", "Open", "High", "Low", "Close", "Volume",
                "CloseTime", "QuoteAssetVolume", "NumberOfTrades",
                "TakerBuyBaseAssetVolume", "TakerBuyQuoteAssetVolume", "Ignore"
            ])
            
            # Convert timestamp to datetime
            df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
            
            # Convert string values to float
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].astype(float)
            
            # Select and rename columns
            df = df[["OpenTime", "Open", "High", "Low", "Close", "Volume"]]
            
            # データを取得した後、インディケーターを計算
            from ..signals.entry_exit_point_generator import calculate_indicators
            df = calculate_indicators(df)
            
            # インディケーターを含むデータを保存
            file_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_data.csv")
            df.to_csv(file_path, index=False)
            
            print(f"Data for {symbol} saved to {file_path}")
            return df
        else:
            print(f"No data found for {symbol}")
            return None
            
    except Exception as e:
        print(f"Exception when fetching data for {symbol}: {str(e)}")
        return None

def get_latest_data(symbol: str = SYMBOL, interval: str = INTERVAL) -> Optional[pd.DataFrame]:
    """
    Get the latest OHLCV data for a symbol
    """
    return fetch_ohlc_data(symbol, interval=interval, limit=100)

if __name__ == "__main__":
    # Test data fetching
    df = get_latest_data()
    if df is not None:
        print("\nLatest data sample:")
        print(df.head()) 