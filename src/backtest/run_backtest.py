import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from .backtest_engine import BacktestEngine
from ..data.crypto_data_collector import fetch_ohlc_data
from ..signals.entry_exit_point_generator import calculate_indicators

def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1h',
    initial_balance: float = 10000.0,
    position_size_percent: float = 0.02,
    sl_multiplier: float = 2.0,
    tp_multiplier: float = 3.0,
    save_results: bool = True
) -> dict:
    """
    Run backtest for a given symbol and date range
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Candlestick interval
        initial_balance: Initial account balance
        position_size_percent: Position size as percentage of balance
        sl_multiplier: Stop loss multiplier for ATR
        tp_multiplier: Take profit multiplier for ATR
        save_results: Whether to save results to file
    
    Returns:
        Dictionary with backtest results
    """
    # Fetch historical data
    data = fetch_ohlc_data(symbol, start_date, end_date, interval)
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Initialize and run backtest
    engine = BacktestEngine(
        data=data,
        initial_balance=initial_balance,
        position_size_percent=position_size_percent,
        sl_multiplier=sl_multiplier,
        tp_multiplier=tp_multiplier
    )
    
    results = engine.run()
    
    # Save results if requested
    if save_results:
        # Create results directory if it doesn't exist
        results_dir = Path('results/backtest')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{interval}_{timestamp}"
        
        # Save results to JSON
        results_file = results_dir / f"{filename}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save plot
        plot_file = results_dir / f"{filename}.png"
        engine.plot_results(save_path=str(plot_file))
        
        print(f"Results saved to {results_dir}")
    
    return results

if __name__ == '__main__':
    # Example usage
    results = run_backtest(
        symbol='BTCUSDT',
        start_date='2023-01-01',
        end_date='2023-12-31',
        interval='1h'
    )
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}") 