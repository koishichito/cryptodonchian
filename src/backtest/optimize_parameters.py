import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from datetime import datetime

from .backtest_engine import BacktestEngine
from ..data.crypto_data_collector import fetch_ohlc_data
from ..signals.entry_exit_point_generator import calculate_indicators

def optimize_parameters(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1h',
    initial_balance: float = 10000.0,
    param_grid: Dict = None,
    n_jobs: int = -1,
    save_results: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimize backtest parameters using grid search
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Candlestick interval
        initial_balance: Initial account balance
        param_grid: Dictionary of parameters to optimize
        n_jobs: Number of parallel jobs (-1 for all cores)
        save_results: Whether to save results to file
    
    Returns:
        Tuple of (best parameters, results DataFrame)
    """
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'position_size_percent': [0.01, 0.02, 0.03, 0.04, 0.05],
            'sl_multiplier': [1.5, 2.0, 2.5, 3.0],
            'tp_multiplier': [2.0, 3.0, 4.0, 5.0]
        }
    
    # Fetch historical data
    data = fetch_ohlc_data(symbol, start_date, end_date, interval)
    data = calculate_indicators(data)
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Function to run single backtest
    def run_single_backtest(params):
        engine = BacktestEngine(
            data=data,
            initial_balance=initial_balance,
            **dict(zip(param_names, params))
        )
        results = engine.run()
        return dict(zip(param_names, params)), results
    
    # Run backtests in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(run_single_backtest, param_combinations))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {**params, **metrics}
        for params, metrics in results
    ])
    
    # Calculate composite score
    results_df['score'] = (
        results_df['total_return'] * 0.4 +
        results_df['sharpe_ratio'] * 0.3 +
        results_df['win_rate'] * 0.2 +
        (1 - results_df['max_drawdown']) * 0.1
    )
    
    # Find best parameters
    best_idx = results_df['score'].idxmax()
    best_params = results_df.iloc[best_idx].to_dict()
    
    # Save results if requested
    if save_results:
        # Create results directory if it doesn't exist
        results_dir = Path('results/optimization')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{interval}_{timestamp}"
        
        # Save results to CSV
        results_file = results_dir / f"{filename}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Save best parameters to JSON
        params_file = results_dir / f"{filename}_best_params.json"
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print(f"Results saved to {results_dir}")
    
    return best_params, results_df

if __name__ == '__main__':
    # Example usage
    best_params, results_df = optimize_parameters(
        symbol='BTCUSDT',
        start_date='2023-01-01',
        end_date='2023-12-31',
        interval='1h'
    )
    
    # Print best parameters
    print("\nBest Parameters:")
    for param, value in best_params.items():
        if param != 'score':
            print(f"{param}: {value}")
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {best_params['total_return']:.2%}")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Win Rate: {best_params['win_rate']:.2%}")
    print(f"Max Drawdown: {best_params['max_drawdown']:.2%}")
    print(f"Score: {best_params['score']:.4f}") 