import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import (
    INITIAL_BALANCE,
    POSITION_SIZE_PERCENT,
    ATR_MULTIPLIER_SL,
    ATR_MULTIPLIER_TP
)

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies
    """
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = INITIAL_BALANCE,
        position_size_percent: float = POSITION_SIZE_PERCENT,
        sl_multiplier: float = ATR_MULTIPLIER_SL,
        tp_multiplier: float = ATR_MULTIPLIER_TP
    ):
        """
        Initialize backtesting engine
        
        Args:
            data: DataFrame with OHLCV data and indicators
            initial_balance: Initial account balance
            position_size_percent: Position size as percentage of balance
            sl_multiplier: Stop loss multiplier for ATR
            tp_multiplier: Take profit multiplier for ATR
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.position_size_percent = position_size_percent
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        
        # Initialize results
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]
        self.current_balance = initial_balance
        self.current_position = None
        
    def run(self) -> Dict:
        """
        Run backtest
        
        Returns:
            Dictionary with backtest results
        """
        for i in range(1, len(self.data)):
            current_bar = self.data.iloc[i]
            prev_bar = self.data.iloc[i-1]
            
            # Check for entry signals
            if self.current_position is None:
                self._check_entry(current_bar, prev_bar)
            
            # Check for exit signals
            else:
                self._check_exit(current_bar)
            
            # Update equity curve
            self.equity_curve.append(self.current_balance)
        
        return self._calculate_performance_metrics()
    
    def _check_entry(self, current_bar: pd.Series, prev_bar: pd.Series) -> None:
        """
        Check for entry signals
        """
        # Check for Donchian breakout
        if current_bar['Close'] > current_bar['DonchianHigh'] and prev_bar['Close'] <= prev_bar['DonchianHigh']:
            if current_bar['ADX'] >= 25:  # Strong trend filter
                self._enter_position(current_bar, 'BUY')
        elif current_bar['Close'] < current_bar['DonchianLow'] and prev_bar['Close'] >= prev_bar['DonchianLow']:
            if current_bar['ADX'] >= 25:  # Strong trend filter
                self._enter_position(current_bar, 'SELL')
    
    def _enter_position(self, bar: pd.Series, direction: str) -> None:
        """
        Enter a new position
        """
        entry_price = bar['Close']
        atr = bar['ATR']
        
        # Calculate stop loss and take profit
        if direction == 'BUY':
            stop_loss = entry_price - (self.sl_multiplier * atr)
            take_profit = entry_price + (self.tp_multiplier * atr)
        else:  # SELL
            stop_loss = entry_price + (self.sl_multiplier * atr)
            take_profit = entry_price - (self.tp_multiplier * atr)
        
        # Calculate position size
        risk_amount = self.current_balance * self.position_size_percent
        position_size = risk_amount / abs(entry_price - stop_loss)
        
        self.current_position = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': bar['OpenTime']
        }
    
    def _check_exit(self, current_bar: pd.Series) -> None:
        """
        Check for exit signals
        """
        if self.current_position is None:
            return
        
        current_price = current_bar['Close']
        position = self.current_position
        
        # Check stop loss
        if position['direction'] == 'BUY':
            if current_price <= position['stop_loss']:
                self._exit_position(current_bar, 'stop_loss')
            elif current_price >= position['take_profit']:
                self._exit_position(current_bar, 'take_profit')
        else:  # SELL
            if current_price >= position['stop_loss']:
                self._exit_position(current_bar, 'stop_loss')
            elif current_price <= position['take_profit']:
                self._exit_position(current_bar, 'take_profit')
    
    def _exit_position(self, bar: pd.Series, exit_type: str) -> None:
        """
        Exit current position
        """
        if self.current_position is None:
            return
        
        position = self.current_position
        exit_price = bar['Close']
        
        # Calculate profit/loss
        if position['direction'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:  # SELL
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Update balance
        self.current_balance += pnl
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': bar['OpenTime'],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'pnl': pnl,
            'exit_type': exit_type
        }
        self.trades.append(trade)
        
        # Reset position
        self.current_position = None
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Calculate drawdown
        equity_curve = pd.Series(self.equity_curve)
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Calculate Sharpe ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results
        
        Args:
            save_path: Path to save the plot (if None, will show instead)
        """
        plt.figure(figsize=(15, 10))
        
        # Create subplot grid
        gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        # Equity curve
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.data['OpenTime'], self.equity_curve, label='Equity Curve')
        ax1.set_title('Backtest Results', fontsize=15)
        ax1.set_ylabel('Account Balance', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Drawdown
        ax2 = plt.subplot(gs[1], sharex=ax1)
        equity_curve = pd.Series(self.equity_curve)
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        ax2.fill_between(self.data['OpenTime'], drawdowns, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Trade distribution
        ax3 = plt.subplot(gs[2], sharex=ax1)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            sns.histplot(data=trades_df, x='pnl', bins=50, ax=ax3)
            ax3.axvline(x=0, color='red', linestyle='--')
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_xlabel('Profit/Loss', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 