import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from config import settings
from models.trade import Trade, TradeStatus
from utils.logger import get_logger

logger = get_logger(__name__)

class PerformanceTracker:
    def __init__(self):
        self.trades: List[Union[Dict, Trade]] = []
        self.daily_returns: List[float] = []
        self.initial_balance: Optional[float] = None
        self.current_balance: float = 0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_profit: float = 0
        self.total_loss: float = 0
        self.peak_balance: float = 0
        self.drawdown: float = 0
        self.last_update: datetime = datetime.now(timezone.utc)
        self.last_report_time: datetime = datetime.now(timezone.utc)
        self.risk_free_rate: float = 0.02  # Assuming 2% annual risk-free rate

    async def update(self, new_trades: List[Union[Dict, Trade]], current_balance: float):
        for trade in new_trades:
            self.add_trade(trade)
        
        self.update_balance(current_balance)
        self.last_update = datetime.now(timezone.utc)

    def add_trade(self, trade: Union[Dict, Trade]):
        self.trades.append(trade)
        self.total_trades += 1
        
        pnl = self.get_pnl(trade)
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
        
        self.update_daily_returns(trade)

    def update_balance(self, balance: float):
        if self.initial_balance is None:
            self.initial_balance = balance
        self.current_balance = balance
        self.peak_balance = max(self.peak_balance, balance)
        self.drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0

    def update_daily_returns(self, trade: Union[Dict, Trade]):
        if isinstance(trade, dict):
            entry_time = datetime.fromisoformat(trade['entry_time'])
            exit_time = datetime.fromisoformat(trade['exit_time'])
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
        else:
            entry_time = trade.entry_time
            exit_time = trade.exit_time
            entry_price = trade.entry_price
            exit_price = trade.exit_price

        if entry_time.date() == exit_time.date():
            daily_return = (exit_price - entry_price) / entry_price
            self.daily_returns.append(daily_return)

    def should_report(self) -> bool:
        return (datetime.now(timezone.utc) - self.last_report_time).total_seconds() >= settings.PERFORMANCE_REPORT_INTERVAL

    def generate_report(self) -> str:
        total_trades = self.total_trades
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = self.total_profit - self.total_loss
        avg_profit = total_pnl / total_trades if total_trades > 0 else 0
        roi = ((self.current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance else 0

        report = f"""
Performance Report ({self.last_update})
Total Trades: {total_trades}
Win Rate: {win_rate:.2f}%
Total PNL: ${total_pnl:.2f}
Average Profit: ${avg_profit:.2f}
Current Balance: ${self.current_balance:.2f}
Peak Balance: ${self.peak_balance:.2f}
Drawdown: {self.drawdown:.2f}%
ROI: {roi:.2f}%
        """
        
        sharpe_ratio = self.calculate_sharpe_ratio()
        if sharpe_ratio is not None:
            report += f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        else:
            report += "Sharpe Ratio: Not available\n"
        
        max_drawdown = self.calculate_max_drawdown()
        report += f"Max Drawdown: {max_drawdown:.2f}%\n"
        
        self.last_report_time = datetime.now(timezone.utc)
        return report.strip()

    def calculate_sharpe_ratio(self) -> Optional[float]:
        if not self.daily_returns:
            logger.warning("No daily returns available for Sharpe ratio calculation.")
            return None

        returns = pd.Series(self.daily_returns)
        if returns.empty or returns.std() == 0:
            logger.warning("Insufficient or constant returns for Sharpe ratio calculation.")
            return None

        sharpe_ratio = (returns.mean() - (self.risk_free_rate / 252)) / returns.std() * np.sqrt(252)
        return sharpe_ratio

    def calculate_max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        
        cumulative_returns = [1]
        for trade in self.trades:
            pnl = self.get_pnl(trade)
            cumulative_returns.append(cumulative_returns[-1] * (1 + pnl / self.initial_balance))
        
        cumulative_returns = pd.Series(cumulative_returns)
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
        return max_drawdown * 100

    @staticmethod
    def get_pnl(trade: Union[Dict, Trade]) -> float:
        if isinstance(trade, dict):
            return trade.get('pnl', 0)
        elif isinstance(trade, Trade):
            return trade.pnl
        else:
            logger.warning(f"Unexpected trade data type: {type(trade)}")
            return 0

    async def run_periodic_update(self, get_trades_func, get_balance_func):
        while True:
            try:
                new_trades = await get_trades_func()
                current_balance = await get_balance_func()
                await self.update(new_trades, current_balance)
                
                if self.should_report():
                    report = self.generate_report()
                    logger.info(f"Performance Report:\n{report}")
                
                await asyncio.sleep(settings.PERFORMANCE_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in performance tracker update: {e}")
                await asyncio.sleep(settings.ERROR_COOLDOWN)

# Usage example
"""
if __name__ == "__main__":
    async def sample_get_trades():
        return [
            Trade(symbol="BTCUSDT", entry_price=50000, exit_price=51000, amount=0.1, side="buy", pnl=100, entry_time=datetime.now(timezone.utc), exit_time=datetime.now(timezone.utc)),
            Trade(symbol="ETHUSDT", entry_price=3000, exit_price=2900, amount=1, side="buy", pnl=-100, entry_time=datetime.now(timezone.utc), exit_time=datetime.now(timezone.utc)),
        ]

    async def sample_get_balance():
        return 10000

    async def main():
        tracker = PerformanceTracker()
        await tracker.update(await sample_get_trades(), await sample_get_balance())
        print(tracker.generate_report())

    asyncio.run(main())

"""
