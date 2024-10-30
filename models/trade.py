from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
import uuid
import time

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

@dataclass
class Trade:
    symbol: str
    entry_price: float
    amount: float
    side: str  # 'buy' or 'sell'
    entry_time: float = time.time()
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TradeStatus = TradeStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fees: float = 0.0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.set_stop_loss_take_profit()

    def set_stop_loss_take_profit(self):
        if self.side == 'buy':
            self.stop_loss = self.entry_price * (1 - settings.STOP_LOSS_PERCENTAGE)
            self.take_profit = self.entry_price * (1 + settings.TAKE_PROFIT_PERCENTAGE)
        elif self.side == 'sell':
            self.stop_loss = self.entry_price * (1 + settings.STOP_LOSS_PERCENTAGE)
            self.take_profit = self.entry_price * (1 - settings.TAKE_PROFIT_PERCENTAGE)

    def update_status(self, new_status: TradeStatus):
        self.status = new_status
        logger.info(f"Trade {self.trade_id} status updated to {new_status.value}")

    def close(self, exit_price: float, exit_time: Optional[datetime] = None):
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.utcnow()
        self.update_status(TradeStatus.CLOSED)
        self.calculate_pnl()

    def cancel(self):
        self.update_status(TradeStatus.CANCELLED)

    def calculate_pnl(self):
        if self.exit_price is None:
            return

        if self.side == 'buy':
            self.pnl = (self.exit_price - self.entry_price) * self.amount
            self.pnl_percentage = (self.exit_price - self.entry_price) / self.entry_price * 100
        elif self.side == 'sell':
            self.pnl = (self.entry_price - self.exit_price) * self.amount
            self.pnl_percentage = (self.entry_price - self.exit_price) / self.entry_price * 100

        # Subtract fees
        self.pnl -= self.fees
        logger.info(f"Trade {self.trade_id} closed with PNL: {self.pnl:.2f} ({self.pnl_percentage:.2f}%)")

    def update_fees(self, fees: float):
        self.fees += fees
        logger.info(f"Updated fees for trade {self.trade_id}: {self.fees}")

    def is_stop_loss_triggered(self, current_price: float) -> bool:
        if self.stop_loss is None:
            return False
        return (self.side == 'buy' and current_price <= self.stop_loss) or \
               (self.side == 'sell' and current_price >= self.stop_loss)

    def is_take_profit_triggered(self, current_price: float) -> bool:
        if self.take_profit is None:
            return False
        return (self.side == 'buy' and current_price >= self.take_profit) or \
               (self.side == 'sell' and current_price <= self.take_profit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status.value,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "fees": self.fees,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        trade = cls(
            symbol=data['symbol'],
            entry_price=data['entry_price'],
            amount=data['amount'],
            side=data['side'],
            entry_time=datetime.fromisoformat(data['entry_time']),
            trade_id=data['trade_id'],
            status=TradeStatus(data['status']),
            exit_price=data.get('exit_price'),
            exit_time=datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None,
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            fees=data.get('fees', 0.0),
            pnl=data.get('pnl', 0.0),
            pnl_percentage=data.get('pnl_percentage', 0.0),
            metadata=data.get('metadata', {})
        )
        return trade

    def __str__(self):
        return (f"Trade(id={self.trade_id}, symbol={self.symbol}, side={self.side}, "
                f"amount={self.amount}, entry_price={self.entry_price}, "
                f"status={self.status.value}, pnl={self.pnl:.2f})")

# Usage example
if __name__ == "__main__":
    # Create a new trade
    trade = Trade(symbol="BTCUSDT", entry_price=50000, amount=0.1, side="buy")
    print(f"New trade created: {trade}")

    # Update trade status
    trade.update_status(TradeStatus.CLOSED)

    # Close the trade
    trade.close(exit_price=51000)
    print(f"Trade closed: {trade}")

    # Convert trade to dictionary and back
    trade_dict = trade.to_dict()
    restored_trade = Trade.from_dict(trade_dict)
    print(f"Restored trade: {restored_trade}")

    # Check stop loss and take profit
    current_price = 49500
    print(f"Stop loss triggered: {trade.is_stop_loss_triggered(current_price)}")
    print(f"Take profit triggered: {trade.is_take_profit_triggered(current_price)}")
