import math
from typing import Dict, Optional, Any
from decimal import Decimal, ROUND_DOWN
from config import settings
from config.trading_pairs import get_pair_config
from models.trade import Trade
from data.market_data import MarketData
from exchange.binance_client import BinanceClientWrapper
from utils.logger import get_logger

logger = get_logger(__name__)

class RiskManager:
    def __init__(self, position_size_percentage: float = 0.02, market_data: MarketData = None, exchange_client: BinanceClientWrapper = None):
        self.position_size_percentage = position_size_percentage
        self.market_data = market_data
        self.exchange_client = exchange_client
        self.max_position_size = 0.02  # 2% of account balance
        self.max_risk_per_trade = 0.01  # 1% of account balance
        self.stop_loss_percentage = 0.015  # 1.5% stop loss
        self.take_profit_percentage = 0.03  # 3% take profit

    async def calculate_position_size(self, symbol: str, current_price: float) -> float:
        account_balance = await self.exchange_client.get_account_balance()
        max_position_size = account_balance * self.position_size_percentage
        
        # Get symbol-specific limits
        symbol_info = await self.exchange_client.get_symbol_info(symbol)
        min_qty = float(symbol_info['filters'][2]['minQty'])
        max_qty = float(symbol_info['filters'][2]['maxQty'])
        step_size = float(symbol_info['filters'][2]['stepSize'])

        # Calculate position size
        position_size = min(max_position_size / current_price, max_qty)
        position_size = max(position_size, min_qty)
        
        # Round to step size
        position_size = round(position_size / step_size) * step_size

        return position_size

    def round_step_size(self, quantity: float, step_size: float) -> float:
        return math.floor(quantity / step_size) * step_size

    async def get_volatility(self, symbol: str) -> float:
        # Implement your volatility calculation here
        # For now, we'll return a default value
        return 1.0  # You should replace this with actual volatility calculation

    async def get_available_balance(self, symbol: str) -> Optional[float]:
        try:
            account_info = await self.exchange_client.get_account()
            quote_asset = symbol.replace(settings.BASE_CURRENCY, '')
            for balance in account_info['balances']:
                if balance['asset'] == quote_asset:
                    return float(balance['free'])
            logger.warning(f"No balance found for {quote_asset}")
            return 0
        except Exception as e:
            logger.error(f"Error getting available balance for {symbol}: {e}")
            return 0

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Calculate the stop loss price based on the entry price and trade side.
        """
        if side == 'buy':
            return entry_price * (1 - self.stop_loss_percentage)
        elif side == 'sell':
            return entry_price * (1 + self.stop_loss_percentage)
        else:
            logger.error(f"Invalid trade side: {side}")
            return 0.0

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Calculate the take profit price based on the entry price and trade side.
        """
        if side == 'buy':
            return entry_price * (1 + self.take_profit_percentage)
        elif side == 'sell':
            return entry_price * (1 - self.take_profit_percentage)
        else:
            logger.error(f"Invalid trade side: {side}")
            return 0.0

    def validate_trade(self, trade: Trade, current_price: float, open_trades: Dict[str, Trade]) -> bool:
        """
        Validate if a trade should be executed based on risk management rules.
        """
        # Check if maximum number of open trades is reached
        if len(open_trades) >= self.max_open_trades:
            logger.warning("Maximum number of open trades reached. Cannot open new trade.")
            return False

        # Check if the trade is within the allowed risk limits
        if trade.amount * current_price > settings.MAX_TRADE_AMOUNT:
            logger.warning(f"Trade amount exceeds maximum allowed. Amount: {trade.amount}, Price: {current_price}")
            return False

        # Additional risk checks can be added here

        return True

    async def adjust_trade_for_risk(self, trade: Trade, account_balance: float) -> Optional[Trade]:
        """
        Adjust the trade parameters to comply with risk management rules.
        """
        max_position_size = await self.calculate_position_size(trade.symbol, account_balance)
        if trade.amount > max_position_size:
            logger.warning(f"Adjusting trade amount from {trade.amount} to {max_position_size} due to risk limits.")
            trade.amount = max_position_size

        trade.stop_loss = self.calculate_stop_loss(trade.entry_price, trade.side)
        trade.take_profit = self.calculate_take_profit(trade.entry_price, trade.side)

        return trade

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """
        Calculate the risk-reward ratio for a trade.
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk == 0:
            return 0
        return reward / risk

    def is_risk_acceptable(self, trade: Trade) -> bool:
        """
        Determine if the risk-reward ratio of a trade is acceptable.
        """
        risk_reward_ratio = self.calculate_risk_reward_ratio(trade.entry_price, trade.stop_loss, trade.take_profit)
        min_risk_reward_ratio = settings.MIN_RISK_REWARD_RATIO
        if risk_reward_ratio < min_risk_reward_ratio:
            logger.warning(f"Risk-reward ratio {risk_reward_ratio} is below minimum {min_risk_reward_ratio} for {trade.symbol}")
            return False
        return True

    def get_pair_config(self, symbol: str):
        return get_pair_config(symbol)

    def calculate_volatility(self, symbol: str) -> float:
        # Calculate and return the symbol's volatility
        # You can use methods like Average True Range (ATR) or standard deviation
        pass

    def calculate_risk_per_trade(self, account_balance: float) -> float:
        return account_balance * self.risk_percentage / 100

    def adjust_stop_loss(self, trade: Trade, current_price: float):
        if trade.side == "buy":
            new_stop_loss = max(trade.stop_loss, current_price * (1 - self.trailing_stop_percentage))
        else:
            new_stop_loss = min(trade.stop_loss, current_price * (1 + self.trailing_stop_percentage))
        trade.stop_loss = new_stop_loss

    def adjust_take_profit(self, trade: Trade, current_price: float):
        if trade.side == "buy":
            new_take_profit = max(trade.take_profit, current_price * (1 + self.take_profit_percentage))
        else:
            new_take_profit = min(trade.take_profit, current_price * (1 - self.take_profit_percentage))
        trade.take_profit = new_take_profit
