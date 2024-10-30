import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config import settings
from utils.logger import get_logger
from exchange.binance_client import BinanceClientWrapper
logger = get_logger(__name__)

class MarketData:
    def __init__(self, exchange_client: BinanceClientWrapper):
        self.exchange_client = exchange_client
        self.cache: Dict[str, pd.DataFrame] = {}
        self.market_data = {}

    async def update_market_data(self, symbol: str) -> None:
        try:
            klines = await self.exchange_client.get_klines(symbol, settings.TIMEFRAME, limit=10000)
            if not klines:
                logger.warning(f"No kline data received for {symbol}")
                return

            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            self.cache[symbol] = df
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
            self.cache[symbol] = pd.DataFrame()

    async def update_all_market_data(self):
        exchange_info = await self.exchange_client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            if symbol['status'] == 'TRADING':
                ticker = await self.exchange_client.get_symbol_ticker(symbol['symbol'])
                self.market_data[symbol['symbol']] = ticker
        logger.info(f"Updated market data for {len(self.market_data)} symbols")

    def get_cached_data(self, symbol: str) -> pd.DataFrame:
        return self.cache.get(symbol, pd.DataFrame())

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = await self.exchange_client.get_ticker(symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def calculate_support_resistance(self, symbol: str) -> Tuple[float, float]:
        # Calculate and return support and resistance levels
        pass

    def detect_trend(self, symbol: str) -> str:
        # Detect and return the current trend (bullish, bearish, or sideways)
        pass

    async def get_order_book_imbalance(self, symbol: str) -> float:
        # Calculate and return the order book imbalance
        pass

    async def get_historical_data(self, symbol: str, interval: str, limit: int = 10000) -> pd.DataFrame:
        try:
            klines = await self.exchange_client.get_klines(symbol, interval, limit=limit)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_klines(self, symbol: str, interval: str, limit: int = 10000) -> List[List]:
        try:
            klines = await self.exchange_client.get_klines(symbol=symbol, interval=interval, limit=limit)
            return klines
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []