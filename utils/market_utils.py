import asyncio
from typing import List
from config import settings
from exchange.binance_client import BinanceClientWrapper
from utils.logger import get_logger

logger = get_logger(__name__)

async def update_trading_pairs(exchange_client: BinanceClientWrapper) -> None:
    try:
        top_pairs = await exchange_client.get_top_trading_pairs(base_asset='USDT', limit=20)
        
        if top_pairs:
            settings.TRADING_PAIRS = top_pairs
            logger.info(f"Updated TRADING_PAIRS with top 20 hot coins: {top_pairs}")
        else:
            logger.warning("Failed to fetch top trading pairs. TRADING_PAIRS remains unchanged.")
    except Exception as e:
        logger.error(f"Error updating trading pairs: {e}")

def run_update_trading_pairs(exchange_client: BinanceClientWrapper) -> None:
    asyncio.run(update_trading_pairs(exchange_client))

