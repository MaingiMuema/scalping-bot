import asyncio
import time
from typing import Dict, Any, List, Optional
from functools import wraps
import math
import ssl
import aiohttp
import certifi
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance.helpers import date_to_milliseconds
from config import settings
from utils.logger import get_logger
from config.trading_pairs import TRADING_PAIRS
import os
import tempfile
from pathlib import Path
import sys

logger = get_logger(__name__)


def retry_on_exception(max_retries=3):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except BinanceAPIException as e:
                    if e.code == -1121:  # Invalid symbol
                        logger.warning(f"Invalid symbol encountered in {func.__name__}")
                        return None
                    elif attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
                    else:
                        logger.warning(f"Retrying {func.__name__} due to error: {e}")
                        await asyncio.sleep(1)  # Wait before retrying
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
        return wrapper
    return decorator

class BinanceClientWrapper:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self.request_weight = 0
        self.last_request_time = 0
        self.time_offset = 0

    async def initialize(self):
        try:
            # Patch the AsyncClient class to disable SSL verification
            from binance import client as binance_client
            import ssl
            
            # Create an unverified SSL context
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Patch the AsyncClient
            binance_client.AsyncClient.SESSION = None
            binance_client.AsyncClient.SSLCONTEXT = ssl_context
            
            # Create the client with patched settings
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=settings.IS_TESTNET,
                tld='com'
            )
            
            # Create our own session with SSL verification disabled
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                force_close=True,
                enable_cleanup_closed=True
            )
            
            session = aiohttp.ClientSession(
                connector=connector,
                trust_env=True
            )
            
            # Replace the client's session
            if hasattr(self.client, 'session') and self.client.session:
                await self.client.session.close()
            self.client.session = session
            self._session = session
            
            await self.update_time_offset()
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            if hasattr(self, '_session'):
                await self._session.close()
            if hasattr(self, 'client') and self.client:
                await self.client.close_connection()
            raise

    async def _init_time_offset(self):
        server_time = await self.client.get_server_time()
        self.time_offset = int(server_time['serverTime']) - int(time.time() * 1000)

    async def _wait_for_rate_limit(self):
        current_time = time.time()
        if current_time - self.last_request_time < 60:
            if self.request_weight >= settings.REQUESTS_PER_MINUTE:
                wait_time = 60 - (current_time - self.last_request_time)
                logger.warning(f"Rate limit reached. Waiting for {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)
                self.request_weight = 0
                self.last_request_time = time.time()
        else:
            self.request_weight = 0
            self.last_request_time = current_time

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            await self._wait_for_rate_limit()
            self.request_weight += 10
            exchange_info = await self.client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                # Add a default MIN_NOTIONAL filter if it's missing
                if not any(f['filterType'] == 'MIN_NOTIONAL' for f in symbol_info['filters']):
                    symbol_info['filters'].append({
                        'filterType': 'MIN_NOTIONAL',
                        'minNotional': '10.0'  # Default value
                    })
            return symbol_info
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    @retry_on_exception()
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        await self._wait_for_rate_limit()
        self.request_weight += 1
        try:
            klines = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            logger.debug(f"Successfully fetched {len(klines)} klines for {symbol}")
            return klines
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            raise

    @retry_on_exception()
    async def get_symbol_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        await self._wait_for_rate_limit()
        self.request_weight += 1
        return await self.client.get_symbol_ticker(symbol=symbol)

    @retry_on_exception()
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        await self._wait_for_rate_limit()
        self.request_weight += 1
        return await self.client.get_symbol_ticker(symbol=symbol)

    @retry_on_exception()
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        await self._wait_for_rate_limit()
        self.request_weight += 1
        return await self.client.get_order_book(symbol=symbol, limit=limit)

    @retry_on_exception()
    async def get_account(self) -> Dict[str, Any]:
        await self.update_time_offset()
        timestamp = int(time.time() * 1000) + self.time_offset
        return await self.client.get_account(timestamp=timestamp)

    @retry_on_exception()
    async def create_order(self, **params):
        await self._wait_for_rate_limit()
        self.request_weight += 1
        try:
            if 'quantity' in params:
                symbol = params['symbol']
                pair_config = self.get_pair_config(symbol)
                step_size = float(pair_config['step_size'])
                quantity_precision = pair_config['quantity_precision']
                quantity = float(params['quantity'])
                quantity = math.floor(quantity / step_size) * step_size
                params['quantity'] = f"{quantity:.{quantity_precision}f}"

            logger.info(f"Creating order with params: {params}")
            order = await self.client.create_order(**params)
            return order
        except BinanceAPIException as e:
            logger.error(f"Binance API error creating order: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating order: {e}")
            raise

    @retry_on_exception()
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        await self._wait_for_rate_limit()
        self.request_weight += 1
        return await self.client.cancel_order(symbol=symbol, orderId=order_id)

    @retry_on_exception()
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        await self._wait_for_rate_limit()
        self.request_weight += 1
        return await self.client.get_open_orders(symbol=symbol)

    @retry_on_exception()
    async def get_all_orders(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        await self._wait_for_rate_limit()
        self.request_weight += 5
        return await self.client.get_all_orders(symbol=symbol, limit=limit)

    @retry_on_exception()
    async def get_my_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        await self._wait_for_rate_limit()
        self.request_weight += 5
        return await self.client.get_my_trades(symbol=symbol, limit=limit)

    async def close_connection(self):
        if hasattr(self, '_session'):
            await self._session.close()
        if hasattr(self, 'client') and self.client:
            await self.client.close_connection()

    async def get_total_balance_in_usdt(self) -> float:
        try:
            account_info = await self.get_account()
            total_balance = 0.0

            for balance in account_info['balances']:
                asset = balance['asset']
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_asset_balance = free_balance + locked_balance

                if total_asset_balance > 0:
                    if asset == 'USDT':
                        total_balance += total_asset_balance
                    else:
                        symbol = f"{asset}USDT"
                        ticker = await self.get_symbol_ticker(symbol)
                        if ticker:
                            price = float(ticker['price'])
                            asset_value_in_usdt = total_asset_balance * price
                            total_balance += asset_value_in_usdt
                        else:
                            logger.warning(f"Skipping {asset} due to invalid symbol or unavailable price")

            return total_balance

        except Exception as e:
            logger.error(f"Error getting total balance in USDT: {e}")
            return 0.0

    def get_pair_config(self, symbol: str) -> Dict[str, Any]:
        return TRADING_PAIRS.get(symbol, {})

    async def get_account_balance(self):
        await self._init_time_offset()
        try:
            account_info = await self.client.get_account()
            usdt_balance = next((float(balance['free']) for balance in account_info['balances'] if balance['asset'] == 'USDT'), 0)
            return usdt_balance
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            return 0

    @retry_on_exception()
    async def create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict[str, Any]]:
        await self._wait_for_rate_limit()
        self.request_weight += 1
        try:
            # Get symbol info to determine the quantity precision
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Symbol info not found for {symbol}")
                return None

            # Find the LOT_SIZE filter to get the step size
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if not lot_size_filter:
                logger.error(f"LOT_SIZE filter not found for {symbol}")
                return None

            step_size = float(lot_size_filter['stepSize'])
            precision = int(round(-math.log(step_size, 10), 0))  # Calculate the precision

            # Round the quantity to the correct precision
            rounded_quantity = round(quantity, precision)

            # Create the order
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=rounded_quantity
            )
            logger.info(f"Market order created: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Binance API error creating market order: {e}")
            if e.code == -2010:  # Insufficient balance
                logger.error(f"Insufficient balance for {symbol} {side} order of quantity {quantity}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating market order: {e}")
            return None

    async def update_time_offset(self):
        server_time = await self.get_server_time()
        local_time = int(time.time() * 1000)
        self.time_offset = server_time - local_time

    async def get_server_time(self):
        server_time = await self.client.get_server_time()
        return server_time['serverTime']
    
    async def get_exchange_info(self):
        if not self.client:
            raise Exception("BinanceClientWrapper not initialized. Call initialize() first.")
        return await self.client.get_exchange_info()

    async def get_total_balance_in_usdt(self) -> float:
        try:
            account_info = await self.get_account()
            total_balance = 0.0

            for balance in account_info['balances']:
                asset = balance['asset']
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_asset_balance = free_balance + locked_balance

                if total_asset_balance > 0:
                    if asset == 'USDT':
                        total_balance += total_asset_balance
                    else:
                        symbol = f"{asset}USDT"
                        ticker = await self.get_symbol_ticker(symbol)
                        if ticker:
                            price = float(ticker['price'])
                            asset_value_in_usdt = total_asset_balance * price
                            total_balance += asset_value_in_usdt
                        else:
                            logger.warning(f"Skipping {asset} due to invalid symbol or unavailable price")

            return total_balance

        except Exception as e:
            logger.error(f"Error getting total balance in USDT: {e}")
            return 0.0

    async def convert_all_assets_to_usdt(self):
        try:
            account_info = await self.get_account()
            total_converted = 0.0

            for balance in account_info['balances']:
                asset = balance['asset']
                free_balance = float(balance['free'])

                if free_balance > 0 and asset != 'USDT':
                    symbol = f"{asset}USDT"
                    try:
                        # Get the current market price
                        ticker = await self.get_symbol_ticker(symbol)
                        if ticker:
                            price = float(ticker['price'])
                            
                            # Get symbol info
                            symbol_info = await self.get_symbol_info(symbol)
                            if not symbol_info:
                                logger.warning(f"No symbol info found for {symbol}, skipping conversion")
                                continue

                            # Get lot size filter
                            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                            if not lot_size_filter:
                                logger.warning(f"No LOT_SIZE filter found for {symbol}, skipping conversion")
                                continue

                            step_size = float(lot_size_filter['stepSize'])
                            precision = int(round(-math.log(step_size, 10), 0))

                            # Calculate the quantity to sell (considering minimum notional value)
                            min_notional = await self.get_min_notional(symbol)
                            quantity = max(free_balance, min_notional / price)
                            
                            # Round the quantity to the correct precision
                            quantity = round(quantity, precision)

                            if quantity * price < min_notional:
                                logger.warning(f"Balance for {asset} is too small to convert (value: {quantity * price} USDT, min notional: {min_notional} USDT)")
                                continue

                            # Create a market sell order
                            order = await self.create_market_order(symbol, 'SELL', quantity)
                            
                            if order and order['status'] == 'FILLED':
                                converted_amount = float(order['cummulativeQuoteQty'])
                                total_converted += converted_amount
                                logger.info(f"Converted {quantity} {asset} to {converted_amount} USDT")
                            else:
                                logger.warning(f"Failed to convert {asset} to USDT. Order status: {order['status'] if order else 'Unknown'}")
                        else:
                            logger.warning(f"No ticker found for {symbol}, skipping conversion")
                    except BinanceAPIException as e:
                        logger.error(f"BinanceAPIException converting {asset} to USDT: {e}")
                    except BinanceRequestException as e:
                        logger.error(f"BinanceRequestException converting {asset} to USDT: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error converting {asset} to USDT: {e}", exc_info=True)

            logger.info(f"Total converted to USDT: {total_converted}")
            return total_converted

        except Exception as e:
            logger.error(f"Error in convert_all_assets_to_usdt: {e}", exc_info=True)
            return 0.0

    async def get_min_notional(self, symbol: str) -> float:
        try:
            symbol_info = await self.get_symbol_info(symbol)
            if symbol_info:
                min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
                if min_notional_filter:
                    return float(min_notional_filter['minNotional'])
            return 10.0  # Default value if not found
        except Exception as e:
            logger.error(f"Error getting min notional for {symbol}: {e}")
            return 10.0  # Default value in case of error

    async def get_top_trading_pairs(self, base_asset: str = 'USDT', limit: int = 20) -> List[str]:
        try:
            ticker_24h = await self.client.get_ticker_24h()
            usdt_pairs = [t for t in ticker_24h if t['symbol'].endswith(base_asset)]
            sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['volume']), reverse=True)
            top_pairs = [pair['symbol'] for pair in sorted_pairs[:limit]]
            return top_pairs
        except Exception as e:
            logger.error(f"Error fetching top trading pairs: {e}")
            return []

# Usage example:
"""
async def main():
    client = BinanceClientWrapper(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
    
    # Get exchange info
    exchange_info = await client.get_exchange_info()
    print(f"Exchange info: {exchange_info}")
    
    # Get BTCUSDT ticker
    btc_ticker = await client.get_ticker('BTCUSDT')
    print(f"BTCUSDT ticker: {btc_ticker}")
    
    # Get account info
    account_info = await client.get_account()
    print(f"Account info: {account_info}")

if __name__ == "__main__":
    asyncio.run(main())

"""

