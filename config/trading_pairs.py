# Trading pair configurations
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

TRADING_PAIRS: Dict[str, Dict[str, Any]] = {
    'BTCUSDT': {
        'symbol': 'BTCUSDT',
        'base_asset': 'BTC',
        'quote_asset': 'USDT',
        'min_qty': 0.00001,
        'max_qty': 9000,
        'step_size': 0.00001,
        'tick_size': 0.01,
        'min_notional': 10,
        'price_precision': 2,
        'quantity_precision': 5,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.2,
        'stop_loss_percentage': 0.1,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'ETHUSDT': {
        'symbol': 'ETHUSDT',
        'base_asset': 'ETH',
        'quote_asset': 'USDT',
        'min_qty': 0.001,
        'max_qty': 5000,
        'step_size': 0.001,
        'tick_size': 0.01,
        'min_notional': 10,
        'price_precision': 2,
        'quantity_precision': 3,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.3,
        'stop_loss_percentage': 0.2,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'BNBUSDT': {
        'symbol': 'BNBUSDT',
        'base_asset': 'BNB',
        'quote_asset': 'USDT',
        'min_qty': 0.01,
        'max_qty': 10000,
        'step_size': 0.01,
        'tick_size': 0.01,
        'min_notional': 10,
        'price_precision': 2,
        'quantity_precision': 2,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.3,
        'stop_loss_percentage': 0.2,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'XRPUSDT': {
        'symbol': 'XRPUSDT',
        'base_asset': 'XRP',
        'quote_asset': 'USDT',
        'min_qty': 1,
        'max_qty': 1000000,
        'step_size': 1,
        'tick_size': 0.00001,
        'min_notional': 10,
        'price_precision': 5,
        'quantity_precision': 0,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.5,
        'stop_loss_percentage': 0.3,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'SOLUSDT': {
        'symbol': 'SOLUSDT',
        'base_asset': 'SOL',
        'quote_asset': 'USDT',
        'min_qty': 0.01,
        'max_qty': 10000,
        'step_size': 0.01,
        'tick_size': 0.01,
        'min_notional': 10,
        'price_precision': 2,
        'quantity_precision': 2,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.4,
        'stop_loss_percentage': 0.3,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'ADAUSDT': {
        'symbol': 'ADAUSDT',
        'base_asset': 'ADA',
        'quote_asset': 'USDT',
        'min_qty': 1,
        'max_qty': 1000000,
        'step_size': 1,
        'tick_size': 0.00001,
        'min_notional': 10,
        'price_precision': 5,
        'quantity_precision': 0,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.5,
        'stop_loss_percentage': 0.3,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'DOTUSDT': {
        'symbol': 'DOTUSDT',
        'base_asset': 'DOT',
        'quote_asset': 'USDT',
        'min_qty': 0.01,
        'max_qty': 10000,
        'step_size': 0.01,
        'tick_size': 0.0001,
        'min_notional': 10,
        'price_precision': 4,
        'quantity_precision': 2,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.4,
        'stop_loss_percentage': 0.3,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'LINKUSDT': {
        'symbol': 'LINKUSDT',
        'base_asset': 'LINK',
        'quote_asset': 'USDT',
        'min_qty': 0.1,
        'max_qty': 100000,
        'step_size': 0.1,
        'tick_size': 0.0001,
        'min_notional': 10,
        'price_precision': 4,
        'quantity_precision': 1,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.4,
        'stop_loss_percentage': 0.3,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
    'DOGEUSDT': {
        'symbol': 'DOGEUSDT',
        'base_asset': 'DOGE',
        'quote_asset': 'USDT',
        'min_qty': 1,
        'max_qty': 10000000,
        'step_size': 1,
        'tick_size': 0.000001,
        'min_notional': 10,
        'price_precision': 6,
        'quantity_precision': 0,
        'bb_period': 20,
        'bb_std_dev': 2,
        'profit_percentage': 0.6,
        'stop_loss_percentage': 0.4,
        'max_open_trades': 3,
        'position_size_percentage': 5,
    },
}

# Function to get trading pair configuration
def get_pair_config(symbol: str) -> Dict[str, Any]:
    return TRADING_PAIRS.get(symbol, None)

# Function to get all trading pair symbols
def get_all_symbols() -> List[str]:
    return list(TRADING_PAIRS.keys())

# Function to validate a trading pair
def is_valid_pair(symbol: str) -> bool:
    return symbol in TRADING_PAIRS

# Function to get specific parameter for a trading pair
def get_pair_parameter(symbol: str, parameter: str) -> Any:
    pair_config = get_pair_config(symbol)
    if pair_config:
        return pair_config.get(parameter, None)
    return None

def update_trading_pairs(new_pairs: List[str]) -> None:
    global TRADING_PAIRS
    updated_pairs = {}
    for symbol in new_pairs:
        if symbol in TRADING_PAIRS:
            updated_pairs[symbol] = TRADING_PAIRS[symbol]
        else:
            # Add a new pair with default configuration
            updated_pairs[symbol] = {
                'symbol': symbol,
                'base_asset': symbol[:-4],  # Assuming USDT pairs
                'quote_asset': 'USDT',
                'min_qty': 0.00001,
                'max_qty': 9000,
                'step_size': 0.00001,
                'tick_size': 0.01,
                'min_notional': 10,
                'price_precision': 2,
                'quantity_precision': 5,
                'bb_period': 20,
                'bb_std_dev': 2,
                'profit_percentage': 0.2,
                'stop_loss_percentage': 0.1,
                'max_open_trades': 3,
                'position_size_percentage': 5,
            }
    TRADING_PAIRS = updated_pairs
    logger.info(f"Updated TRADING_PAIRS with {len(TRADING_PAIRS)} pairs")
