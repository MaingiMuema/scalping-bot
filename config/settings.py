import os
from typing import List
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Binance API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading settings
TRADING_MODE = 'paper'  # 'paper' or 'live'
BASE_CURRENCY = 'USDT'

# Timeframe for the scalping strategy
TIMEFRAME = '1m'  # 1 minute candles

# Bollinger Bands settings
BB_PERIOD = 20
BB_STD_DEV = 2

# Scalping strategy parameters
PROFIT_PERCENTAGE = 0.2  # 0.2%
STOP_LOSS_PERCENTAGE = 0.4  # 0.4%
TAKE_PROFIT_PERCENTAGE = 0.6  # 0.6%
TRAILING_STOP_PERCENTAGE = 0.1  # 0.1%

# Risk management
MAX_OPEN_TRADES = 3
POSITION_SIZE_PERCENTAGE = 2  # 2% of account balance per trade
RISK_PERCENTAGE = 1  # 1% risk per trade

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'bot_activity.log'

# Database settings (if you decide to use a database later)
DB_NAME = 'scalping_bot.db'

# Backtesting settings
BACKTEST_START_DATE = '2023-01-01'
BACKTEST_END_DATE = '2023-12-31'

# Performance analysis
BENCHMARK_SYMBOL = 'BTCUSDT'  # for comparing bot performance
PERFORMANCE_REPORT_INTERVAL = 300  # Generate performance report every 5 minutes (in seconds)

# Telegram notifications (if you decide to add this feature)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Error handling
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Rate limiting
REQUESTS_PER_MINUTE = 1200  # Binance limit for API requests per minute

# Debugging
DEBUG_MODE = False

TRADING_PAIRS: List[str] = [
    'BTCUSDT',
    'ETHUSDT',
    'BNBUSDT',
    'XRPUSDT',
    'SOLUSDT',
    'ADAUSDT',
    'DOTUSDT',
    'LINKUSDT',
    'DOGEUSDT'
]

# Function to update the trading pairs
def update_trading_pairs(new_pairs: List[str]) -> None:
    global TRADING_PAIRS
    TRADING_PAIRS = new_pairs

# Feature flags
ENABLE_TELEGRAM_NOTIFICATIONS = False
ENABLE_EMAIL_NOTIFICATIONS = False
ENABLE_PERFORMANCE_ANALYTICS = True

# Async settings
ASYNC_CONCURRENCY_LIMIT = 10

# Backtesting specific settings
COMMISSION_RATE = 0.001  # 0.1% commission rate
SLIPPAGE = 0.0005  # 0.05% slippage

# Web interface settings (if you decide to add a web interface later)
WEB_SERVER_PORT = 5000
WEB_SERVER_HOST = '0.0.0.0'

# Bot loop interval
BOT_LOOP_INTERVAL = 60  # Run the bot loop every 60 seconds

# Error cooldown
ERROR_COOLDOWN = 300  # Wait for 5 minutes after an error before continuing

# Performance update interval
PERFORMANCE_UPDATE_INTERVAL = 60  # Update performance metrics every 60 seconds

# Minimum risk-reward ratio
MIN_RISK_REWARD_RATIO = 2  # Minimum risk-reward ratio for trades

# Testnet-specific settings
IS_TESTNET = True
TESTNET_BASE_URL = 'https://testnet.binance.vision'

# Adjust these values for testnet
POSITION_SIZE_PERCENTAGE = 1  # Use 1% of balance for each trade
MAX_OPEN_TRADES = 10
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = 0.03  # 3% take profit

BOT_INTERVAL = 60  # Run the bot every 60 seconds