import asyncio
from config import settings
from exchange.binance_client import BinanceClientWrapper
from data.market_data import MarketData
from strategies.scalping import ScalpingStrategy
from utils.risk_management import RiskManager
from utils.performance_tracker import PerformanceTracker
from utils.logger import get_logger

logger = get_logger(__name__)

class AlphaBot:
    def __init__(self):
        self.exchange_client = None
        self.market_data = None
        self.risk_manager = None
        self.strategy = None
        self.performance_tracker = None
        self.initialized = False

    async def initialize(self):
        logger.info("Initializing bot...")
        try:
            self.exchange_client = BinanceClientWrapper(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
            await self.exchange_client.initialize()
            self.market_data = MarketData(self.exchange_client)
            await self.market_data.update_all_market_data()
            self.risk_manager = RiskManager(
                position_size_percentage=0.02,
                market_data=self.market_data,
                exchange_client=self.exchange_client
            )
            self.strategy = ScalpingStrategy(self.market_data, self.exchange_client, self.risk_manager)
            self.performance_tracker = PerformanceTracker()
            
            try:
                account_balance = await self.exchange_client.get_total_balance_in_usdt()
                logger.info(f"Account balance: ${account_balance}")
            except Exception as e:
                logger.error(f"Error getting account balance: {e}")
            
            self.initialized = True
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            self.initialized = False

    async def start(self):
        logger.info("Starting Alpha Bot...")
        try:
            await self.initialize()
            if not self.initialized:
                logger.error("Bot initialization failed. Exiting.")
                return
            await self.exchange_client.convert_all_assets_to_usdt()
            await self.run()
        except Exception as e:
            logger.error(f"An error occurred while running the bot: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def run(self):
        if not self.initialized or not self.strategy:
            logger.error("Bot not properly initialized. Cannot run.")
            return

        logger.info("Bot is now running...")
        try:
            while True:
                await self.strategy.run_iteration()
                await asyncio.sleep(settings.BOT_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Terminated Bot Operation!")
        except Exception as e:
            logger.error(f"An error occurred while running the bot: {e}", exc_info=True)

    async def shutdown(self):
        logger.info("Shutting down Alpha Bot...")
        if self.strategy:
            try:
                await self.strategy.close_all_trades()
            except Exception as e:
                logger.error(f"Error closing trades during shutdown: {e}", exc_info=True)
        if self.exchange_client:
            try:
                await self.exchange_client.close_connection()
            except Exception as e:
                logger.error(f"Error closing exchange connection during shutdown: {e}", exc_info=True)
        if self.performance_tracker:
            report = self.performance_tracker.generate_report()
            logger.info(f"Final performance report:\n{report}")

async def main():    
    bot = AlphaBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
