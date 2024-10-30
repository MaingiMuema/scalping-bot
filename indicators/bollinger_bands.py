import numpy as np
import pandas as pd
from typing import Tuple, Union
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class BollingerBands:
    def __init__(self, window=20, num_std=2):
        self.window = window  # Change 'period' to 'window'
        self.num_std = num_std

    def calculate(self, close_prices):
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.values.astype(float)

        sma = np.convolve(close_prices, np.ones(self.window), 'valid') / self.window
        std = np.std(close_prices[-self.window:])
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)

        return {
            'middle': sma[-1],
            'upper': upper_band[-1],
            'lower': lower_band[-1]
        }

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        return np.convolve(data, np.ones(period), 'valid') / period

    @staticmethod
    def _rolling_std(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        return np.array([np.std(data[i:i+period]) for i in range(len(data)-period+1)])

    def get_bb_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Bollinger Bands.

        Args:
            data (pd.DataFrame): DataFrame with 'close' column.

        Returns:
            pd.DataFrame: DataFrame with added Bollinger Bands and signals columns.
        """
        if len(data) < self.window: 
            logger.warning(f"Not enough data to calculate Bollinger Bands. Required: {self.window}, Got: {len(data)}")
            return data

        # Ensure 'close' column is numeric
        data['close'] = pd.to_numeric(data['close'], errors='coerce')

        # Calculate Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=self.window).mean()
        data['bb_std'] = data['close'].rolling(window=self.window).std()
        data['bb_upper'] = data['bb_middle'] + (self.num_std * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (self.num_std * data['bb_std'])

        # Generate signals
        data['bb_signal'] = np.where(data['close'] <= data['bb_lower'], 1,  # Buy signal
                            np.where(data['close'] >= data['bb_upper'], -1, 0))  # Sell signal

        return data

    @staticmethod
    def bb_squeeze(data: pd.DataFrame, squeeze_threshold: float = 0.1) -> pd.Series:
        """
        Detect Bollinger Bands squeeze.

        Args:
            data (pd.DataFrame): DataFrame with 'bb_upper' and 'bb_lower' columns.
            squeeze_threshold (float): Threshold for detecting squeeze.

        Returns:
            pd.Series: Boolean series indicating squeeze conditions.
        """
        band_width = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        return band_width < squeeze_threshold

    @staticmethod
    def bb_trend(data: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """
        Determine the trend based on Bollinger Bands.

        Args:
            data (pd.DataFrame): DataFrame with 'close' and 'bb_middle' columns.
            lookback (int): Number of periods to look back for trend determination.

        Returns:
            pd.Series: Series indicating trend (1 for uptrend, -1 for downtrend, 0 for no trend).
        """
        close_above_middle = data['close'] > data['bb_middle']
        trend = close_above_middle.rolling(window=lookback).sum()
        return np.where(trend == lookback, 1, np.where(trend == 0, -1, 0))

    def plot_bollinger_bands(self, data: pd.DataFrame, title: str = "Bollinger Bands"):
        """
        Plot Bollinger Bands with price data.

        Args:
            data (pd.DataFrame): DataFrame with 'close', 'bb_upper', 'bb_middle', and 'bb_lower' columns.
            title (str): Title for the plot.
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['close'], label='Close Price', alpha=0.5)
            plt.plot(data.index, data['bb_upper'], label='Upper BB', color='g', alpha=0.5)
            plt.plot(data.index, data['bb_middle'], label='Middle BB', color='r', alpha=0.5)
            plt.plot(data.index, data['bb_lower'], label='Lower BB', color='g', alpha=0.5)
            
            plt.fill_between(data.index, data['bb_upper'], data['bb_lower'], alpha=0.1)
            
            plt.title(title)
            plt.legend(loc='upper left')
            plt.show()
        except ImportError:
            logger.warning("Matplotlib is not installed. Unable to plot Bollinger Bands.")

# Usage example
"""
if __name__ == "__main__":
    # Sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100
    })

    bb = BollingerBands()
    result = bb.get_bb_signals(sample_data)

    print(result.tail())

    # Detect squeeze
    squeeze = bb.bb_squeeze(result)
    print("Squeeze detected:", squeeze.sum())

    # Determine trend
    trend = bb.bb_trend(result)
    print("Trend:", trend.value_counts())

    # Plot Bollinger Bands
    bb.plot_bollinger_bands(result)

"""