from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy
from backtesting.test import SMA
from strategies.utils import RSI
import pandas as pd
import pandas_ta as pta
import numpy as np

"""
When market width is low and the index begins to rise, this strategy suggests caution.
This may indicate that the rise is not broad-based, but rather driven by a smaller number of stocks. In this scenario, you might reduce your exposure, either by selling some holdings or implementing protective measures like stop orders or options.

When market width is high and the index rises, this strategy suggests greed, implying
a strong, broad-based rally. In this case, you might increase your long exposure,
buying more stocks or index funds.

If market width suddenly drops (exceeding a defined threshold) while the index is
rising, your strategy suggests that this is abnormal and might be a precursor to
a downturn. You would take a short position in anticipation of a potential decline.

To initialize the strategy, you need to pass in the following parameters:
    threshold: the threshold of market width change

It require the inputed dataframe has the following columns:
    market-sma50_larger_price, market-sma150_larger_price, market-sma200_larger_price
"""
class DivergeWithMarketBreadthStrategy(SignalStrategy, TrailingStrategy):
    
    low_threshold_in_value = 40
    high_threshold_in_value = 60
    threshold_in_change = 0.1
    period = 50
    exit_portion = 0.5
    
    def init(self):
        super().init()
        
        self.market_width_short = self.data.df['market-sma50_larger_price']
        
        self.market_width_change = self.market_width_short.diff()
        
        # print(self.market_width_change.describe())
        
        self.assert_low_market_width = self.I(lambda x: x < self.low_threshold_in_value, self.market_width_short)

        self.assert_high_market_width = self.I(lambda x: x > self.high_threshold_in_value, self.market_width_short)
        
        
        
        self.entry_size = self.threshold_in_change * self.market_width_short * 0.5
        
        self.set_signal(entry_size = self.entry_size)
        
        self.set_trailing_sl(8)
        self.set_atr_periods(50)
    def next(self):
        super().next()
        
        # If market width is low and the index begins to rise, this strategy suggests caution.
        
        if(self.assert_low_market_width):
            self.buy()
        elif(self.assert_high_market_width):
            self.sell()
    