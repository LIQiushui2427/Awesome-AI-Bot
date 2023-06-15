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
    
    threshold_in_change = 0.02
    period = 50
    exit_portion = 0.5
    
    def init(self):
        super().init()
        
        self.market_width_short = self.data.df['market-sma50_larger_price']
        print(self.market_width_short.describe())
        self.market_width_change_in_pct = self.market_width_short.diff()/self.market_width_short
        
        # print(self.market_width_change.describe())
        
        self.assert_sma50_market_width_drop = self.I(lambda x: x < -self.threshold_in_change, self.market_width_change_in_pct)
        self.assert_sma50_market_width_rise = self.I(lambda x: x > self.threshold_in_change, self.market_width_change_in_pct)
        
        self.assert_index_rise = self.I(lambda x: x > 0, self.data.Close.df.diff()).reshape(-1)
        
        self.signal = -1 * self.assert_index_rise * self.assert_sma50_market_width_drop + self.assert_sma50_market_width_rise
        
        self.entry_size = self.signal * 0.95
        
        self.set_signal(entry_size = self.entry_size)
        
        # print("Shape of assert_low_market_width: ", self.assert_low_market_width.shape)
        # print("Shape of assert_high_market_width: ", self.assert_high_market_width.shape)
        # print("Shape of assert_index_rise: ", self.assert_index_rise.shape)
        # print("Shape of entry_size: ", self.entry_size.shape)
        
        self.set_trailing_sl(2)
        self.set_atr_periods(20)
    # def next(self):
    #     super().next()
        
    #     # If market width is low and the index begins to rise, this strategy suggests caution.
        
    #     if(self.assert_low_market_width):
    #         self.buy(size=self.entry_size)
    #     elif(self.assert_high_market_width):
    #         self.sell(size=-self.entry_size)
    