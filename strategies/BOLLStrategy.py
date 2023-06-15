from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy
from backtesting.test import SMA
from strategies.utils import RSI
import pandas as pd
import pandas_ta as pta
import numpy as np

"""
Bollinger Bands Mean Reversion Strategy
This strategy will buy when the price is below the lower band and sell when the price is above the upper band.

To initialize the strategy, you need to pass in the following parameters:
    bollinger_period: the period of Bollinger Bands
    bollinger_std: the standard deviation of Bollinger Bands
It request the inputed dataframe has the following columns:
    Close
"""

class BOLLStrategy(SignalStrategy):
    
    bollinger_period = 8
    bollinger_std = 2
    exit_portion = 0.5
    n1 = 5
    def init(self):
        
        super().init()
        
        self.bollinger = pta.bbands(self.data.df['Close'], length = self.bollinger_period, std = self.bollinger_std)
        
        # print(self.bollinger.columns)
        
        self.bollinger = self.bollinger.fillna(self.bollinger.mean())
        
        # print("shape of bollinger: ", self.bollinger[['BBL_20_2.0']].shape)
        
        self.Bollinger_Lower = pd.Series(self.bollinger.iloc[:,0])
        self.Bollinger_Middle = pd.Series(self.bollinger.iloc[:,1])
        self.Bollinger_Upper = pd.Series(self.bollinger.iloc[:,2])
        
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        
        
        # Buy when price is below the lower band
        
        # print("shape of Bollinger_Series: ", self.Bollinger_Series.shape)
        # print("shape of Close_Series: ", self.Close_Series.shape)
        
        
        self.buy_sig = self.I(lambda x , y: x < y, self.data.Close, self.Bollinger_Upper)
        self.sell_sig = self.I(lambda x , y: x > y, self.data.Close, self.Bollinger_Lower)
        
        # print("shape of buy_sig: ", self.buy_sig.shape)
        
        self.signal = self.buy_sig * 1 - self.sell_sig * 0.5
        
        self.entry_size = self.signal * 0.95
        
        # print("Shape of signal: ", self.signal.shape)
        
        self.set_signal(entry_size = self.entry_size)
        