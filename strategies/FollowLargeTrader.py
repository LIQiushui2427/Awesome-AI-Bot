from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy, plot_heatmaps, crossover
from backtesting.test import SMA
import pandas as pd

"""
Follow Large Trader Strategy
This strategy will follow the large trader by buying when the large trader is buying and selling when the large trader is selling.
If the large trader is buying, the strategy will buy and hold until the large trader sells.

It request the inputed dataframe has the following columns:
Change_in_M_Money_Long_All,Change_in_M_Money_Long_All,Change_in_M_Money_Spread_All
M_Money_Positions_Long_All,M_Money_Positions_Short_All,M_Money_Positions_Spread_All
"""
def FollowLargeTrader(df , threshold = 0.05, n1 = 5, exit_portion = 0.5, weights = [0.6, 0.3, 0.1]):
    '''
    input data, and return several indecators
    it requires the inputed dataframe has the following columns:
    Change_in_M_Money_Long_All, Change_in_M_Money_Short_All, Change_in_M_Money_Spread_All
    M_Money_Positions_Long_All, M_Money_Positions_Short_All, M_Money_Positions_Spread_All
    '''
    # assert 'Change_in_M_Money_Long_All' in df.columns and 'Change_in_M_Money_Short_All' in df.columns and 'Change_in_M_Money_Spread_All' in df.columns
    # assert 'M_Money_Positions_Long_All' in df.columns and 'M_Money_Positions_Short_All' in df.columns and 'M_Money_Positions_Spread_All' in df.columns
    
    # print("FollowLargeTrader: your dataframe has the following columns: ", df.columns)
    # Change_in_M_Money_Long_All_in_Pct = df['Change_in_M_Money_Long_All'].astype(float) / (df['M_Money_Positions_Long_All'].astype(float) + df['Change_in_M_Money_Long_All'].astype(float))
    Change_in_M_Money_Long_All_in_Pct = df.Change_in_M_Money_Long_All.astype(float) / (df.M_Money_Positions_Long_All.astype(float) + df.Change_in_M_Money_Long_All.astype(float))
    Change_in_M_Money_Short_All_in_Pct = df.Change_in_M_Money_Short_All.astype(float) / (df.M_Money_Positions_Short_All.astype(float) + df.Change_in_M_Money_Short_All.astype(float))
    Change_in_M_Money_Spread_All_in_Pct = df.Change_in_M_Money_Spread_All.astype(float) / (df.M_Money_Positions_Spread_All.astype(float) + df.Change_in_M_Money_Spread_All.astype(float))
    print("Change_in_M_Money_Long_All_in_Pct is: ", Change_in_M_Money_Long_All_in_Pct)
    assert_M_Money_long_increase = Strategy.I(lambda x: x > threshold, Change_in_M_Money_Long_All_in_Pct)
    assert_M_Money_short_increase = Strategy.I(lambda x: x > threshold, Change_in_M_Money_Short_All_in_Pct)
    assert_M_Money_spread_increase = Strategy.I(lambda x: x > threshold, Change_in_M_Money_Spread_All_in_Pct)
    
    signal = pd.Series(assert_M_Money_long_increase * weights[0] - assert_M_Money_short_increase + weights[1] + assert_M_Money_spread_increase * weights[2])
    
    return assert_M_Money_long_increase, assert_M_Money_short_increase, assert_M_Money_spread_increase, signal


class FollowLargeTraderStrategy(SignalStrategy,
                                TrailingStrategy):
    threshold = 0
    n1 = 5
    exit_portion = 0.5
    x = 0.45
    y = 0.45
    weights = [x, y, 1 - x - y]
    
    def init(self):
        super().init()
        
        
        Change_in_M_Money_Long_All_in_Pct = self.data.Change_in_M_Money_Long_All.astype(float) / (self.data.M_Money_Positions_Long_All.astype(float) + self.data.Change_in_M_Money_Long_All.astype(float))
        Change_in_M_Money_Short_All_in_Pct = self.data.Change_in_M_Money_Short_All.astype(float) / (self.data.M_Money_Positions_Short_All.astype(float) + self.data.Change_in_M_Money_Short_All.astype(float))
        Change_in_M_Money_Spread_All_in_Pct = self.data.Change_in_M_Money_Spread_All.astype(float) / (self.data.M_Money_Positions_Spread_All.astype(float) + self.data.Change_in_M_Money_Spread_All.astype(float))
        
        self.assert_M_Money_long_decrease = self.I(lambda x: x > self.threshold, Change_in_M_Money_Long_All_in_Pct)
        self.assert_M_Money_short_decrease = self.I(lambda x: x > self.threshold, Change_in_M_Money_Short_All_in_Pct)
        self.assert_M_Money_spread_decrease = self.I(lambda x: x > self.threshold, Change_in_M_Money_Spread_All_in_Pct)
        
        
        
        self.FollowLargeTraderSignal = pd.Series(self.assert_M_Money_long_decrease * self.weights[0] - self.assert_M_Money_short_decrease + self.weights[1] + self.assert_M_Money_spread_decrease * self.weights[2]) \
        
        self.entry_size = self.FollowLargeTraderSignal
                
 
        
        
        # self.set_signal(entry_size=entry_size, exit_portion=[self.exit_portion for _ in range(len(entry_size))])
        # self.set_trailing_sl(8)
        # self.set_atr_periods(50)
        
        self.set_signal(entry_size=self.entry_size, exit_portion=[self.exit_portion for _ in range(len(entry_size))])
        self.set_trailing_sl(8)
        self.set_atr_periods(50)