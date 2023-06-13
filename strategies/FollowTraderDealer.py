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
Asset_Mgr_Positions_Long_All,Asset_Mgr_Positions_Short_All,Asset_Mgr_Positions_Spread_All
"""

class FollowTraderDealer(SignalStrategy,
                                TrailingStrategy):
    threshold = 0.05
    n1 = 5
    exit_portion = 0.5
    x = 0.6
    y = 0.3
    weights = [x, y, 1 - x - y]
    
    def init(self):
        
        super().init()
        
        
        Change_in_Dealer_Long_All_in_Pct = self.data.Change_in_Dealer_Long_All.astype(float) / (self.data.Asset_Mgr_Positions_Long_All.astype(float) + self.data.Change_in_Dealer_Long_All.astype(float))
        Change_in_Dealer_Short_All_in_Pct = self.data.Change_in_Dealer_Short_All.astype(float) / (self.data.Asset_Mgr_Positions_Short_All.astype(float) + self.data.Change_in_Dealer_Short_All.astype(float))
        Change_in_Dealer_Spread_All_in_Pct = self.data.Change_in_Dealer_Spread_All.astype(float) / (self.data.Asset_Mgr_Positions_Spread_All.astype(float) + self.data.Change_in_Dealer_Spread_All.astype(float))
        
        self.assert_Asset_Mgr_Positions_Long_All_increase = self.I(lambda x: x > self.threshold, Change_in_Dealer_Long_All_in_Pct)
        self.assert_Asset_Mgr_Positions_Short_All_increase = self.I(lambda x: x > self.threshold, Change_in_Dealer_Short_All_in_Pct)
        self.assert_Asset_Mgr_Positions_Spread_All_increase = self.I(lambda x: x > self.threshold, Change_in_Dealer_Spread_All_in_Pct)
        
        
        self.signal = self.assert_Asset_Mgr_Positions_Long_All_increase * self.weights[0] - self.assert_Asset_Mgr_Positions_Short_All_increase * self.weights[1] + self.assert_Asset_Mgr_Positions_Spread_All_increase * self.weights[2]
        
        entry_size = self.signal * .95
        
        
        self.set_signal(entry_size=entry_size, exit_portion=[self.exit_portion for _ in range(len(entry_size))])
        self.set_trailing_sl(8)
        self.set_atr_periods(50)