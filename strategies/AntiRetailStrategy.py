from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy, plot_heatmaps, crossover
from backtesting.test import SMA
import pandas as pd

"""
Anti retail strategy.
This strategy will short when the retail trader is buying and long when the retail trader is selling.


It request the inputed dataframe has the following columns:
Change_in_NonRept_Positions_Long_All,Change_in_NonRept_Positions_Short_All,Change_in_NonRept_Positions_Spread_All

"""

class AntiRetailStrategy(SignalStrategy,
    TrailingStrategy):
    threshold = 0.05
    