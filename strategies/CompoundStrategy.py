from backtesting import Backtest, Strategy
from backtesting.lib import (
    crossover,
    SignalStrategy,
    TrailingStrategy,
    plot_heatmaps,
    crossover,
)
import pandas as pd
from strategies.FollowLargeTrader import FollowLargeTraderStrategy
from strategies.SmaCross import SmaCross

"""
This Strategy will take some strategies and combine them together
To initialize this strategy, you need to pass in a list of strategies, and a list of weights.
"""


class CompoundStrategy(SmaCross, FollowLargeTraderStrategy):
    exit_portion = 0.5
    entry_size_ratio = 0.8

    def init(self):
        super().init()

        self.signal = self.SMACrossSignal * self.FollowLargeTraderSignal

        self.entry_size = self.signal * self.entry_size_ratio

        self.set_signal(entry_size=self.entry_size)

        self.set_trailing_sl(8)
        self.set_atr_periods(50)

    def compose(self, strategies, weights):
        print(
            "Composing Strategies... your strategies: ",
            strategies,
            " your weights: ",
            weights,
        )

        self.strategies = strategies
        self.weights = weights

        assert len(strategies) == len(
            weights
        ), "The number of strategies and weights must be the same"
        assert sum(weights) == 1, "The sum of weights must be 1"

        return self
