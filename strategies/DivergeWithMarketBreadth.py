from backtesting.lib import SignalStrategy, TrailingStrategy
from backtesting.test import SMA
from strategies.SmaCross import SmaCross
import pandas as pd
import pandas_ta as pta


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


class LookMarketBreadthStrategy(SmaCross):
    period = 20
    exit_portion = 0.5
    sma_n1 = 10
    sma_n2 = 25
    cooldown = 5  # number of bars to wait after a trade

    def init(self):
        super().init()

        self.market_breadth_larger_sma50 = self.data.df[
            "constituent-sma50_larger_price"
        ]
        self.market_breadth_smaller_sma50 = self.data.df[
            "constituent-sma50_smaller_price"
        ]
        self.market_sma_50 = self.data.df["market-sma50_smaller_price"] / (
            self.data.df["market-sma50_larger_price"]
            + self.data.df["market-sma50_smaller_price"]
        )

        self.market_breadth_flag = pd.Series(0, index=self.market_sma_50.index)
        self.market_breadth_flag[self.market_sma_50 > 0.6] = 1
        self.market_breadth_flag[self.market_sma_50 < 0.4] = -1

        self.market_breadth_indicator = self.I(lambda x: x, self.market_breadth_flag)
        # print("self.market_breadth_flag.describe()", self.market_breadth_flag.describe())
        # print("sma1", sma1)

        # print("SMACrossSignal", self.SMACrossSignal.describe())

        # print("shape of buy_sig", self.buy_sig.shape)
        # print("shape of sell_sig", self.sell_sig.shape)
        self.entry_size = self.SMACrossSignal * 0.95
        # print("shape of entry_size", self.entry_size.shape)
        # self.set_signal(entry_size=self.entry_size)
        self.set_trailing_sl(8)
        self.set_atr_periods(50)

    def next(self):
        super().next()

        last_data_index = self.data.df.index[-1].timestamp()
        # print("Type of self.SMACrossSignal is", type(self.SMACrossSignal))
        # print("market_breadth_indicator", self.market_breadth_indicator)
        # Check if there is a buy signal
        if self.market_breadth_indicator == 1:
            if self.position.is_short:
                self.position.close()
            if self.market_breadth_flag[-1] < 1:
                self.buy(size=0.8)
            else:
                self.buy(size=0.2)
            self.last_trade_index = last_data_index  # Update last trade index

        # Check if there is a sell signal
        elif self.market_breadth_indicator == -1:
            if self.position.is_long:
                self.position.close()
            if self.market_breadth_flag[-1] > -1:
                self.sell(size=0.2)
            else:
                self.sell(size=0.8)
            self.last_trade_index = last_data_index  # Update last trade index


import pandas_ta as pta


class DivergenceStrategy(SignalStrategy, TrailingStrategy):
    rsi_period = 20
    sma_period = 10
    cooldown = 5  # number of bars to wait after a trade
    long_weight = 0.8
    short_weight = 0.2
    divergence_threshold = 50  # Threshold to define divergence between price and RSI
    threshold = 50  # Threshold of market width change

    def init(self):
        super().init()

        self.rsi = pta.rsi(self.data.df["Close"], length=self.rsi_period)
        self.sma = self.I(SMA, self.data.Close, self.sma_period)

        # Divergence: 1 for bullish divergence, -1 for bearish divergence
        self.market_breadth_larger_sma50 = self.data.df[
            "constituent-sma50_larger_price"
        ]
        self.market_breadth_smaller_sma50 = self.data.df[
            "constituent-sma50_smaller_price"
        ]

        self.market_sma_50 = self.data.df["market-sma50_smaller_price"] / (
            self.data.df["market-sma50_larger_price"]
            + self.data.df["market-sma50_smaller_price"]
        )
        # print("Self.RSI", self.rsi.describe())
        self.divergence = pd.Series(0, index=self.data.index)

        self.divergence[
            (self.rsi.shift() < 100 - self.threshold)
            & (self.market_sma_50 > self.market_sma_50.shift())
        ] = 1
        self.divergence[
            (self.rsi.shift() > self.threshold)
            & (self.market_sma_50 < self.market_sma_50.shift())
        ] = -1

        self.buy_sig = (self.divergence.shift() != 1) & (self.data.Close > self.sma)
        self.sell_sig = self.divergence == -1

        # self.index_trend = self.I(lambda x: x, self.data.df['Close'].pct_change(), name = "index_trend")
        # self.market_sma_50 = self.I(lambda x: x, self.market_sma_50, name = "market_sma_50")
        self.divergence = self.I(lambda x: x, self.divergence, name="divergence")
        self.buy_sig = self.I(lambda x: x, self.buy_sig, name="buy_sig")
        self.sell_sig = self.I(lambda x: x, self.sell_sig, name="sell_sig")

        self.entry_size = self.buy_sig * 0.85 - self.sell_sig * 0.8

        self.set_signal(entry_size=self.entry_size)
        self.set_trailing_sl(8)
        self.set_atr_periods(50)

    def next(self):
        super().next()

        last_data_index = self.data.index[-1]

        # Check if there is a buy signal
        if self.buy_sig[-1]:
            if self.position.is_short:
                self.position.close()
            self.buy()

        # Check if there is a sell signal
        elif self.sell_sig[-1]:
            if self.position.is_long:
                self.position.close()
            self.sell()

        self.last_trade_index = last_data_index  # Update last trade index
