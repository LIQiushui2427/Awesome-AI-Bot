# Analysis report 

## By YI Jian on 16/05/2023

This is an analysis report on SHI data. First I'll analysis the trend, then I will give a combined analysis based on the structure of HSI. Then I am going to compare the bullish/bearish voting.

### Introduction

The HSI, an essential price index for the Hong Kong Stock Exchange, is meticulously calculated and updated in real-time every two seconds. This market capitalization-weighted index categorizes its constituents into finance, properties, utilities, and commerce and industry sectors. Notably, prominent Chinese companies constitute a significant portion of the index's largest constituents. Consequently, the HSI serves as a reliable indicator of Hong Kong stock market quotes, providing valuable insights into market trends and performances. The collected data records the open, close, high, low of the index. Additionally, votes from Facebook users every premarket time of the day is collected, which is utilized for the proposed strategy.

### Overall trend analysis

<img src="C:\Users\lqs\OneDrive - The Chinese University of Hong Kong\projects\Figure_0.png" alt="Overall" style="zoom:150%;" />

The Hang Seng Index has been experiencing an overall downward trend, with intermittent periods of price swings, temporary rallies, and declines. However, despite these fluctuations, the general direction of the index has been downwards.

The Hang Seng Index has experienced temporary price increases and decreases within the downtrend. These temporary movements may be influenced by short-term factors, market sentiment, or news events, but they haven't altered the overall downward trajectory of the index.

**Sharp Fall in 17 Feb - 14 Mar 2 in 2022:** From February 17th to March 14th, there was a sharp decline in the Hang Seng Index. This event likely had a significant impact on market participants, potentially driven by negative news, economic indicators, or geopolitical factors.

**Deep Fall Around October 2022:** There was another notable decline in the Hang Seng Index around October. The index experienced a substantial drop during this period, indicating a heightened level of selling pressure and potentially driven by economic factors, market sentiment, or global events.

**Recent Falling Trend:** After a significant rise, the Hang Seng Index is currently experiencing another phase of falling prices. This recent decline suggests a continuation of the overall bearish trend, indicating ongoing negative sentiment and selling pressure in the market.

### Target  Shareholders strategy analysis

So individual investors in Facebook may be more susceptible to the effects of broader market indices compared to non-index stocks. This means that changes in the overall market or specific index in which Facebook is included could have a significant impact on the individual investor's portfolio. Consequently, the proposed strategy should consider market trends, index movements, and their potential effects on Facebook's stock.

When individual shareholders observe HSI history records, they will make a prediction whether the HSI will increase or not. So one can gain advantage if they know the overall trend of voting head. We are going to use them as part of indicators.

### **Indicators**: 

1. **Moving Average (MA)**: Two simple moving averages (SMA) are used - one with a length of 10 and another with a length of 25. Moving averages are widely used trend-following indicators that smooth out price data over a specified period, helping to identify the overall direction of the market.
2. **Bullish Votes and Bearish Votes**: These are derived from simple moving averages calculated over different periods. The bullish votes are obtained by applying a simple moving average (SMA) with a length of 8 to the "睇升" (percentage of premarket bullish vote) data. Similarly, the bearish votes are derived using an SMA with a length of 14. These indicators provide a measure of sentiment based on the premarket bullish and bearish votes.
3. **Average True Range (ATR)**: The Average True Range is used in the trailing stop-loss component of the strategy. It measures market volatility by calculating the average range between the high and low prices over a specified period. In this strategy, the ATR is used to dynamically adjust the stop-loss level, with a multiplier of 2 applied to it.

### Proposed Strategy:

The strategy is based on a combination of moving average crossovers, bullish votes, and bearish votes. It uses two moving averages, with lengths of 10 and 25, to identify upward crossovers as buy signals. Additionally, it incorporates the bullish and bearish votes, calculated using simple moving averages with lengths of 8 and 14, respectively. The strategy aims to capture bullish market conditions while considering the sentiment indicated by the votes.



```python
class SmaCross(SignalStrategy,
               TrailingStrategy):
    n1 = 10
    n2 = 25
    n_bullish = 8
    n_bearish = 14
    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        
        print(self.data.df.睇升.head())
        # Precompute the two moving averages
        sma1 = self.I(SMA, self.data.Close, self.n1)
        sma2 = self.I(SMA, self.data.Close, self.n2)
        bullish = self.I(SMA, self.data.睇升, self.n_bullish)
        bearish = self.I(SMA, self.data.睇跌, self.n_bearish)

        # Where sma1 crosses sma2 upwards. Diff gives us [-1,0, *1*]
        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)

        
        signal += pd.Series(bullish).diff().fillna(0)  # Add bullish signal
        signal += pd.Series(bearish).shift(2).diff().fillna(0)  # Add bearish signal

        signal = signal.replace(-1, 0)  # Upwards/long only
        
        
        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        entry_size = signal * .95
                
        # Set order entry sizes using the method provided by 
        # `SignalStrategy`. See the docs.
        self.set_signal(entry_size=entry_size)
        
        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
        self.set_trailing_sl(8)
        self.set_atr_periods(50)
```



### Back testing:

We use [backtesting](https://kernc.github.io/backtesting.py/) library to run backtesting of our strategy.

below configuration for backtesting:

```python
bt = Backtest(mydata, SmaCross, cash=1e12, hedging=True,exclusive_orders=True, trade_on_close=True) # 112%
stats = bt.run()
bt.plot()
print(stats)
```

Here are the result :

![image-20230516145808340](C:\Users\lqs\AppData\Roaming\Typora\typora-user-images\image-20230516145808340.png)

The stats are:

```
Start                     2021-08-30 00:00:00
End                       2023-03-20 00:00:00
Avg. Drawdown Duration       31 days 00:00:00

# Trades                                  302

Win Rate [%]                        51.324503
Best Trade [%]                       9.081796
Worst Trade [%]                     -7.739674
Avg. Trade [%]                       0.136216
Max. Trade Duration          29 days 00:00:00
Avg. Trade Duration           2 days 00:00:00
Profit Factor                        1.236521
Expectancy [%]                       0.157421
SQN                                  1.752266
_strategy                            SmaCross
_equity_curve                             ...
_trades                           Size  En...
dtype: object
```



### **Conclusion**: 

The Hang Seng Index has been in an overall downtrend, with intermittent price swings, temporary rises, and falls. The proposed SmaCrossWithVote strategy, incorporating moving averages, bullish and bearish votes, and a trailing stop-loss, showed positive results in backtesting.

### **Risk Disclosure**: 

It's important to include a disclaimer highlighting that investing in the stock market carries risks and that the analysis provided is for informational purposes only. Encourage readers to conduct their own research and consult with a financial advisor before making any investment decisions.