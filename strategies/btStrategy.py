import backtrader as bt
import backtrader.indicators as btind


class MyStrategy(bt.Strategy):
    params = (
        ("period", 15),
        ("printlog", False),
    )

    def __init__(self):
        print("init")
        print("---------  self.datas  lines ----------")
        print(self.data0.lines.getlinealiases())
        # print("Type of self.data: {}".format(type(self.data)))
        # print(inspect.getmembers(self.data))

        self.sma = btind.SimpleMovingAverage(period=15)

        self.M_Money_Long = btind.BollingerBands(
            self.data0.Pct_of_OI_M_Money_Long_All,
            period=50,
            devfactor=1,
            plotname="Bollinger_Pct_M_Money_Long",
            subplot=True,
        )
        self.M_Money_Short = btind.BollingerBands(
            self.data0.Pct_of_OI_M_Money_Short_All,
            period=50,
            devfactor=1,
            plotname="Bollinger_Pct_M_Money_Short",
            subplot=True,
        )

        self.Fib = btind.FibonacciPivotPoint(self.data)

        self.rsi2 = btind.RSI(
            self.data0.close, period=3, plotname="RSI4", subplot=False
        )
        self.rsi12 = btind.RSI(
            self.data0.close, period=20, plotname="RSI14", subplot=False
        )

        self.SMACross = btind.CrossOver(
            self.rsi2, self.rsi12, plotname="SMACross", subplot=True
        )

    def next(self):
        # if rsi12 > 70 and M_Money_short breaks the bottom line of the Bollinger Band, buy
        if self.SMACross[0] > 0 and self.data0.Pct_of_OI_M_Money_Long_All[0] > 20:
            if self.position.size > 0:
                self.close()
            else:
                self.sell()

        if self.SMACross[0] < 0 and self.data0.Pct_of_OI_M_Money_Short_All[0] > 20:
            if self.position.size < 0:
                self.close()
            else:
                self.buy()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def log(self, txt, dt=None, doprint=False):
        """Logging function fot this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))
