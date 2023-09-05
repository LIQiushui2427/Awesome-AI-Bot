import backtrader as bt
import pandas as pd

pd.set_option("display.max_columns", None)
import datetime as dt


class TradeListAnalyzer(bt.Analyzer):
    """

    https://community.backtrader.com/topic/1274/closed-trade-list-including-mfe-mae-analyzer/2
    """

    def __init__(self):
        self.trades = []
        self.cum_profit = 0.0

    def get_analysis(self) -> tuple:
        """

        @return: ，
        """
        trade_list_df = pd.DataFrame(self.trades)
        return trade_list_df, self._get_trade_date(trade_list_df)

    def _get_trade_date(self, trade_list_df):
        """

        @return: ，，
        ，key，value(，)
        """
        trade_dict = dict()
        if not trade_list_df.empty:
            # ，
            grouped = trade_list_df.groupby("stock")
            for name, group in grouped:
                buy_date_list = list(group["in_date"])
                sell_date_list = list(group["out_date"])
                #
                if trade_dict.get(name) is None:
                    trade_dict[name] = (buy_date_list, sell_date_list)
                else:
                    trade_dict[name][0].extend(buy_date_list)
                    trade_dict[name][1].extend(sell_date_list)
        return trade_dict

    def notify_trade(self, trade):
        total_value = self.strategy.broker.getvalue()

        dir = "short"
        if trade.history[0].event.size > 0:
            dir = "long"

        pricein = trade.history[len(trade.history) - 1].status.price
        priceout = trade.history[len(trade.history) - 1].event.price
        datein = bt.num2date(trade.history[0].status.dt)
        dateout = bt.num2date(trade.history[len(trade.history) - 1].status.dt)
        if trade.data._timeframe >= bt.TimeFrame.Days:
            datein = datein.date()
            dateout = dateout.date()

        pcntchange = 100 * priceout / pricein - 100
        pnl = trade.history[len(trade.history) - 1].status.pnlcomm
        pnlpcnt = 100 * pnl / total_value
        barlen = trade.history[len(trade.history) - 1].status.barlen
        pbar = pnl / barlen if barlen > 0 else pnl
        self.cum_profit += pnl

        size = value = 0.0
        for record in trade.history:
            if abs(size) < abs(record.status.size):
                size = record.status.size
                value = record.status.value

        highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
        lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
        hp = 100 * (highest_in_trade - pricein) / pricein
        lp = 100 * (lowest_in_trade - pricein) / pricein
        if dir == "long":
            mfe = hp
            mae = lp
        if dir == "short":
            mfe = -lp
            mae = -hp

        self.trades.append(
            {
                "stock": trade.data._name,
                "in_date": datein,
                "size": size,
                "buy_price": round(pricein, 2),
                "out_date": dateout,
                "sell_price": round(priceout, 2),
            }
        )
