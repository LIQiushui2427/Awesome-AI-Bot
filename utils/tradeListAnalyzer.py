import backtrader as bt
import pandas as pd
pd.set_option('display.max_columns', None)

class TradeListAnalyzer(bt.Analyzer):
    """
    交易列表分析器
    https://community.backtrader.com/topic/1274/closed-trade-list-including-mfe-mae-analyzer/2
    """

    def __init__(self):
        self.trades = []
        self.cum_profit = 0.0

    def get_analysis(self) -> tuple:
        """
        获取分析数据
        @return: 交易订单列表，交易日期
        """
        trade_list_df = pd.DataFrame(self.trades)
        return trade_list_df, self._get_trade_date(trade_list_df)

    def _get_trade_date(self, trade_list_df):
        """
        获取交易日期
        @return: 交易日期，获取某只股票的买卖日期，
        返回字典，key为股票名，value为(买入日期列表，卖出日期列表)
        """
        trade_dict = dict()
        if not trade_list_df.empty:
            # 分组，找出买卖日期
            grouped = trade_list_df.groupby('stock')
            for name, group in grouped:
                buy_date_list = list(group['in_date'])
                sell_date_list = list(group['out_date'])
                # 判断是否有买卖日期
                if trade_dict.get(name) is None:
                    trade_dict[name] = (buy_date_list, sell_date_list)
                else:
                    trade_dict[name][0].extend(buy_date_list)
                    trade_dict[name][1].extend(sell_date_list)
        return trade_dict

    def notify_trade(self, trade):
        if trade.isclosed:

            total_value = self.strategy.broker.getvalue()

            dir = 'short'
            if trade.history[0].event.size > 0: dir = 'long'

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
            pbar = pnl / barlen
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
            if dir == 'long':
                mfe = hp
                mae = lp
            if dir == 'short':
                mfe = -lp
                mae = -hp

            self.trades.append(
                {'order': trade.ref,
                 'stock': trade.data._name,
                 # 'dir': dir,
                 'in_date': datein,
                 'buy_price': round(pricein, 2),
                 'out_date': dateout,
                 'sell_price': round(priceout, 2),
                 'return%': round(pcntchange, 2),
                 'profit': round(pnl, 2),
                 'profit/equity%': round(pnlpcnt, 2),
                 'size': size,
                 'value': round(value, 2),
                 'value/tot_value%': round(value / total_value * 100, 2),
                 'cum_profit': round(self.cum_profit, 2),
                 'barlen': barlen,  # 以每根 bar 的时间为单位，这里按天计算
                 # 'pnl/bar': round(pbar, 2),
                 'max_profit%': round(mfe, 2),
                 'max_loss%': round(mae, 2)})