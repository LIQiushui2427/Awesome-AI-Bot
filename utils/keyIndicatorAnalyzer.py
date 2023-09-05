import backtrader as bt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)


class KeyIndicatorAnalyzer(bt.Analyzer):
    """ """

    def __init__(self):
        super(KeyIndicatorAnalyzer, self).__init__()
        #  period
        self.year_period = 252
        #  period
        self.month_period = 21
        #  period
        self.week_period = 5

        #
        self.daily_details = []
        #
        self.commission = 0

        #
        self.win_list = []
        #
        self.loss_list = []

        #
        self.key_indicators_df = pd.DataFrame(
            columns=["", "", "", "", "", "", "", "7", "30", "", ""]
        )
        # ，，{：DataFrame, ：DataFrame}，，
        self.daily_chart_dict = dict()

    def get_analysis_data(self, benchmark_df, benchmark_name):
        """
        ，，。
        @param benchmark_df:
        @param benchmark_name:
        """
        self._calculate_benchmark_indicators(benchmark_df, benchmark_name)
        return self.key_indicators_df, self.daily_chart_dict

    def _calculate_benchmark_indicators(self, benchmark_df, benchmark_name):
        """ """
        series = benchmark_df["close"]
        total_return = self.total_return(series)
        annual_return = self.annual_return(series)
        period = self.week_period
        recent_7_days_return = self.recent_period_return(series, period)
        period = self.month_period
        recent_30_days_return = self.recent_period_return(series, period)
        max_drawdown = self.max_drawdown(series)
        sharp_ratio = self.sharp_ratio(series)
        self.key_indicators_df.loc[len(self.key_indicators_df)] = [
            benchmark_name,
            total_return,
            annual_return,
            max_drawdown,
            None,
            sharp_ratio,
            None,
            recent_7_days_return,
            recent_30_days_return,
            None,
            None,
        ]
        #
        df = pd.DataFrame(index=benchmark_df.index)
        s = self.yield_curve(series)
        #
        df.insert(0, "", s)
        df.index.name = ""
        self.daily_chart_dict[benchmark_name] = df

    def next(self):
        super(KeyIndicatorAnalyzer, self).next()
        #
        current_date = self.strategy.data.datetime.date(0)
        #
        total_value = self.strategy.broker.getvalue()
        #
        cash = self.strategy.broker.getcash()
        self.daily_details.append({"": current_date, "": total_value, "": cash})

    def notify_trade(self, trade):
        #
        if trade.isclosed:
            #
            self.commission += trade.commission
            #
            if trade.pnlcomm >= 0:
                # ， 0
                self.win_list.append(trade.pnlcomm)
            else:
                #
                self.loss_list.append(trade.pnlcomm)

    def stop(self):
        #
        if self._win_times() + self._loss_times() == 0:
            win_rate = 0
        else:
            win_percent = self._win_times() / (self._win_times() + self._loss_times())
            win_rate = f"{round(win_percent * 100, 2)}%"

        df = pd.DataFrame(self.daily_details)

        #
        total_return = self.total_return(df[""])

        #
        annual_return = self.annual_return(df[""])

        # 7
        period = self.week_period
        recent_7_days_return = self.recent_period_return(df[""], period)

        # 30
        period = self.month_period
        recent_30_days_return = self.recent_period_return(df[""], period)

        #
        max_drawdown = self.max_drawdown(df[""])
        #
        sharp_ratio = self.sharp_ratio(df[""])

        #
        kelly_percent = self.kelly_percent()

        #
        commission_percent = self.commission_percent(df[""])

        #
        trade_times = self._win_times() + self._loss_times()

        #
        self.key_indicators_df.loc[len(self.key_indicators_df)] = [
            "",
            total_return,
            annual_return,
            max_drawdown,
            win_rate,
            sharp_ratio,
            kelly_percent,
            recent_7_days_return,
            recent_30_days_return,
            commission_percent,
            trade_times,
        ]

        #
        df[""] = self.yield_curve(df[""])
        df.set_index("", inplace=True)
        #
        self.daily_chart_dict[""] = df

    def commission_percent(self, series) -> str:
        """ """
        percent = self.commission / series.iloc[0]
        return f"{round(percent * 100, 2)}%"

    def yield_curve(self, series) -> pd.Series:
        """ """
        percent = (series - series.iloc[0]) / series.iloc[0]
        return round(percent * 100, 2)

    def total_return(self, series) -> str:
        """ """
        percent = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
        return f"{round(percent * 100, 2)}%"

    def annual_return(self, series) -> str:
        """ """
        percent = (
            (series.iloc[-1] - series.iloc[0])
            / series.iloc[0]
            / len(series)
            * self.year_period
        )
        return f"{round(percent * 100, 2)}%"

    def recent_period_return(self, series, period) -> str:
        """ """
        percent = (series.iloc[-1] - series.iloc[-period]) / series.iloc[-period]
        return f"{round(percent * 100, 2)}%"

    def max_drawdown(self, series) -> str:
        """ """
        s = (series - series.expanding().max()) / series.expanding().max()
        percent = s.min()
        return f"{round(percent * 100, 2)}%"

    def sharp_ratio(self, series) -> float:
        """

        ：，（）
        ，，。
        ，，。
        ，，。
        ，，。
        ，，1.0。
        ：(Rp-Rf)/σp
        ，Rp，Rf，σp。
        3%
        ：sharpe = ( - ) /
        """
        ret_s = series.pct_change().fillna(0)
        avg_ret_s = ret_s.mean()
        avg_risk_free = 0.03 / self.year_period
        sd_ret_s = ret_s.std()
        sharp = (avg_ret_s - avg_risk_free) / sd_ret_s
        sharp_year = round(np.sqrt(self.year_period) * sharp, 3)
        return sharp_year

    def kelly_percent(self) -> str:
        """

        ：，，
        ，。
        ：K = W - [(1 - W) / R]
        ，K，W，R，。
        ：，，；，
         kelly_percent = 0.2，20%。
        ，
        """
        win_times = self._win_times()
        loss_times = self._loss_times()
        if win_times > 0 and loss_times > 0:
            avg_win = np.average(self.win_list)  #
            avg_loss = abs(np.average(self.loss_list))  # ，
            win_loss_ratio = avg_win / avg_loss  #
            if win_loss_ratio == 0:
                kelly_percent = None
            else:
                sum_trades = win_times + loss_times
                win_percent = win_times / sum_trades  #
                #
                #
                kelly_percent = win_percent - ((1 - win_percent) / win_loss_ratio)
        else:
            kelly_percent = None  #

        return f"{round(kelly_percent * 100, 2)}%" if kelly_percent else None

    def _win_times(self):
        """ """
        return len(self.win_list)

    def _loss_times(self):
        """ """
        return len(self.loss_list)
