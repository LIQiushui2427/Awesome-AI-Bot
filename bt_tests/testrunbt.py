import sys
import os
# sys.path.append(os.path.abspath('..')) # add parent folder to path
import backtrader as bt
import pandas as pd
import matplotlib
# matplotlib.use('QT5Agg')

# %%
dataname='../data/Temp.csv'

cerebro = bt.Cerebro()
df = pd.read_csv(dataname, index_col=0)
df.index = pd.to_datetime(df.index)
# df

# %%
from data_feeds.CoT_Dissag import DataFeedForAI
from strategies.AIStrategy import AIStrategy

data = DataFeedForAI(dataname=df)

cerebro.addstrategy(AIStrategy)

cerebro.adddata(data)

# 初始资金 100,000,000
cerebro.broker.setcash(2e3)
# 佣金，双边各 0.0003
cerebro.broker.setcommission(commission=0.0003)
# 滑点：双边各 0.0001
cerebro.broker.set_slippage_perc(perc=0.0001)

cerebro.addobserver(bt.observers.BuySell) # 买卖交易点
cerebro.addobserver(bt.observers.Value) # 价值


cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl') # 返回收益率时序数据
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn') # 年化收益率
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio') # 夏普比率
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown') # 回撤

# %%
# 启动回测
result = cerebro.run()
cerebro.plot(iplot= False)
# 从返回的 result 中提取回测结果
strat = result[0]
# 返回日度收益率序列
daily_return = pd.Series(strat.analyzers.pnl.get_analysis())
# 打印评价指标
print("--------------- AnnualReturn -----------------")
print(strat.analyzers._AnnualReturn.get_analysis())
print("--------------- SharpeRatio -----------------")
print(strat.analyzers._SharpeRatio.get_analysis())
print("--------------- DrawDown -----------------")
print(strat.analyzers._DrawDown.get_analysis())


