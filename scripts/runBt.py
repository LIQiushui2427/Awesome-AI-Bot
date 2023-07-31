import os
import backtrader as bt
import pandas as pd
pd.set_option('display.max_columns', None)
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.tradeListAnalyzer import TradeListAnalyzer
from utils.keyIndicatorAnalyzer import KeyIndicatorAnalyzer

from data_feeds.CoT_Dissag import DataFeedForAI
from strategies.AIStrategy import AIStrategy
# matplotlib.use('QT5Agg')

def runBt(datapath:str, ticker:str, mode:str, end_date:str):
    # output_path = os.path.join(os.path.dirname(os.getcwd()), 'outputsByBt') # debug mode
    output_path = os.path.join(os.getcwd(), 'outputsByBt') # app mode
    
    print("Running backtest for", ticker, "in mode", mode, "until", end_date, "...")
    
    output_file = os.path.join(output_path, f'{ticker}_{mode}_{end_date}.txt') if mode != '' else os.path.join(output_path, f'{ticker}_{end_date}.txt')

    # if os.path.exists(output_file):
    #     print("Have already run backtest for", ticker, "in mode", mode, "until", end_date, "skipping...")
    #     return
    if os.path.exists(output_file):
        os.remove(output_file)
    cerebro = bt.Cerebro()
    df = pd.read_csv(datapath, index_col=0)
    df.index = pd.to_datetime(df.index)
    # df

    data = DataFeedForAI(dataname=df)

    cerebro.addstrategy(AIStrategy)

    cerebro.adddata(data)

    # 初始资金 100,000,000
    cerebro.broker.setcash(1e5)
    # 佣金，双边各 0.0003
    cerebro.broker.setcommission(commission=0.0003)
    # 滑点：双边各 0.0001
    cerebro.broker.set_slippage_perc(perc=0.0001)
    
    cerebro.addsizer(bt.sizers.PercentSizer, percents = 90)

    cerebro.addobserver(bt.observers.BuySell, csv = False) # 买卖记录
    cerebro.addobserver(bt.observers.Value, csv = False) # 账户价值
    cerebro.addobserver(bt.observers.TimeReturn) # 收益率
    


    cerebro.addanalyzer(TradeListAnalyzer,  _name="trade_list")
    # cerebro.addanalyzer(KeyIndicatorAnalyzer, _name="key_indicator")
    # cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl') # 返回收益率时序数据
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='Transactions') # 交易记录
    cerebro.addanalyzer(bt.analyzers.VWR, _name='Variability-Weighted Return: Better SharpeRatio with Log Returns') # 年化收益率
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='TimeReturn') # 收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name='DrawDown', timeframe=bt.TimeFrame.Days, compression=1) # 最大回撤

    cerebro.addwriter(bt.WriterFile, csv=False, out = output_file, rounding = 4)
    # 启动回测
    result = cerebro.run(tradehistory=True)
    img = cerebro.plot(iplot= False, figsize=(15, 10), dpi=168, style='bar', barup='green', bardown='red', volume=True, vlines=True, grid=True, use='agg') # 'plotly
    img[0][0].savefig(os.path.join(output_path, f'{ticker}_{mode}_{end_date}.png'), dpi = 168) if mode != '' else img[0][0].savefig(os.path.join(output_path, f'{ticker}_{end_date}.png'), dpi = 168)
    # 从返回的 result 中提取回测结果
    strat = result[0]

    # 打印评价指标
    print("--------------- SharpeRatio -----------------")
    print(strat.analyzers.SharpeRatio.get_analysis())
    print("--------------- DrawDown -----------------")
    print(strat.analyzers.DrawDown.get_analysis())

if __name__ == '__main__':
    print('Testing runBt.py')
    # runBt(datapath = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), '^GSPC_fut_fin_2023-07-26.csv'),
    #       ticker = '^GSPC' , mode = 'fut_fin', end_date = '2023-07-26')
    runBt(datapath = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), 'GC=F_com_disagg_2023-07-31.csv'),
        ticker = 'GC=F' , mode = 'com_disagg', end_date = '2023-07-31')
    