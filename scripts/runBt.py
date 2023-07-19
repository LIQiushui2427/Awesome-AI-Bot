import os
import backtrader as bt
import pandas as pd
import matplotlib
# matplotlib.use('QT5Agg')

def runBt(datapath:str, ticker:str, mode:str, end_date:str):
    postfix_1 = 'BT'
    output_path = os.path.join(os.getcwd(), 'outputsByBt')
    print("Running backtest for data stroed at:", datapath)
    output_file = os.path.join(output_path, f'{ticker}_{mode}_{end_date}_{postfix_1}.csv')

    # if os.path.exists(output_file):
    #     print("Have already run backtest for", ticker, "in mode", mode, "until", end_date, "skipping...")
    #     return
    if os.path.exists(output_file):
        os.remove(output_file)
    cerebro = bt.Cerebro()
    print(f'C://Users//lqs//OneDrive - The Chinese University of Hong Kong/projects/outputsByAI/^GSPC_fut_fin_2023-07-18_processed.csv')
    df = pd.read_csv(datapath, index_col=0)
    df.index = pd.to_datetime(df.index)
    # df


    from data_feeds.CoT_Dissag import DataFeedForAI
    from strategies.AIStrategy import AIStrategy

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

    cerebro.addobserver(bt.observers.BuySell) # 买卖交易点
    cerebro.addobserver(bt.observers.Value) # 价值


    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl') # 返回收益率时序数据
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn') # 年化收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio') # 夏普比率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown') # 回撤

    cerebro.addwriter(bt.WriterFile, csv=True, out = output_file)
    
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

if __name__ == '__main__':
    print('Testing runBt.py')
    runBt(datapath = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), '^GSPC_fut_fin_2023-07-18_processed.csv'),
          ticker = '^GSPC' , mode = 'fut_fin', end_date = '2023-07-18')
    