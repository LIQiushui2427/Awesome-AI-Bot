import os
import backtrader as bt
import pandas as pd
pd.set_option('display.max_columns', None)
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.tradeListAnalyzer import TradeListAnalyzer
from utils.keyIndicatorAnalyzer import KeyIndicatorAnalyzer
from utils.getSignals import get_signals
from utils.utils import saveplots
from data_feeds.CoT_Dissag import DataFeedForAI
from strategies.AIStrategy import AIStrategy
from utils.basicStats import BasicTradeStats
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

    #  100,000
    cerebro.broker.setcash(1e5)
    # ， 0.0003
    cerebro.broker.setcommission(commission=0.0003)
    # ： 0.0001
    cerebro.broker.set_slippage_perc(perc=0.0001)
    
    cerebro.addsizer(bt.sizers.PercentSizer, percents = 80)

    cerebro.addobserver(bt.observers.BuySell, csv = False) # 
    cerebro.addobserver(bt.observers.Value, csv = False) # 
    # cerebro.addobserver(bt.observers.TimeReturn) # 
    


    cerebro.addanalyzer(TradeListAnalyzer,  _name="trade_list")
    cerebro.addanalyzer(KeyIndicatorAnalyzer, _name="key_indicator")
    cerebro.addanalyzer(BasicTradeStats, _name="basic_trade_stats")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='Transactions') # 
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='AnnualReturn') # 
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='TimeReturn', timeframe=bt.TimeFrame.Days, compression=1) # 
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio', timeframe=bt.TimeFrame.Days, compression=1, annualize = True)
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name='DrawDown', timeframe=bt.TimeFrame.Days, compression=1) # 

    cerebro.addwriter(bt.WriterFile, csv=False, out = output_file, rounding = 4)
    # 
    result = cerebro.run(tradehistory=True)
    
    figs = saveplots(cerebro=cerebro, file_path=os.path.join(output_path, f'{ticker}_{mode}_{end_date}.png', ), dpi = 720, style='bar', barup='green', bardown='red', volume=True, vlines=True, grid=True) \
        if mode != '' else saveplots(cerebro=cerebro, file_path=os.path.join(output_path, f'{ticker}_{end_date}.png'), dpi=720, style='bar', barup='green', bardown='red', volume=True, vlines=True, grid=True)
    
    # img = cerebro.plot(iplot= False, figsize=(16, 12), dpi=280, style='bar', barup='green', bardown='red', volume=True, vlines=True, grid=True, use='TkAgg') # 'plotly
    # img[0][0].savefig(os.path.join(output_path, f'{ticker}_{mode}_{end_date}.png'), dpi = 280) if mode != '' else img[0][0].savefig(os.path.join(output_path, f'{ticker}_{end_date}.png'), dpi = 280)
    #  result 
    strat = result[0]

    # 
    print("--------------- SharpeRatio -----------------")
    print(strat.analyzers.SharpeRatio.get_analysis())
    print("--------------- DrawDown -----------------")
    print(strat.analyzers.DrawDown.get_analysis())

if __name__ == '__main__':
    print('Testing runBt.py')
    runBt(datapath = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), '^GSPC_fut_fin_2023-08-29.csv'),
          ticker = '^GSPC' , mode = 'fut_fin', end_date = '2023-08-29')
    runBt(datapath = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), 'GC=F_com_disagg_2023-08-29.csv'),
        ticker = 'GC=F' , mode = 'com_disagg', end_date = '2023-08-29')

    # processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), 'BILI_2023-08-29.csv')
    # get_signals(file_path=processedDataByAI)
    # runBt(datapath = processedDataByAI,
    #     ticker = 'BILI' , mode = '', end_date = '2023-08-29')    
    
    processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), 'AAPL_2023-08-29.csv')
    get_signals(file_path=processedDataByAI)
    runBt(datapath = processedDataByAI,
        ticker = 'AAPL' , mode = '', end_date = '2023-08-29')
    
    processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), '^HSCE_2023-08-29.csv')
    get_signals(file_path=processedDataByAI)
    runBt(datapath = processedDataByAI,
        ticker = '^HSCE' , mode = '', end_date = '2023-08-29')
    
    
    processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), 'GC=F_com_disagg_2023-08-29.csv')
    get_signals(file_path=processedDataByAI)
    runBt(datapath = processedDataByAI,
        ticker = 'GC=F' , mode = 'com_disagg', end_date = '2023-08-29')
    
    processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), '0388.HK_2023-08-29.csv')
    get_signals(file_path=processedDataByAI)
    runBt(datapath = processedDataByAI,
        ticker = '0388.HK' , mode = '', end_date = '2023-08-29')
    
    processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), '^GSPC_fut_fin_2023-08-29.csv')
    get_signals(file_path=processedDataByAI)
    runBt(datapath = processedDataByAI,
        ticker = '^GSPC' , mode = 'fut_fin', end_date = '2023-08-29')
    
    processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), '^HSI_2023-08-29.csv')
    get_signals(file_path=processedDataByAI)
    runBt(datapath = processedDataByAI,
        ticker = '^HSI' , mode = '', end_date = '2023-08-29')
    
    processedDataByAI = os.path.join(os.path.join(os.getcwd(), 'outputsByAI'), 'TSLA_2023-08-29.csv')
    get_signals(file_path=processedDataByAI)
    runBt(datapath = processedDataByAI,
        ticker = 'TSLA' , mode = '', end_date = '2023-08-29')
    