from data_fetchers.data_fetchers_for_Cot import *
from scripts.runAI import *
from scripts.runBt import *
def basicSingleRun(ticker, mode, start_date, end_date):
    """
    A single run of fetching data, training AI and running backtest.
    For only one ticker and one mode, and the end_date is set to today.
    """
    print("Fetching data for", ticker, "in mode", mode, "from", start_date, "to", end_date)
    fetch_and_update_yf(yf_code=ticker, mode=mode,start_date=start_date,end_date=end_date)
    processedDataByAI = trainAI(ticker = ticker, mode = mode, end_date = end_date)
    runBt(datapath=processedDataByAI, ticker=ticker, mode=mode, end_date=end_date)
    print("Single run finished.")