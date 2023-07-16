from data_fetchers.data_fetchers_for_Cot import *
from scripts.runAI import *
from scripts.runBt import *
def basicSingleRun(ticker, mode, start_date, end_date):
    """
    A single run of fetching data, training AI and running backtest.
    For only one ticker and one mode, and the end_date is set to today.
    """
    
    print("Fetching data for", ticker, "in mode", mode, "from", start_date, "to", end_date)
    if mode == 'Com_Disagg':
        fetchers_for_com_disagg(yf_code=ticker, cftc_market_code = '001602', start_date= start_date,end_date=end_date)
    elif mode == 'Fut_Disagg':
        fetcher_for_fut_disgg(yf_code=ticker, cftc_market_code = '001602', start_date= start_date,end_date=end_date)
    elif mode == 'TFF_Com':
        fetchers_for_Traders_Finance_Combined(yf_code=ticker, cftc_market_code = '001602', start_date= start_date,end_date=end_date)
    elif mode == 'TFF_Fut':
        fetchers_for_Traders_Finance_Futures(yf_code=ticker, cftc_market_code = '001602', start_date= start_date,end_date=end_date)
    else:
        print("Unexpected mode.")
        return
    trainAI(ticker = ticker, mode = mode, end_date = end_date)
    runBt(ticker = ticker, mode = mode, end_date = end_date)
    print("Single run finished.")