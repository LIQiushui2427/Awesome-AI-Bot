from data_fetchers.data_fetchers_for_Cot import *
from scriptControllers.singleRunController import basicSingleRun
from scripts.runAI import *
from scripts.runBt import *

TICKER_LIST = {
    'GC=F' : 'com_disagg',
    '^GSPC': '',
    '^DJI' : '',
    '^IXIC' : '',
}

def basicMultiRun(ticker_list: dict = TICKER_LIST, start_date = "2015-01-01", end_date = "2021-01-01"):
    """
    A basic multi run with respect to basicSingleRun.
    It will iterate through all the tickers and modes in TICKER_LIST.
    """
    
    print("Running basicMultiRun with TICKER_LIST:", ticker_list)
    for ticker, mode in ticker_list.items():
        basicSingleRun(ticker, mode, start_date, end_date)
    print("basicMultiRun finished.")

if __name__ == "__main__":
    basicMultiRun(ticker = "^GSPC", mode = "com_disagg",
            start_date = "2015-01-01",
            end_date = "2021-01-01")
    print("multiRunController.py executed.")