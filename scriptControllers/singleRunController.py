import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_fetchers.data_fetchers_for_Cot import *
from scripts.runAI import *
from scripts.runBt import *
from scripts.sendTgMsg import *
from utils.getSignals import *
import datetime as dt


def basicSingleRun(
    ticker, mode, start_date, end_date: dt.datetime = dt.datetime.today()
):
    """
    A single run of fetching data, training AI and running backtest.
    For only one ticker and one mode, and the end_date is set to today.
    """
    print(
        "Fetching data for", ticker, "in mode", mode, "from", start_date, "to", end_date
    )
    fetch_and_update_yf(
        yf_code=ticker, mode=mode, start_date=start_date, end_date=end_date
    )
    processedDataByAI, res = trainAI(
        ticker=ticker, mode=mode, end_date=end_date.strftime("%Y-%m-%d")
    )
    get_signals(file_path=processedDataByAI)
    runBt(
        datapath=processedDataByAI,
        ticker=ticker,
        mode=mode,
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    print("Single run finished.")


if __name__ == "__main__":
    basicSingleRun(ticker="^IXIC", mode="", end_date=dt.datetime(2023, 7, 27))
