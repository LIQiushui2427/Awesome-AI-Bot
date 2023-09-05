from data_fetchers.data_fetchers_for_Cot import *
from scriptControllers.singleRunController import basicSingleRun
from scripts.runAI import *
from scripts.runBt import *
from scripts.sendTgMsg import *
import datetime as dt
from data_fetchers.market_breadth import get_all_market_breadth
from utils.dict import *
from apis.api_datalouder import *


def basicMultiRun(
    ticker_list: dict = TICKER_LIST,
    start_date=dt.datetime(2015, 1, 1),
    end_date=dt.datetime.today(),
):
    """
    A basic multi run with respect to basicSingleRun.
    It will iterate through all the tickers and modes in TICKER_LIST.
    """

    print("Running basicMultiRun with TICKER_LIST:", ticker_list)
    # Get market breadth data
    get_all_market_breadth(
        token=get_token(LOGIN, PASSWORD),
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    for ticker, mode in ticker_list.items():
        basicSingleRun(ticker, mode, start_date, end_date)
    daily(end_date.strftime("%Y-%m-%d"))
    print("basicMultiRun finished.")


if __name__ == "__main__":
    basicMultiRun(
        ticker="^GSPC",
        mode="com_disagg",
        start_date=dt.datetime(2015, 1, 1),
        end_date=dt.datetime.today(),
    )
