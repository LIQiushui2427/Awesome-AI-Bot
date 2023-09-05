import pandas as pd
import re


def get_trades(file_path):
    """Use regular expression to get trades from file_path."""
    date_regex = r"datetime.date"
    size_regex = r"-?[0-9]+\.[0-9]+"
    sharpeRatio_regex = r"(?<=sharperatio: )-?\d*.\d*"
    winRate_regex = r"(?<=winRate: )\d*.\d*"
    profitFactor_regex = r"(?<=profitFactor: )-?\d*.\d*"
    tradesPerYear_regex = r"(?<=tradesPerYear: )\d*.\d*"

    sharpeRatio_ = None
    winRate_ = None
    date_ = None
    size_ = None
    profitFactor_ = None
    tradesPerYear_ = None

    for line in open(file_path):
        m = re.search(date_regex, line)
        l = re.search(sharpeRatio_regex, line)
        w = re.search(winRate_regex, line)
        p = re.search(profitFactor_regex, line)
        t = re.search(tradesPerYear_regex, line)
        if p:
            profitFactor_ = p.group(0)
        if t:
            tradesPerYear_ = t.group(0)
        if w:
            winRate_ = w.group(0)
        if l:
            sharpeRatio_ = l.group(0)
        if m:
            # print(line)
            date = re.search(r"\d{4}-\d{2}-\d{2}", line)
            # print(date.group(0))
            size = re.search(size_regex, line)
            # print()
            # print(size.group(0))
            date_, size_ = date.group(0), size.group(0)

    # print("Size of the trade: ", size_)
    return date_, size_, sharpeRatio_, winRate_, profitFactor_, tradesPerYear_


def get_signals(file_path):
    """Get signal, last trade signal and last close price from file_path."""
    df = pd.read_csv(file_path)
    # print(df.head(10))
    # get the last 7 lines in last 7 days

    df_ = df.iloc[6:, -7:]

    weights_in_day = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.2]
    weights_between_days = [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.45]

    # calculate weighted average of the last 7 days, and assign signal to df['signal']
    for i in range(df_.shape[0]):
        df.loc[i + 7, "weighted"] = sum(df_.iloc[i, :] * weights_in_day)

    # calculate weighted average of the last 7 days, and assign signal to df['signal']
    for i in range(df_.shape[0] - 1):
        df.loc[i + 7, "signal"] = sum(df.iloc[i : i + 7, -2] * weights_between_days)

    df.dropna(subset=["date"], inplace=True)

    df.fillna(0, inplace=True)
    # print(df.tail(10))
    # print(df.head(10))
    # print(df.shape)

    df.to_csv(file_path, index=False)

    # return the last signal
    today_signal = df.iloc[-1, -1]
    # print("today_signal: ", today_signal)
    # return the last day close price as the buy/sell price for today
    yesterday_close = df.iloc[-1, 4]
    # print(yesterday_close)
    # return the first date
    bt_start_date = df.iloc[0, 0]
    bt_end_date = df.iloc[-1, 0]
    return today_signal, yesterday_close, bt_start_date, bt_end_date


if __name__ == "__main__":
    # pass
    get_signals(
        "C:\\Users\\lqs\\Downloads\\CoT_Strategy\\outputsByAI\\AAPL_2023-08-28.csv"
    )
    # get_signals('C:\\Users\\lqs\\Downloads\\CoT_Strategy\\outputsByAI\\^GSPC_fut_fin_2023-07-27.csv')
    # get_signals('C:\\Users\\lqs\\Downloads\\CoT_Strategy\\outputsByAI\\^HSI_2023-08-15.csv')
    # get_trades('C:\\Users\\lqs\\Downloads\\CoT_Strategy\\outputsByBt\\^HSI_2023-08-28.txt')
    # get_trades('C:\\Users\\lqs\\Downloads\\CoT_Strategy\\outputsByBt\\^GSPC_fut_fin_2023-08-09.txt')
