# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from apis.api_datalouder import *
import datetime as dt
# %%
def get_all_market_breadth(token = API_TOKEN, start_date = '20150101', end_date = dt.datetime.today().strftime('%Y%m%d')):
    """
    Get all market breadth data from start_date to end_date, stored in a dict.\
    input data should be in format of 'YYYYMMDD'.
    """
    market_list = ['HK#HSI', 'US#IXIC', 'US#GSPC', 'US#DJI']
    
    for market in market_list:
        print(f"Caller: Querying {market}..." + "with api token", token)
        df =  pd.DataFrame(query_market_breadth(api_token = token, partial_index_id = market, end_date = end_date))
        df = df[df['dt'] <= end_date]
        df = df[df['dt'] >= start_date]
        # add market name to column name
        for col in df.columns:
            if col != 'dt':
                df.rename(columns={col: market + '_' + col}, inplace=True)
        if market == market_list[0]:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on='dt', how='outer')
    # change all dt index to format of 'YYYY-MM-DD'
    df_all['dt'] = pd.to_datetime(df_all['dt'], format='%Y%m%d')
    # rename dt column to date
    df_all.rename(columns={'dt': 'Date'}, inplace=True)
    df_all.set_index('Date', inplace=True)
    # sort by date
    df_all.sort_index(inplace=True)
    # fill na with previous value
    df_all.fillna(method='ffill', inplace=True)
    df_all.fillna(0, inplace=True)
    dir_name = os.path.join(os.getcwd(), 'data')
    path = os.path.join(dir_name, 'market_breadth.csv')
    df_all.to_csv(path)
    print("Market breadth of date", end_date, "has been saved to", path)
    return path
if __name__ == '__main__':
    api_token = get_token()
    df = get_all_market_breadth()
