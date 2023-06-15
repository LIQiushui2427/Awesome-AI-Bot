# Fetch data for market width divergence analysis. The market width data comes from datalouder api. After downloaded the data, the data will be saved in the data fold.
import sys
sys.path.append('C:/Users/lqs/OneDrive - The Chinese University of Hong Kong/projects')
import apis.api_datalouder as api
import pandas as pd


info_00700 = pd.DataFrame(api.query_price('US#^GSPC'))
# info_00700.columns : Index(['dt', 'o', 'l', 'h', 'c', 'v', 't'], dtype='object')
info_00700.rename(columns={'dt':'Date', 'o':'Open', 'l':'Low', 'h':'High', 'c':'Close', 'v':'Volume', 't':'Ticker'}, inplace=True)
info_00700.set_index('Date', inplace=True)
info_00700.index = pd.to_datetime(info_00700.index)
info_00700


market_breadth_hsi = pd.DataFrame(api.query_market_breadth('US#^GSPC'))
market_breadth_hsi.rename(columns={'dt':'Date'}, inplace=True)
market_breadth_hsi.set_index('Date', inplace=True)
market_breadth_hsi.index = pd.to_datetime(market_breadth_hsi.index)
market_breadth_hsi


assert info_00700 is not None
assert market_breadth_hsi is not None
join_df = pd.merge(info_00700, market_breadth_hsi, left_index=True, right_index=True)


join_df.head()


# save the data
join_df.to_csv('./data/US#^GSPC.csv')