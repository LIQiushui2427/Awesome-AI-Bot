# %%
import sys
import os
# setting path
sys.path.append('C:/Users/lqs/OneDrive - The Chinese University of Hong Kong/projects')
from data_fetchers.data_fetchers_for_Cot import *

# %%
df = fetchers_for_com_disagg(output_dir = os.path.join(os.path.dirname(os.getcwd()), 'data'), output_file = 'GOLD.csv', yf_code = 'GC=F', cftc_market_code = '088691',   start_date = '2015-01-01', end_date='2023-6-27')
df

# %%
df = fetcher_for_fut_disgg(output_dir = os.path.join(os.path.dirname(os.getcwd()), 'data'), output_file = 'GOLD.txt', yf_code = 'GC=F', cftc_market_code = '088691',   start_date = '2022-01-01', end_date='2022-12-31')

# %%
df = fetchers_for_Traders_Finance_Futures(output_dir = os.path.join(os.path.dirname(os.getcwd()), 'data'), output_file = 'S&P.txt', yf_code = '^GSPC', cftc_market_code = '13874+',   start_date = '2022-01-01', end_date='2022-12-31')
df
# %%
df = fetchers_for_Traders_Finance_Combined(output_dir = os.path.join(os.path.dirname(os.getcwd()), 'data'), output_file = 'S&P.txt', yf_code = '^GSPC', cftc_market_code = '13874+',   start_date = '2022-01-01', end_date='2022-12-31')
df


