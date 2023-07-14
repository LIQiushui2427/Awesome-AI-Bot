from testrunai import *
from data_fetchers.data_fetchers_for_Cot import *
import os
import argparse

# parser = argparse.ArgumentParser()

# parser.add_argument()
print("Starting crawling data... if already existed, skip")
fetchers_for_com_disagg(output_dir = os.path.join((os.getcwd()), 'data'), output_file = 'GOLD.csv', yf_code = 'GC=F', cftc_market_code = '088691',   start_date = '2014-01-01', end_date='2023-7-12')
print("Start training AI...")
trainAI()
print("Done...")