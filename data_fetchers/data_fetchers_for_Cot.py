import os
import glob
import re
import requests
import zipfile
import pandas as pd
import datetime as dt
import yfinance as  yf

MAPPING = {
    '^GSPC' : '13874+',
    'GC=F'  : '001602'
}

def find_file(partial_name: str, directory: str):
    """Find file with partial name in directory.

    Args:
        partial_name (str): partial name of the file.
        directory (str): directory of the file.

    Returns:
        str: file path.
    """
    try:
        temp_path = glob.glob(os.path.join(directory, f'*{partial_name}*'))[0]
        old_date = temp_path.split('_')[-1].split('.')[0]
        return old_date
    
    except IndexError:
        return None

     
def fetch_and_update_yf(yf_code: str, mode: str,start_date, end_date) -> pd.DataFrame:
    """Fetch daily data for given ticker, mode up to end_date, update the file upto end_date, and return the dataframe.
    make sure input dates are in YYYY-MM-DD format.
    """
    assert start_date<= end_date, 'The start year should be no later than the end year.'
    output_dir = os.path.join(os.getcwd(), 'data')
    output_file = os.path.join(output_dir, f'{yf_code}_{mode}_{end_date}.csv')

    if os.path.exists(output_file):
        print(f"Already processed data for {yf_code}_{mode}_{end_date}.csv, skip")
        return
    # seach in output folder, if already exists ticker, hot load.
    old_date = find_file(partial_name = yf_code + '_' + mode, directory = output_dir)
    if old_date is not None:
        if old_date >= end_date:
            print(f"Already processed data for {yf_code}, {mode}, in {old_date}, return...")
            return
        else:
            print(f"Already processed data for {yf_code}, {mode}, in {old_date}, hot load...")
            fetch_yf_cftc(yf_code=yf_code, mode=mode,start_date=dt.date.strftime(dt.datetime.strptime(old_date, '%Y-%m-%d'), '%Y-%m-%d'), end_date=end_date, hot_load=True)
    else:
        print(f"New ticker {yf_code}, {mode}, start download...")
        fetch_yf_cftc(yf_code=yf_code,mode=mode,start_date=start_date,end_date=end_date)
        
def fetch_yf_cftc(yf_code: str, mode: str,start_date:str, end_date:str, hot_load:bool=False):
    """Fetch daily data for given ticker, mode up to end_date.
    if the same ticker has been download (the end_date in postfix is prior to input one), the script will update the file.


    Args:
        yf_code (str): yahoo code of the derivative.
        mode (str): mode of cftc. there are 4 types: com_disagg and fut_disagg(for a dissaggregated deveritives), and fut_fin and com_fin for consolidated deveritives
        start_date (dt.datetime): start date of date fetcher.
        end_date (str, optional): end date of date fetcher. Defaults to dt.datetime.today().
    """
    cftc_market_code = MAPPING[yf_code] if yf_code in MAPPING.keys() else None
    output_dir = os.path.join(os.getcwd(), 'data')
    base_url = 'https://www.cftc.gov/files/dea/history/{}_txt_{}.zip'
    if cftc_market_code is not None: # add mode to output file name
        temp_path = os.path.join(output_dir, mode + '.txt')
        end_file = os.path.join(output_dir, os.path.join(output_dir, f'{yf_code}_{mode}_{end_date}.csv'))
        start_file = os.path.join(output_dir, os.path.join(output_dir, f'{yf_code}_{mode}_{start_date}.csv'))    
    else:
        temp_path = os.path.join(output_dir, 'temp.txt')
        end_file = os.path.join(output_dir, os.path.join(output_dir, f'{yf_code}_{end_date}.csv'))
        start_file = os.path.join(output_dir, os.path.join(output_dir, f'{yf_code}_{start_date}.csv'))
        
    if isinstance(end_date, str):
        end_date_ = dt.datetime.strptime(end_date, '%Y-%m-%d')
    if isinstance(start_date, str):
        start_date_ = dt.datetime.strptime(start_date, '%Y-%m-%d')
        
    assert start_date_ >= dt.datetime(2010,1,1), 'Do not support data fetch earlier than 2010-01-01.'
    assert end_date_ <= dt.datetime.today(), 'The end date should be no later than today.'
    assert start_date_<= end_date_, 'The start date should be no later than the end date.'
    
    # if hot load, start date should be the next day of the last date of the file
    if hot_load:
        print("Hot load for date:", start_date_)
        start_date_ = start_date_ + dt.timedelta(days=1)
        end_date_ = end_date_ + dt.timedelta(days=1)
        
    # remove the temp file if exists
    if os.path.exists(temp_path):
        print(f"Remove {temp_path}")
        os.remove(temp_path)
        
    # Loop through all years from 2010 to 2023
    if cftc_market_code is not None:
        for year in range(end_date_.year, start_date_.year - 1, -1):
            print(f"Downloading data for {yf_code}, {cftc_market_code}, with mode {mode}, in {year}...")

            # Construct the URL and download path for this year
            url = base_url.format(mode,year)
            output_zip = os.path.join(output_dir, f'{mode}_txt_{year}.zip')
            # print("Output_zip:", output_zip)
            # Download the file
            r = requests.get(url)

            # Save it as a binary file
            with open(output_zip, 'wb') as f:
                f.write(r.content)

            # Open the downloaded zip file
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                # Extract all the contents into the data directory
                zip_ref.extractall(output_dir)

            # The zip file is now unzipped. You can remove the zip file if you wish:
            os.remove(output_zip)

            # Load the data from the extracted file
            # switch mode
            match (mode):
                case 'fut_disagg':
                    new_data = pd.read_csv(os.path.join(output_dir, f'f_year.txt'), low_memory=False)
                case 'com_disagg':
                    new_data = pd.read_csv(os.path.join(output_dir, f'c_year.txt'), low_memory=False)
                case 'fut_fin':
                    new_data = pd.read_csv(os.path.join(output_dir, f'FinFutYY.txt'), low_memory=False)
                case 'com_fin':
                    new_data = pd.read_csv(os.path.join(output_dir, f'FinComYY.txt'), low_memory=False)

            # Append the data to the output file
            if os.path.exists(temp_path):
                new_data.to_csv(temp_path, mode='a', header=False,index=False)
            else:
                # print("Df: ", new_data.head())
                new_data.to_csv(temp_path, mode='w', header=True,index=False)
        
        print("start_date:", start_date_, "end_date:", end_date_)
        yf_df = yf.download(yf_code, start = start_date_, end = end_date_, progress = False)
        yf_df['date'] =  yf_df.index.strftime('%Y-%m-%d')
        yf_df.index = yf_df.index.strftime('%Y-%m-%d')
        
        cftc_df = pd.read_csv(temp_path, low_memory=False)

        cftc_df = cftc_df[(cftc_df["CFTC_Contract_Market_Code"] == cftc_market_code) & (cftc_df["Report_Date_as_YYYY-MM-DD"] >= start_date)\
            & (cftc_df["Report_Date_as_YYYY-MM-DD"] <= end_date)]
        
        df = pd.merge(yf_df, cftc_df, left_index=True, right_on="Report_Date_as_YYYY-MM-DD", how='outer')
        
        df.interpolate(inplace=True, limit_direction='forward')
        df.set_index("date", inplace=True)
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    else: # just yf
        print("start_date:", start_date_, "end_date:", end_date_)
        df = yf.download(yf_code, start = start_date_, end = end_date_, progress = False)
        df['date'] =  df.index.strftime('%Y-%m-%d')
        df.index = df.index.strftime('%Y-%m-%d')
        
    if hot_load:
        # reload previous data
        df_old = pd.read_csv(start_file)
        df = pd.concat([df_old, df], axis=0)
        

    
    if os.path.exists(start_file):# exists, add to the end
        os.rename(start_file, end_file)
        df.to_csv(end_file, mode='a', header=False, index=True)
    else:
        df.to_csv(os.path.join(output_dir, end_file), index=True)
    
    print("Data fetcher downloaded/updated, File saved to: ", end_file)

if __name__ == '__main__':
    # fetch_and_update_yf(yf_code = 'GC=F', mode='com_disagg', cftc_market_code = '001602', start_date = '2019-01-01', end_date = '2023-05-12')
    # fetch_and_update_yf(yf_code = 'GC=F', mode='fut_disagg', cftc_market_code = '001602', start_date = '2019-01-01', end_date = '2023-05-12')
    fetch_and_update_yf(yf_code = '^GSPC', mode='fut_fin', start_date = '2019-01-01', end_date = '2023-07-10')
    # fetch_and_update_yf(yf_code = '^GSPC', mode='com_fin', start_date = '2019-01-01', end_date = '2023-05-25')