import os
import requests
import zipfile
import pandas as pd
import datetime as dt
import yfinance as  yf

def fetcher_for_fut_disgg(output_dir, output_file, yf_code: str, cftc_market_code:str, start_date:dt.datetime, end_date = dt.datetime.today()):
    """Data fetcher for the CFTC Commitments of Traders Report (COTR) Disaggregated Futures Only Reports.
    
    Whenever called, It will instantly download and save the result in a fixed directory. Recall it will erase previous data.
    @output_dir: Typically os.getcwd() + 'data'
    @output_file: File name: i.e. 'fut_disagg.txt'
    @start_date: The start year of the data you want to fetch. Format: YYYYMMDD
    @end_date: The end year of the data you want to fetch. Format: YYYYMMDD
    @yf_code: The Yahoo Finance code. For example, 'ZW=F' for Wheat Futures, Chicago SRW Wheat Futures, CBOT
    @cftc_market_code: The CFTC market code. For example, '001602' for WHEAT-SRW - CHICAGO BOARD OF TRADE
    """
    base_url = 'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{}.zip'
    file_path = os.path.join(output_dir, output_file)
    
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if(isinstance(end_date, str)):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
    assert start_date >= dt.datetime(2010,1,1), 'Do not support data fetch earlier than 20100101.'
    assert end_date <= dt.datetime.today(), 'The end year should be no later than today.'
    assert start_date <= end_date, 'The start year should be no later than the end year.'
    
    if os.path.exists(file_path):
        print("Removing previous file: ", file_path)
        os.remove(file_path)
    
    # Loop through all years from 2010 to 2023
    for year in range(end_date.year, start_date.year - 1, -1):
        print(f"Downloading Disaggregated Futures Only Reports from CFTC for {year}...")

        # Construct the URL and download path for this year
        url = base_url.format(year)
        output_zip = os.path.join(output_dir, f'fut_disagg_txt_{year}.zip')

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
        new_data = pd.read_csv(os.path.join(output_dir, f'f_year.txt'))

        # Append the data to the output file
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False,index=False)
        else:
            # print("Df: ", new_data.head())
            new_data.to_csv(file_path, mode='w', header=True,index=False)
    
    print("Downloaded, File saved to: ", file_path)
        
    yf_df = yf.download(yf_code, start = start_date, end = end_date, progress = False)

    yf_df['date'] =  yf_df.index.strftime('%Y-%m-%d')
    yf_df.index = yf_df.index.strftime('%Y-%m-%d')
    
    cftc_df = pd.read_csv(file_path)

    cftc_df = cftc_df[(cftc_df["CFTC_Contract_Market_Code"] == cftc_market_code) & (cftc_df["Report_Date_as_YYYY-MM-DD"] >= start_date.strftime('%Y-%m-%d'))\
        & (cftc_df["Report_Date_as_YYYY-MM-DD"] <= end_date.strftime('%Y-%m-%d'))]
    
    
    df = pd.merge(yf_df, cftc_df, left_index=True, right_on="Report_Date_as_YYYY-MM-DD", how='outer')
    df.interpolate(inplace=True, limit_direction='forward')
    df.set_index("date", inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    df.to_csv(os.path.join(output_dir, f'{yf_code}_fut_disagg.csv'), index=True)
    
    return df
        
        

# https://www.cftc.gov/files/dea/history/com_disagg_txt_2023.zip
def fetchers_for_com_disagg(output_dir, output_file, yf_code: str, cftc_market_code:str, start_date:dt.datetime, end_date = dt.datetime.today()):
    """Data fetcher for the CFTC Commitments of Traders Report (COTR) Disaggregated Futures-and-options Combined Reports.
    
    Whenever called, It will instantly download and save the result in a fixed directory. Recall it will erase previous data.
    @output_dir: Typically os.getcwd() + 'data'
    @output_file: File name: i.e. 'fut_disagg.txt'
    @start_date: The start year of the data you want to fetch. Format: YYYYMMDD
    @end_date: The end year of the data you want to fetch. Format: YYYYMMDD
    @yf_code: The Yahoo Finance code. For example, 'ZW=F' for Wheat Futures, Chicago SRW Wheat Futures, CBOT
    @cftc_market_code: The CFTC market code. For example, '001602' for WHEAT-SRW - CHICAGO BOARD OF TRADE
    """
    base_url = 'https://www.cftc.gov/files/dea/history/com_disagg_txt_{}.zip'
    file_path = os.path.join(output_dir, output_file)
    
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if(isinstance(end_date, str)):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
    assert start_date >= dt.datetime(2010,1,1), 'Do not support data fetch earlier than 20100101.'
    assert end_date <= dt.datetime.today(), 'The end year should be no later than today.'
    assert start_date <= end_date, 'The start year should be no later than the end year.'
    
    if os.path.exists(file_path):
        print("Removing previous file: ", file_path)
        os.remove(file_path)

    # Loop through all years from 2010 to 2023
    for year in range(end_date.year, start_date.year - 1, -1):
        print(f"Downloading  Futures-and-options Combined Reports data from CFTC for year {year}...")

        # Construct the URL and download path for this year
        url = base_url.format(year)
        output_zip = os.path.join(output_dir, f'com_disagg_txt_{year}.zip')

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
        new_data = pd.read_csv(os.path.join(output_dir, f'c_year.txt'))

        # Append the data to the output file
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False,index=False)
        else:
            # print("Df: ", new_data.head())
            new_data.to_csv(file_path, mode='w', header=True,index=False)
    
    print("Downloaded, File saved to: ", file_path)
        
    yf_df = yf.download(yf_code, start = start_date, end = end_date, progress = False)

    
    yf_df['date'] =  yf_df.index.strftime('%Y-%m-%d')
    yf_df.index = yf_df.index.strftime('%Y-%m-%d')
    # print("yf_df cols:", yf_df)

    
    cftc_df = pd.read_csv(file_path)

    cftc_df = cftc_df[(cftc_df["CFTC_Contract_Market_Code"] == cftc_market_code) & (cftc_df["Report_Date_as_YYYY-MM-DD"] >= start_date.strftime('%Y-%m-%d'))\
        & (cftc_df["Report_Date_as_YYYY-MM-DD"] <= end_date.strftime('%Y-%m-%d'))]
    
    
    
    df = pd.merge(yf_df, cftc_df, left_index=True, right_on="Report_Date_as_YYYY-MM-DD", how='outer')
    df.interpolate(inplace=True, limit_direction='forward')
    df.set_index("date", inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    df.to_csv(os.path.join(output_dir, f'{yf_code}_com_disagg.csv'), index=True)
    
    return df



# https://www.cftc.gov/files/dea/history/fut_fin_txt_2023.zip
def fetchers_for_Traders_Finance_Futures(output_dir, output_file, yf_code: str, cftc_market_code:str, start_date:dt.datetime, end_date = dt.datetime.today()):
    """Data fetcher for the CFTC Commitments of Traders Report (COTR) Disaggregated Futures-and-options Combined Reports.
    
    Whenever called, It will instantly download and save the result in a fixed directory. Recall it will erase previous data.
    @output_dir: Typically os.path.join(os.getcwd(), 'data')
    @output_file: File name: i.e. 'TFF_Futures.txt'
    @start_date: The start year of the data you want to fetch. Format: YYYYMMDD
    @end_date: The end year of the data you want to fetch. Format: YYYYMMDD
    @yf_code: The Yahoo Finance code. For example, '^GSPC' for SNP - SNP Real Time Price (Currency in USD)
    @cftc_market_code: The CFTC market code. For example, '13874+' for S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE
    """
    
    base_url = 'https://www.cftc.gov/files/dea/history/fut_fin_txt_{}.zip'
    file_path = os.path.join(output_dir, output_file)


        
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if(isinstance(end_date, str)):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
    assert start_date >= dt.datetime(2010,1,1), 'Do not support data fetch earlier than 20100101.'
    assert end_date <= dt.datetime.today(), 'The end year should be no later than today.'
    assert start_date <= end_date, 'The start year should be no later than the end year.'
    
    if os.path.exists(file_path):
        print("Removing previous file: ", file_path)
        os.remove(file_path)
        
    for year in range(end_date.year, start_date.year - 1, -1):
        print(f"Downloading Traders in Financial Futures (Futures Only) Reports from CFTC for {year}...")

        # Construct the URL and download path for this year
        url = base_url.format(year)
        output_zip = os.path.join(output_dir, f'TFF_Fut_{year}.zip')

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
        new_data = pd.read_csv(os.path.join(output_dir, f'FinFutYY.txt'))

        # Append the data to the output file
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False,index=False)
        else:
            # print("Df: ", new_data.head())
            new_data.to_csv(file_path, mode='w', header=True,index=False)
    
    print("Downloaded, File saved to: ", file_path)
        
    yf_df = yf.download(yf_code, start = start_date, end = end_date, progress = False)

    yf_df['date'] =  yf_df.index.strftime('%Y-%m-%d')
    yf_df.index = yf_df.index.strftime('%Y-%m-%d')
    
    cftc_df = pd.read_csv(file_path)

    cftc_df = cftc_df[(cftc_df["CFTC_Contract_Market_Code"] == cftc_market_code) & (cftc_df["Report_Date_as_YYYY-MM-DD"] >= start_date.strftime('%Y-%m-%d'))\
        & (cftc_df["Report_Date_as_YYYY-MM-DD"] <= end_date.strftime('%Y-%m-%d'))]
    
    
    df = pd.merge(yf_df, cftc_df, left_index=True, right_on="Report_Date_as_YYYY-MM-DD", how='outer')
    df.interpolate(inplace=True, limit_direction='forward')
    df.set_index("date", inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    
    df.to_csv(os.path.join(output_dir, f'{yf_code}_fin_fut.csv'), index=True)
    
    return df



def fetchers_for_Traders_Finance_Combined(output_dir, output_file, yf_code: str, cftc_market_code:str, start_date:dt.datetime, end_date = dt.datetime.today()):
    """Data fetcher for the CFTC Commitments of Traders Report (COTR) Disaggregated Futures-and-options Combined Reports.
    
    Whenever called, It will instantly download and save the result in a fixed directory. Recall it will erase previous data.
    @output_dir: Typically os.path.join(os.getcwd(), 'data')
    @output_file: File name: i.e. 'TFF_Futures.txt'
    @start_date: The start year of the data you want to fetch. Format: YYYYMMDD
    @end_date: The end year of the data you want to fetch. Format: YYYYMMDD
    @yf_code: The Yahoo Finance code. For example, '^GSPC' for SNP - SNP Real Time Price (Currency in USD)
    @cftc_market_code: The CFTC market code. For example, '13874+' for S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE
    """
    
    base_url = 'https://www.cftc.gov/files/dea/history/com_fin_txt_{}.zip'
    file_path = os.path.join(output_dir, output_file)


        
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if(isinstance(end_date, str)):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        
    assert start_date >= dt.datetime(2010,1,1), 'Do not support data fetch earlier than 20100101.'
    assert end_date <= dt.datetime.today(), 'The end year should be no later than today.'
    assert start_date <= end_date, 'The start year should be no later than the end year.'
    
    if os.path.exists(file_path):
        print("Removing previous file: ", file_path)
        os.remove(file_path)
        
    for year in range(end_date.year, start_date.year - 1, -1):
        print(f"Downloading Traders in Financial Futures (Futures-and-options-combined) Reports from CFTC for {year}...")

        # Construct the URL and download path for this year
        url = base_url.format(year)
        output_zip = os.path.join(output_dir, f'TFF_Com_{year}.zip')

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
        new_data = pd.read_csv(os.path.join(output_dir, f'FinComYY.txt'))

        # Append the data to the output file
        if os.path.exists(file_path):
            new_data.to_csv(file_path, mode='a', header=False,index=False)
        else:
            # print("Df: ", new_data.head())
            new_data.to_csv(file_path, mode='w', header=True,index=False)
    
    print("Downloaded, File saved to: ", file_path)
        
    yf_df = yf.download(yf_code, start = start_date, end = end_date, progress = False)

    yf_df['date'] =  yf_df.index.strftime('%Y-%m-%d')
    yf_df.index = yf_df.index.strftime('%Y-%m-%d')
    
    cftc_df = pd.read_csv(file_path)

    cftc_df = cftc_df[(cftc_df["CFTC_Contract_Market_Code"] == cftc_market_code) & (cftc_df["Report_Date_as_YYYY-MM-DD"] >= start_date.strftime('%Y-%m-%d'))\
        & (cftc_df["Report_Date_as_YYYY-MM-DD"] <= end_date.strftime('%Y-%m-%d'))]
    
    df = pd.merge(yf_df, cftc_df, left_index=True, right_on="Report_Date_as_YYYY-MM-DD", how='outer')
    df.interpolate(inplace=True, limit_direction='forward')
    df.set_index("date", inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    df.to_csv(os.path.join(output_dir, f'{yf_code}_fin_com.csv'), index=False)
    return df


if __name__ == '__main__':
    output_dir = os.path.join(os.getcwd(), 'data')
    output_file = os.path.join(output_dir, 'fut_disagg.txt')
    fetcher_for_fut_disgg(output_dir, output_file, 'ZW=F', '001602', '2019-01-01', '2020-01-31')