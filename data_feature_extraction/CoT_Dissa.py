from ta import add_all_ta_features
from ta.utils import dropna
import pandas as pd
from minepy import MINE
import os
# Load datas
def extract_data(datasoursepath, finalextracteddatapath, nCorrTop=50, nMICTop=20):
    """
    Given CoT data, extract tas, and return top n correlated cols for feature extraction.
    """
    if os.path.exists(finalextracteddatapath):
        print("ouput file already exsist, just read this file")
        return pd.read_csv(finalextracteddatapath)
    df = pd.read_csv(datasoursepath)

    # Clean NaN values

    df_ = dropna(df)

    # Add ta features filling NaN values
    df_ = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    df_ # 284 cols


    # drop numeric but meaningless cols
    df.drop(columns=['As_of_Date_In_Form_YYMMDD','CFTC_Contract_Market_Code', 'CFTC_Region_Code', 'CFTC_Commodity_Code'], axis=1, inplace=True)
    print("Number of Cols: ", len(df.columns))
    # Select columns after Volume
    columns_to_drop = df.select_dtypes(include=['int64', 'float64']).columns[df.columns.get_loc('Volume') + 1:]
    
    print("Col to drop:", columns_to_drop)


    # Calculate the correlation matrix for the selected columns
    correlation_matrix = df[columns_to_drop].corr()

    # Identify highly correlated columns
    threshold = 0.9
    correlated_columns = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_columns.add(colname)

    # Drop the highly correlated columns

    df.drop(columns=correlated_columns, inplace=True)


    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns


    correlation_with_close = df.select_dtypes(include=['int64', 'float64']).corr().loc[:, 'Close'].abs()

    columns_after_volume = numeric_columns[numeric_columns.get_loc('Volume') + 1:]

    top_correlated_columns = correlation_with_close[columns_after_volume].nlargest(nCorrTop).index



    cols_to_drop_corr = set(columns_after_volume) - set(top_correlated_columns)

    df.drop(columns=cols_to_drop_corr, inplace=True)
    
    print(f"Dropped cols not in top {nCorrTop} corr:", cols_to_drop_corr)
    
    
    print("Starting to drop cols with MIC..")
    
    # reset col indexes
    
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    columns_after_volume = numeric_columns[numeric_columns.get_loc('Volume') + 1:]
    
    mic_values = []
    
    for column in columns_after_volume:
        mine = MINE()
        mine.compute_score(df[column], df['Close'])
        mic_values.append((column, mine.mic()))
    
    mic_values.sort(key=lambda x: x[1], reverse=True)
    
    top_mic_columns = [column for column, _ in mic_values[:nMICTop]]
    

    
    cols_to_drop = set(top_correlated_columns) - set(top_mic_columns)
    
    print(f"Cols with Top{nMICTop} MIC:", mic_values)
    
    df.drop(columns=cols_to_drop, inplace= True)
    
    df.to_csv(finalextracteddatapath, index=False)

    print("Have dropped cols, final data saved to:", finalextracteddatapath)
    
    return df


    
    


