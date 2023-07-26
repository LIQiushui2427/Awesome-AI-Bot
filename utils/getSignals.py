import pandas as pd

def get_signals(file_path):
    """Get signal for the day, inside the file_path folder.
    Args:
        file_path (str): path of AI output file.
    """
    df = pd.read_csv(file_path)
   
    # get the last 7 lines in last 7 days
    
    df_ = df.iloc[6:, -7:]
    
    weights_in_day = [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]
    weights_between_days = [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]
    #calculate weighted average of the last 7 days, and assign signal to df['signal']
    for i in range(df_.shape[0]):
        df.loc[i + 7, 'weighted'] = sum(df_.iloc[i, :] * weights_in_day)
    
    #calculate weighted average of the last 7 days, and assign signal to df['signal']
    for i in range(df_.shape[0] - 1):
        df.loc[i + 7, 'signal'] = sum(df.iloc[i : i + 7, -2])
    
    df.dropna(subset=['date'], inplace=True)
    
    df.fillna(0, inplace=True)
    # print(df.tail(10))
    # print(df.head(10))
    # print(df.shape)
    
    df.to_csv(file_path, index=False)
    
    # return the last signal
    today_signal = df.iloc[-1, -1]
    yesterday_close = df.iloc[-1, 4]
    return today_signal, yesterday_close

if __name__ == '__main__':
    # get_signals('C:\\Users\\lqs\\Downloads\\CoT_Strategy\\outputsByAI\\GC=F_com_disagg_2023-07-26.csv')
    get_signals('C:\\Users\\lqs\\Downloads\\CoT_Strategy\\outputsByAI\\^GSPC_fut_fin_2023-07-26.csv')