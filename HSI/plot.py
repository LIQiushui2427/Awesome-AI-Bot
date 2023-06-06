import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import mplfinance as mpf
if __name__ == '__main__':
    # Read data
    data = pd.read_excel('HSI.xlsx', index_col='Date', usecols=['Date', 'Open', 'High', 'Low', 'Close', '睇升', '睇跌'])
    data1 = data[['Open', 'High', 'Low', 'Close']]
    data2 = data[['睇升']]
    data2 = data2.apply(lambda x: (x - 0.5) * 2)
    print(data2.head())
    print(data.head())
    style = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(up="r", down="#0000CC",inherit=True),
                           gridcolor="gray", gridstyle="--", gridaxis="both")     
    
    apd = mpf.make_addplot(data2, type='line', width=0.7, panel=1, color='g', alpha=0.5, ylabel='(bullish - 0.5) * 2')
    
    fig, axes= mpf.plot(data1, type='candle', mav=(5, 10, 20), volume=False, show_nontrading=False,  title='HSI', returnfig=True, style=style, addplot=apd)
    
    axes[0].legend(['Close', 'Open', 'High', 'Low'], loc='upper right')
    
    
    mpf.show()
    
    
    