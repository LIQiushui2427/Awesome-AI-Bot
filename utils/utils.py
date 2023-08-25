from statsmodels.tsa.seasonal import STL
import pandas as pd
import random
import os
import functools
import glob
import platform
def get_seperators():
    return '/' if platform.system() == 'Linux' else '\\'

if __name__ == '__main__':
    print(get_seperators())
def saveplots(cerebro, numfigs=1, iplot=True, start=None, end=None,
             width=16, height=9, dpi=300, tight=True, use=None, file_path = '', **kwargs):

        from backtrader import plot
        if cerebro.p.oldsync:
            plotter = plot.Plot_OldSync(**kwargs)
        else:
            plotter = plot.Plot(**kwargs)

        figs = []
        for stratlist in cerebro.runstrats:
            for si, strat in enumerate(stratlist):
                rfig = plotter.plot(strat, figid=si * 100,
                                    numfigs=numfigs, iplot=iplot,
                                    start=start, end=end, use=use)
                figs.append(rfig)

        for fig in figs:
            for f in fig:
                f.savefig(file_path, bbox_inches='tight', dpi=dpi)
        return figs

def find_file_date(partial_name: str, directory: str):
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
def find_file(partial_name: str, directory: str):
    """Find file with partial name in directory.

    Args:
        partial_name (str): partial name of the file.
        directory (str): directory of the file.

    Returns:
        str: file path.
    """
    try:
        return glob.glob(os.path.join(directory, f'*{partial_name}*'))
    except IndexError:
        return None
    
def add_STL(df: pd.DataFrame, period: int ,seasonal: int, robust = False):
    """Add STL decomposition to input. It requires 'Close' column is in the input dataframe.

    Args:
        df (pd.DataFrame): input dataframe. It must have 'Close'
        
    """
    #，period=52，robustTruetrendseason，resid
    
    stl=STL(df['Close'],period=period,seasonal=seasonal,robust=robust)
    
    res = stl.fit(inner_iter=None, outer_iter=None)
    
    df['trend']=res.trend#
    df['seasonal']=res.seasonal
    df['resid']=res.resid
    
    return df


import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from sklearn.metrics import accuracy_score

def get_available_tickers_and_paths(date: str):
    """Get available tickers for given dates.
    Search through the outputsByBt folder, and return the available tickers for given date.
    """
    sep = get_seperators()
    AI_path = os.path.join(os.getcwd(), 'outputsByAI')
    Bt_path = os.path.join(os.getcwd(), 'outputsByBt')
    Log_path = os.path.join(os.getcwd(), 'outputsFromTraining')
    # print("AI_path: ", AI_path)
    
    Bt_paths = find_file(partial_name = date, directory = Bt_path)
    Log_paths = find_file(partial_name = date, directory = Log_path)
    AI_paths = find_file(partial_name = date, directory = AI_path)
    
    mytickertSet = set()
    
    for path in AI_paths:
        # windows
        temp = path.split(sep)[-1].split('_')[0]
        # print("Path: ", path, " Ticker: ", temp)
        mytickertSet.add(temp)
        
    return Log_paths, Bt_paths, AI_paths, list(mytickertSet)

if __name__ == '__main__':
    # print(get_available_tickers_and_paths('2023-08-01'))
    pass

def get_grouped_files(paths, tickers) -> list:
    """Get available files for given tickers in one day.
    """
    mySet = dict()
    for ticker in tickers:
        mySet[ticker] = []
        for file in paths:
            if ticker in file:
                mySet[ticker].append(file)
    return mySet


def evaluate(model, dataname, device,test_X, test_Y, plot = False, logger = None, ):
    """Evaluate the model by ploting the prediction results
    it will loop i in range(1, pr) to get the next i days is higher/lower than the first predicted day (output[0]).
    for real trend, it will start from the same day but with real value.
    Args:
        model (nn.module): A trained ai model that can take l lookback, pr prediction length.
        batch_X (ndarray): created by create_dataset function.
        batch_Y (ndarry): created by create_dataset function.
    """
    # print("Evaluating the model...")
    folderpath = os.path.join(os.getcwd(), 'outputsFromTraining')
    classname = str(model.__class__).split('.')[2].split("'")[0]
    res = []
    # print(classname)
    model.eval()
    real_trends = []
    pr = len(test_Y[0])
    
    # Convert numpy arrays to PyTorch tensors
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
    test_Y_tensor = torch.tensor(test_Y, dtype=torch.float32)

    with torch.no_grad():
        pred = []
        real_price = []
        for i in range(test_X_tensor.shape[0]):
            inputs = test_X_tensor[i].unsqueeze(0)  # Add batch dimension
            inputs = inputs.to(device)  # Don't forget to move your tensor to device
            
            outputs = model(inputs)
            
            
            outputs = outputs.squeeze().cpu().numpy()  # Remove batch dimension and move to cpu
            # print("output", outputs)
            # Calculate trends

            
            
            pred.append(outputs)
            real_price.append(test_Y[i][0])
            
    pred = np.stack(pred, axis=0)
    
    # print("pred: ", pred.shape)
    # print("real_price", real_price) # (predictable days)

    for l in range(1, pr):
        
       
        predicted_trends = np.sign(np.array(pred[l:,0]) - np.array(pred[:-l, l]))

        real_trends = np.sign(np.array(real_price[l:]) - np.array(real_price[:-l]))

        # Calculate accuracy of the trend prediction
        accuracy = accuracy_score(real_trends, predicted_trends)
        title = f"Accuracy for prediction length {l} is {accuracy * 100:.2f}%"
        res.append(accuracy)
        print(title)
        
        if logger is not None:
            logger.info(title)
            
        days = list(range(len(predicted_trends)))

        plt.ioff()
        
        plt.figure(figsize=(12, 9))

        # Plot prediction trends
        for i, trend in enumerate(predicted_trends):
            color = 'green' if trend == 1 else 'red' if trend == -1 else 'gray'
            plt.plot(days[i], pred[i,0], marker='o', markersize=5, color=color)

        # Plot real trends
        for i, trend in enumerate(real_trends):
            color = 'lightgreen' if trend == 1 else 'pink' if trend == -1 else 'lightgray'
            plt.plot(days[i], test_Y[i][0], marker='o', markersize=5, color=color)

        # Set title
        plt.title(f"{classname}: {title}")
        # Create unique legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # Plot prediction line
        plt.plot(days, pred[:-l,0], label="Prediction line")

        # Plot truth line
        truth = [test_Y[i][0] for i in range(len(test_X))]
        plt.plot(days, truth[:-l], label="Truth line")

        plt.legend()
        plt.grid(True)
        
        if l % 2 == 1:
            plt.savefig(os.path.join(folderpath, f"{dataname}_{classname}_pred_{l}.png"), dpi = 280)
            
        plt.close()
        
    # if plot:
        #     #plt.show()
        #     pass
    return res
import asyncio
def rerun_AI_until_criterion_met(mean_accur = 0.55, min_accur = 0.3, max_accur = 0.6,max_attempt = 10):
    def decorator_runAI(f):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempt:
                print(f"Attempt train {attempts}/{max_attempt}")
                final_data_path, res = f(*args, **kwargs)
                if sum(res) / len(res) > mean_accur and min(res) > min_accur and max(res) > max_accur:
                    return final_data_path, res
                else:
                    attempts += 1
            
            return final_data_path, res
        return wrapper
    return decorator_runAI
def retry_with_backoff(retries=5, backoff_in_ms=300):
    def wrapper(f):
        @functools.wraps(f)
        async def wrapped(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await f(*args, **kwargs)
                except Exception as e:
                    print('Fetch error:', e)

                    if x == retries:
                        raise
                    else:
                        sleep_ms = (backoff_in_ms * 2 ** x +
                                    random.uniform(0.5, 1))
                        await asyncio.sleep(sleep_ms / 1000)
                        x += 1
                        print(f'Retrying {x + 1}/{retries}')

        return wrapped

    return wrapper