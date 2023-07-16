import sys
import pandas as pd

sys.path.append('..')
from utils.utils import add_STL
from data_feature_extraction.CoT_Dissa import extract_data
from data_feature_selection.cot_dissag import select_feature
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.LSTMWithAttention import StockPredictor3
import os
from utils.utils import plot_predictions
def create_dataset(scaler: MinMaxScaler, df: pd.DataFrame, l, pr):
        """
        For train: Create dataset just for model training. It should be large in whole dataset.
        It will firstly do scaling on all cols (over whole input dataset), so make sure there is no string columns.
        For test: create test dataset. The set have lookback period and future date, 
        for the testing data should not contain future data, you need to  May be need to calculate rolling minmax for each input:
        it should keep moving and calculate all using.
        """
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        X, Y = [], []
        for i in range(len(df)-l-pr+1):
            X.append(df.iloc[i:i+l].values)  # Get the values for l days
            Y.append(df.iloc[i+l:i+l+pr]['Close'].values)  # Get the closing price for the 16th day
        return np.array(X), np.array(Y), scaler
def train_model(model, train_dataloader, criterion, optimizer, num_epochs, device):
        print("model:", model)
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                # print("Output:", outputs)
                
                # print("Shape of outputs", outputs.shape)
                # print("Shape of labels:", labels.shape)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {running_loss/len(train_dataloader)}')

        print('Finished Training')
        return model
def test_model(model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            trend_correct = 0  # Counter for correct trend predictions

            for i, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                # print("Output:", outputs)
                
                loss = criterion(outputs, labels)

                tot_loss += loss.item()



            print(f'Evaluation on test data:  \
                Total loss: {tot_loss} \
                    ')

def trainAI(ticker = "GC=F", mode = "com_disagg",
            end_date = "2021-01-01",
            model = StockPredictor3):
    """Train AI, return and save it. 
    Args:
        datapath (str): path of data source path
        outputpath (str): path of output path
        model (str, optional): AI model to be used. Defaults to StockPredictor3

    Returns:
        trained ai's path in string.
    """
    print("Training AI for", ticker, "in mode", mode, "until", end_date)
    postfix_1 = '_temp'
    postfix_2 = '_processed'
    postfix_3 = '_model'
    
    dataname = ticker + "_" + mode
    folderpath = os.path.join(os.getcwd(), 'data')
    outputpath = os.path.join(os.getcwd(), 'outputs')
    datapath = os.path.join(folderpath, dataname + ".csv")
    final_data_path = os.path.join(outputpath, dataname + postfix_2 + "_" + end_date + ".csv")
    # if os.path.exists(final_data_path):
    #     print("Already processed data at:", final_data_path)
    #     print("Have already processed data for", ticker, "in mode", mode, "until", end_date)
    #     return
    model_save_path = os.path.join(outputpath, dataname + postfix_3 + "_" + end_date + ".pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    df = extract_data(datasoursepath = datapath,
                        finalextracteddatapath = os.path.join(folderpath, dataname + postfix_1 + "_" + end_date + ".csv"),
                        nCorrTop=100, nMICTop= 70)
    df = add_STL(df, period=32, seasonal = 3)


    df.set_index('date', drop=True)
    # print(len(df.columns))
    # print(df.columns)

    test_size = 0.1
    df_selected = select_feature(df, test_size= test_size, m = 12)



    train_size = int((1 - test_size) * len(df_selected))
    # Apply the MinMaxScaler to the df_selected
    train_df = df_selected.iloc[:train_size]
    test_df = df_selected

    # print(train_df.columns)
    # print(test_df.describe())
    # print(test_df.head(20))

    # Initialize a MinMaxScaler
    scaler = MinMaxScaler()


    batch_size = 24
    # Split the data into training and test sets


    # Create the dataset
    l = 64
    pr = 8

    # Create the dataset
    train_X, train_Y, scaler = create_dataset(scaler,train_df, l, pr)
    test_X, test_Y, scaler = create_dataset(scaler,test_df, l, pr)
    test_X, test_Y= test_X[len(train_Y):], test_Y[len(train_Y):]
    
    # Create TensorDatasets
    train_data = TensorDataset(torch.from_numpy(train_X).to(torch.float32), torch.from_numpy(train_Y).to(torch.float32))
    test_data = TensorDataset(torch.from_numpy(test_X).to(torch.float32), torch.from_numpy(test_Y).to(torch.float32))
    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)



    print(model)
    model = model(input_dim = train_df.shape[1], hidden_dim = 128, num_layers = 1, pr = pr, output_dim = 1, dropout_prob = 0.2, num_heads = 2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0004)
    criterion = torch.nn.MSELoss()
    num_epochs = 50

    train_model(model, train_loader, criterion = criterion, optimizer = optimiser, num_epochs = num_epochs, device = device)
    test_model(model, test_dataloader = test_loader, criterion = criterion,device = device)


    
    # plot_predictions(model, device = device,test_X = test_X,test_Y = test_Y)


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


    pred_df = pd.DataFrame()
    for i in range(1, pr):
        # print("Shape of np.array(pred[i:,0]):", np.array(pred[i:,0]).shape)
        # print("Shape of np.array(pred[:-i, i]):", np.array(pred[:-i, i].shape))
        temp_df = pd.DataFrame(np.append(np.array(pred[i:,0]) - np.array(pred[:-i, i]), np.nan))
        pred_df = pd.concat([pred_df, temp_df], ignore_index=True, axis=1)
    pred_df = pred_df.fillna(method = 'ffill', axis = 1).fillna(method = 'ffill')
    # print(pred_df)


    real_df = df[-len(test_Y):]
    pred_df.index = real_df.index
    # pd.merge(real_df, pred_df, left_on = real_df.index, right_on=pred_df.index)
    bt_df = pd.concat([real_df, pred_df], axis = 1, join = 'inner')
    # raname the numeric cols:

    for i in range(1, pr):
        bt_df.rename(columns={bt_df.columns[-i]: "Col_" + str(pr - i) }, inplace = True)  

    bt_df.to_csv(final_data_path)
    
    print("Training finised, saving output file to..", final_data_path)
    
    torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    trainAI()