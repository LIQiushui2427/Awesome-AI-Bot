import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming that df is your DataFrame and it has been loaded properly

def select_feature(df: pd.DataFrame, test_size=0.2,m = 48):
    """Select feature from dataframe with extracted columns.
    It will calculated and preserve columns which are most correlated  with Close.
    
    Args:
        @df (pd.DataFrame): dataframe to be seleced. It should at least have date, OHLCVA.
        @m (int): top m columns you want to preserve.
    """
    
    # Convert 'Date' to datetime if it's not
    df['date'] = pd.to_datetime(df['date'])

    # Set 'Date' as the index
    df.set_index('date', inplace=True)

    # Drop the non-numerical columns
    df = df.select_dtypes(include=[np.number])

    # Define your target variable
    
    target = 'Close'  # Replace with your actual target column
    preserve_cols = df.columns.tolist()[:df.columns.tolist().index('Volume')+1]
    
    # Split the data into input features (X) and target variable (y)
    y = df[target]
    X = df.drop(preserve_cols, axis=1)
    

    # Split your data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize the Random Forest model
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Get the feature importances
    feature_importances = rf.feature_importances_

    # Sort the features by their importances, in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]

    # Print each feature and its importance
    for index in sorted_indices:
        print(f"{X.columns[index]}: {feature_importances[index]}")
        
  

    # Get top m features
    top_m_features = [X.columns[i] for i in sorted_indices[:m]]

    # Add top m features to preserve_cols
    selectedcol = preserve_cols + top_m_features

    # Create a new DataFrame with the desired columns
    df_selected = df[selectedcol]

    print("Ramdom forest: Preseved columns:", df_selected.columns)
    
    return df_selected
    