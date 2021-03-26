import numpy as np

np.random.seed(1337)  # for reproducibility
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics._regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from dbn import SupervisedDBNRegression
from dbn import SupervisedDBNClassification
import time
import matplotlib.pyplot as plt 

def testing(csv_filepath ,  save_output_image , model_path):
    df = pd.read_csv(csv_filepath)
    dt = list()
    for i in range(len(df)):
        dt.append(df['Date'][i]+ " " +df['Time'][i])


    df['DateTime'] = dt
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.index = df['DateTime']

    df.drop(df[df['Volume']==0].index , axis=0 ,inplace=True)
    idx = df[df['Low']==df['High']].index
    df.drop(idx , axis=0 , inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.index = df['Date']
    data = df['Close'].copy()

    # Training and testing data
    train_size = int(len(data) * 0.80)
    train = data[:train_size]

    # Live Testing 
    test_size = 60
    test = data[len(df) - test_size :]

    # Data scaling
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(np.array(train).reshape(-1,1))

    # Test
    scaled_test = scaler.transform(np.array(test).reshape(-1,1))
    X_test = [i for i in test]

    X_test = np.array(X_test)


    try:
        FILE_PATH = model_path
        regressor = SupervisedDBNRegression.load(FILE_PATH)

    except:
        print('Unable to load model')

    y_pred = regressor.predict(X_test)
    print('Predicted value is {}'.format(y_pred[0][0]))