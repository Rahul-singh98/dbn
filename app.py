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

import os 
import sys

from flask import Flask , request ,jsonify

port = int(os.environ.get('PORT',5000))

# FILE_PATH = './models/15minregressor.pkl'
# regressor = SupervisedDBNRegression.load(FILE_PATH)

app = Flask(__name__)

@app.route('/predict' , methods=['GET'])
def prediction():
    filepath = request.files['csv_file'].read()
    model = request.files['model'].read()

    df = pd.read_csv(filepath)
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
    # test = data[train_size :]

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
    regressor = SupervisedDBNRegression.load(model)

    y_pred = regressor.predict(X_test)
    # print('Metrices of Regressor ')
    # print('\nR-squared: %f\nMSE: %f' % (r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)))
    # print('Accuracy = {}'.format(r2_score(y_test , y_pred) * 100)) 
    # y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))

    # plt.figure(figsize=(20,9))
    # # plt.plot(train.index , train , label = 'Training data')
    # plt.plot(test.index[60:] , y_test , label='Test' , color ='b' )
    # plt.plot(test.index[60:], y_pred , label='Regressor Predictions'  , color='g')
    # plt.title('Actual v/s Predicted')
    # plt.legend()

    # plt.savefig('./assets/output.png')
    # print(y_pred)
    return jsonify({'Prediction':y_pred[0][0]})


if __name__== '__main__':

    app.run(debug=True , 
            host='0.0.0.0' ,
            port = port)
