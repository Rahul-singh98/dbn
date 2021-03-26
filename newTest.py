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

df = pd.read_csv('/home/rahul/Desktop/data/Equity/NSE50.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
data = df['Close'].copy()

# Training and testing data
train_size = int(len(data) * 0.80)
train = data[:train_size]
test = data[train_size :]

# Data scaling
# scaler = MinMaxScaler()
# scaled_train = scaler.fit_transform(np.array(train).reshape(-1,1))
# log_change_train = np.log(train).pct_change().dropna()
# log_change_test = np.log(test).pct_change().dropna()
# print(log_change_train)

# Test
# scaled_test = scaler.transform(np.array(test).reshape(-1,1))
X_test , y_test = [] , []
for i in range(60 , len(scaled_test)):
    X_test.append(scaled_test[i-60: i ,0])
    y_test.append(scaled_test[i ,0])

X_test = np.array(X_test)
y_test = np.array(y_test)
# print(X_test.shape)

try:
    FILE_PATH = '/home/rahul/Desktop/dbn/nifty_20_03_2021.pkl'
    regressor = SupervisedDBNRegression.load(FILE_PATH)
    print('Model loaded Successfully ')

except:
    print('Unable to load model')

try:
    y_pred = regressor.predict(X_test)
    print('\nR-squared: %f\nMSE: %f' % (r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)))
    try:
        y_test = scaler.inverse_transform(y_test.reshape(-1,1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
        # y_test = return_log_normal(y_test)
        # y_pred = return_log_normal(y_pred)
        plt.figure(figsize=(20,9))
        # plt.plot(train.index , train , label = 'Training data')
        plt.plot(test.index[60:] , y_test , label='Test' , color ='b' )
        plt.plot(test.index[60:], y_pred , label='Regressor Predictions'  , color='g')
        plt.title('Actual v/s Predicted')
        plt.legend()
        # if save_output_image==True:
        plt.savefig('./assets/output{}.png'.format(time.ctime(time.time())))
        plt.show()

    except Exception as e:
        print(e)

except Exception as e:
    print(e)