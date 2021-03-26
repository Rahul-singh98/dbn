import numpy as np

np.random.seed(1337)  # for reproducibility
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics._regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn import SupervisedDBNRegression
from dbn import SupervisedDBNClassification
import time
import matplotlib.pyplot as plt 

# Loading dataset
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

# Preparing training data
scaled_train = np.array(train)
X_train , y_train = [] , []
for i in range(60 , len(scaled_train)):
    X_train.append(scaled_train[i-60 : i ])
    y_train.append(scaled_train[i ])

# Training
X_train = np.array(X_train)
y_train = np.array(y_train)

regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.001,
                                    learning_rate=0.001,
                                    n_epochs_rbm=60,
                                    n_iter_backprop=100,
                                    batch_size=32,
                                    activation_function='relu')

regressor.fit(X_train, y_train)


print('*'*15 , ' Model is saving ' , '*'*15)
regressor.save('./models/nifty_24_03_2021.pkl')
print('*'*15 , ' Model saved ' , '*'*15)
