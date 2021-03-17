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
df = pd.read_csv('/home/rahul/Desktop/data/FO_15min/NIFTY1.csv')

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
test = data[train_size :]

# Data scaling
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(np.array(train).reshape(-1,1))

# Preparing training data
X_train , y_train = [] , []
for i in range(60 , len(scaled_train)):
    X_train.append(scaled_train[i-60 : i , 0])
    y_train.append(scaled_train[i ,0])

# Training
X_train = np.array(X_train)
y_train = np.array(y_train)

regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.001,
                                    learning_rate=0.001,
                                    n_epochs_rbm=60,
                                    n_iter_backprop=800,
                                    batch_size=32,
                                    activation_function='relu')

regressor.fit(X_train, y_train)

print('*'*15 , ' Model is saving ' , '*'*15)
regressor.save('./15minregressor.pkl')
print('*'*15 , ' Model saved ' , '*'*15)
