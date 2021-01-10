import yahoo_fin
import collections
from yahoo_fin import stock_info as si
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from paramaters import *
import datetime
from sklearn.linear_model import LinearRegression


def load_data(ticker, Plotting, n_steps=50, scale=True, shuffle=True, lookup_step=1,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'], ):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step)) - 100
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = \
        train_test_split(X, y,test_size=test_size, shuffle=shuffle)
    # return the result
    # load the data
    if Plotting == True:
        print("true")
        return result
    if Plotting == False:
        print("False")
        return X

def load_inflation(inflation, Plotting, n_steps=50, scale=True, shuffle=True, lookup_step=1,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'], ):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(inflation, str):
        # load it from yahoo_fin library
        df = si.get_data(inflation)
    elif isinstance(inflation, pd.DataFrame):
        # already loaded, use it directly
        df = inflation
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    XI, yI = [], []
    for seq, target in sequence_data:
        XI.append(seq)
        yI.append(target)
    # convert to numpy arrays
    XI = np.array(XI)
    yI = np.array(yI   )
    XI = XI.reshape((XI.shape[0], XI.shape[2], XI.shape[1]))
    # split the dataset
    result["XI_train"], result["XI_test"], result["yI_train"], result["yI_test"] = \
        train_test_split(XI, yI,test_size=test_size, shuffle=shuffle)
    # return the result
    # load the data
    if Plotting == True:
        print("true")
        return result
    if Plotting == False:
        print("False")
        return XI


def plot_graph(model, data,inflation):
    y_test = data["y_test"]
    X_test = data["X_test"]
    #yI_test = inflation["yI_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    #yI_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(yI_test, axis=0)))
    # last 200 days, feel free to edit that
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred, c='g')
    #plt.plot(yI_test[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


inflationdata = load_inflation(inflationtickerGBP,True, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,feature_columns=FEATURE_COLUMNS, shuffle=False)


def TrainingData(data,samplesize):
    print(len(data))
    X = []
    Y = []

    for i in range(len(data) - samplesize -1):
        Y.append(data[i + samplesize])
        X.append(data[i:(i + samplesize), 0])
    return(np.array(X),np.array(Y))


# load the data
data = load_data(ticker,False, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,feature_columns=FEATURE_COLUMNS, shuffle=False)

X_train,Y_train = TrainingData(data,N_STEPS)
print(len(X_train))
X_train = np.reshape(X_train, ((X_train.shape[0], X_train.shape[2], X_train.shape[1])))


data = load_data(ticker,True, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,feature_columns=FEATURE_COLUMNS, shuffle=False)
def predictfuture():
    model = Sequential()
    loss = "mean_absolute_error"
    optimizer = "rmsprop"
    model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], N_STEPS)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss=loss,optimizer=optimizer)
    model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=3,
                        validation_data=(data["X_test"], data["y_test"]),
                        verbose=1)
    model = model.predict
    return model
"""
more = []
for i in range(len(data)):
    future = data.get(-1)
    one_day = 86400
    next_unix = i + one_day
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    data[next_date] = predictfuture()

    more.append(data.get(next_date))
    print(more)

np.array(more)
"""
def TrainedModel():
    loss = "mean_absolute_error"
    optimizer = "rmsprop"
    model = Sequential()
    model.add(LSTM(units=50, activation= 'relu', input_shape = (X_train.shape[1],N_STEPS)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss=loss,optimizer=optimizer)
    model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        verbose=1)

    return model



fig = plt.figure()
ax2 = fig.gca()
def on_plot_hover(event):
    # Iterating over each data member plotted
    for curve in ax2.get_lines():
        # Searching which data member corresponds to current mouse position
        if curve.contains(event)[0]:
            x,y = curve.get_data()
            print("over %s", x,y)


#fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)


plot_graph(TrainedModel(), data, inflationdata)

