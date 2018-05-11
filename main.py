import numpy as np
import pandas as pd 
from source.linear import (
    LogisticRegressor
)


def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)

X_all, Y_all, X_test = load_data('Data\IncomePrediction\X_train.csv', 'Data\IncomePrediction\Y_train.csv', 'Data\IncomePrediction\X_test.csv')