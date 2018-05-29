import numpy as np
import pandas as pd 
from math import * 
import sklearn.linear_model as skl_lm
from sklearn.model_selection import cross_val_score

from PRML.linear import (
    LogisticRegressor
)


def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)
    return X_train, Y_train,X_test
    
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])    
    
def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid 
    
def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    #print("before ",mu.shape)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    #print("after",mu.shape)
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma
    

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test    

#X_train = pd.read_csv('Data/IncomePrediction/X_train.csv', sep=',', header=0)

X_all, Y_all, X_test = load_data('Data/IncomePrediction/X_train.csv', 'Data/IncomePrediction/Y_train.csv', 'Data/IncomePrediction/X_test.csv')

clf = skl_lm.LogisticRegression(solver='newton-cg')

scores = cross_val_score(clf, X_all, Y_all,
                                scoring="accuracy", cv=5)
print(scores)
#clf.fit(X_all,Y_all)
'''
print(clf)
print('classes: ',clf.classes_)
print('coefficients: ',clf.coef_)
print('intercept :', clf.intercept_)
'''

X_all, X_test = normalize(X_all,X_test)

Y_all = Y_all.flatten()
X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, 0.3)

#print(Y_all.shape)


lg = LogisticRegressor()
lg.fit(X_train, Y_train)
result = lg.classify(X_valid)
result = (result == Y_valid)

acc = (float(result.sum()) / len(X_all))

print(acc)



