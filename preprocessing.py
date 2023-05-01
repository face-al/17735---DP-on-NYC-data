import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocessing(filename):
    # read csv file into matrix
    df = pd.read_csv(filename, sep=',', header=None)[1:]
    df = np.array(df)
    data = df[:, 10:14]
    data = np.array(list(data), dtype=np.float64)
    target = df[:, 8]
    target = np.array(target,dtype=np.float64)

    # delete broken data
    idx = np.where(~data.any(axis=1))[0]
    data = np.delete(data, idx, axis=0) * 100
    target = np.delete(target, idx, axis=0)
    idx = np.where(~target.any(axis=0))[0]
    data = np.delete(data, idx, axis=0)
    target = np.delete(target, idx, axis=0)

    # normalization
    for i in range(4):
        data[:, i] -= data[:, i].mean()

    # split into test and train set
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    return X_train, X_test, y_train, y_test
