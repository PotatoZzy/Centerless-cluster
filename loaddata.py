import os
import time
import scipy.io as sio
from scipy import sparse
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

from mpeg7_feature import process_mpeg7


def load_isolet(train_data_path, test_data_path):
    # read train data
    df = pd.read_csv(train_data_path, header=None, sep=',',
                     na_values='?', keep_default_na=False)
    X_train = df.iloc[:, :-1].values.astype(float)
    y_train = df.iloc[:, -1].values.astype(int)

    # read test data
    df = pd.read_csv(test_data_path, header=None, sep=',',
                     na_values='?', keep_default_na=False)
    X_test = df.iloc[:, :-1].values.astype(float)
    y_test = df.iloc[:, -1].values.astype(int)

    X = np.vstack([X_train, X_test])  # (7797, 617)
    y = np.concatenate([y_train, y_test])  # (7797,)
    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y))
    return X, y, N, dim, c_true


def load_mpeg7(path):
    X, y, classes = process_mpeg7(path)
    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y))
    return X, y, N, dim, c_true

def load_mat(path, to_dense=True):
    data = sio.loadmat(path)
    X = data["X"]
    if "y_true" in data.keys():
        y_true = data["y_true"].astype(np.int32).reshape(-1)
    elif "Y" in data.keys():
        y_true = data["Y"].astype(np.int32).reshape(-1)
    else:
        assert 1==0

    if sparse.isspmatrix(X) and to_dense:
        X = X.toarray()

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    return X, y_true, N, dim, c_true

def load_data(data_name):
    current_path = os.path.dirname(__file__)

    if data_name == 'isolet':
        train_data_full_name = os.path.join(current_path, f"data/{data_name}1234.data")
        test_data_full_name = os.path.join(current_path, f"data/{data_name}5.data")
        X, y_true, N, dim, c_true = load_isolet(train_data_full_name, test_data_full_name)
    elif data_name == 'mpeg7':
        data_full_name = os.path.join(current_path, f'data/{data_name}')
        X, y_true, N, dim, c_true = load_mpeg7(data_full_name)
    else:
        data_full_name = os.path.join(current_path, f"data/{data_name}.mat")
        X, y_true, N, dim, c_true = load_mat(data_full_name)

    X = X.astype(np.float64)
    print(f"X size {X.shape}")
    knn = int(N/c_true * 1.2)
    t_start = time.time()
    NN, NND = getNNwithDistance(X, knn)
    t_end = time.time()
    t1 = t_end - t_start
    return X, y_true, c_true, NN, NND, t1

def getNNwithDistance(X, knn):
    D_full = cdist(X, X, metric='sqeuclidean')
    np.fill_diagonal(D_full, -1)
    NN_full = np.argsort(D_full, axis=1)
    np.fill_diagonal(D_full, 0)

    NN = NN_full[:, :knn]

    ind_M = NN
    n, k = NN.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    NND = D_full[row, col].reshape((n, k))

    return NN, NND
##########################
## input has been tested
##########################