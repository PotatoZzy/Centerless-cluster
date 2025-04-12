import os
import time
import scipy.io as sio
from scipy import sparse
from scipy.spatial.distance import cdist
import numpy as np


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
    data_full_name = os.path.join(current_path, f"data/{data_name}.mat")
    X, y_true, N, dim, c_true = load_mat(data_full_name)
    X = X.astype(np.float64)

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