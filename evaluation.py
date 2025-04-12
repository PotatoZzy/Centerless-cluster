import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score


def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cost_m = np.max(cm) - cm
    indices = linear_sum_assignment(cost_m)
    indices = np.asarray(indices)
    indexes = np.transpose(indices)
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value
    return total * 1. / np.sum(cm)

def multi_accuracy(y_true, Y_predict):
    ret = np.array([accuracy(y_true=y_true, y_pred=y_pred) for y_pred in Y_predict])
    return ret

def multi_nmi(y_true, Y_predict):
    ret = np.array([normalized_mutual_info_score(labels_true=y_true, labels_pred=y_pred,  average_method="max") for y_pred in Y_predict])
    return ret

def multi_ari(y_true, Y_predict):
    ret = np.array([adjusted_rand_score(labels_true=y_true, labels_pred=y_pred) for y_pred in Y_predict])
    return ret