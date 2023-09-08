import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

"""
Evaluation code adopted from https://github.com/fanolabs/NID_ACLARR2022/blob/main/utils/tools.py
"""


def hungarian_alignment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungarian_alignment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 4),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 4),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 4)}