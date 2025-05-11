import numpy as np
import numba
from numba import prange

@staticmethod
@numba.njit(cache=True, fastmath=True, parallel=True)
def gower_similarity_to_prototype_numba(X, prototype, isCategorical, numericFeaturesRanges):
    n_samples, n_features = X.shape

    numeric_indices = np.empty(0, dtype=np.int32)
    categorical_indices = np.empty(0, dtype=np.int32)

    n_numeric = 0
    n_categorical = 0

    for i in range(isCategorical.shape[0]):
        if isCategorical[i]:
            n_categorical += 1
        else:
            n_numeric += 1

    numeric_indices = np.empty(n_numeric, dtype=np.int32)
    categorical_indices = np.empty(n_categorical, dtype=np.int32)

    idx_num = 0
    idx_cat = 0

    for i in range(isCategorical.shape[0]):
        if isCategorical[i]:
            categorical_indices[idx_cat] = i
            idx_cat += 1
        else:
            numeric_indices[idx_num] = i
            idx_num += 1

    similarities = np.empty(n_samples, dtype=np.float64)

    for i in prange(n_samples):
        sum_similarity = 0.0

        for k_idx in range(n_numeric):
            k = numeric_indices[k_idx]
            range_k = numericFeaturesRanges[k_idx]
            diff = X[i, k] - prototype[k]
            abs_diff = diff if diff >= 0 else -diff
            sim_k = 1.0 - (abs_diff / range_k)
            sum_similarity += sim_k

        for k_idx in range(n_categorical):
            k = categorical_indices[k_idx]
            if X[i, k] == prototype[k]:
                sum_similarity += 1.0

        similarities[i] = sum_similarity / n_features

    return similarities
