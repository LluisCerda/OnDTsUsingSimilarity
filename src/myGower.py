import numpy as np
import numba

@staticmethod
@numba.njit(cache=True, fastmath=True)
def gower_similarity_to_prototype_numba(X, prototype, isCategorical, numericFeaturesRanges):
 
    # n_samples = X.shape[0]
    # n_features = X.shape[1]

    # numeric_indices = np.where(~isCategorical)[0]
    # categorical_indices = np.where(isCategorical)[0]

    # similarities = np.zeros(n_samples, dtype=np.float64)

    # # Loop over samples (Numba optimizes this loop)
    # for i in range(n_samples):
    #     sum_similarity = 0.0

    #     # Numerical features
    #     for k_idx, k in enumerate(numeric_indices):

    #         abs_diff = np.abs(X[i, k] - prototype[k])

    #         range_k = numericFeaturesRanges[k_idx]

    #         sim_k = 1.0 - (abs_diff / range_k)
    #         sum_similarity += sim_k

    #     # Categorical features
    #     for k in categorical_indices:
    #         if X[i, k] == prototype[k]:
    #             sum_similarity += 1.0

    #     similarities[i] = sum_similarity / n_features

    # return similarities

    numMask = ~isCategorical
    catMask = isCategorical
    numericalRanges = numericFeaturesRanges

    numericaDifferences = 1 - (np.abs( X[:,numMask] - prototype[numMask] ) / numericalRanges)
    numericaDifferences = np.sum(numericaDifferences, axis=1)

    categoricalDifferences = X[:, catMask] != prototype[catMask]
    categoricalDifferences = np.sum(~categoricalDifferences, axis=1)

    return (numericaDifferences + categoricalDifferences) / X.shape[1]
