# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed

from myGowerV2 import gower_similarity_to_prototype_numba

'''

D11 optimized: Uses indices to avoid data copying during recursion. And myGowerV2

'''

class SimilarityDecisionTree_D12:

    def __init__(self, isClassifier=True, categoricalFeatures=None, max_depth=4, n_jobs=-1, par=100000, min_samples_leaf=3):
        self.max_depth = max_depth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
        self.numericFeaturesRanges = None
        self.n_jobs = n_jobs
        self.isClassifier = isClassifier
        self.par = par # Threshold for parallelization (based on n_samples * n_features in node)
        self.min_samples_leaf = min_samples_leaf # Minimum samples required to split a node further
        self._X_fit = None # Store reference to original X
        self._y_fit = None # Store reference to original y

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self._X_fit = X
        self._y_fit = y

        self.compute_categorical_mask(n_features)
        self.compute_numeric_ranges(X) 

        initial_indices = np.arange(n_samples)
        self.tree = self._build_tree(initial_indices, depth=0)

    def _build_tree(self, indices, depth):
        
        current_y = self._y_fit[indices]
        n_current_samples = len(indices)

        if self.isClassifier:
            unique_classes, counts = np.unique(current_y, return_counts=True)
            if depth >= self.max_depth or len(unique_classes) == 1 or n_current_samples < self.min_samples_leaf * 2: # Need enough samples for a potential split
                return unique_classes[np.argmax(counts)]
        else: 
             if depth >= self.max_depth or n_current_samples < self.min_samples_leaf * 2:
                return np.mean(current_y)
                
        current_X = self._X_fit[indices]

        prototype_idx_in_indices = np.random.randint(0, n_current_samples)
        prototype_global_idx = indices[prototype_idx_in_indices]
        prototype = self._X_fit[prototype_global_idx]

        similaritiesToPrototype = gower_similarity_to_prototype_numba(
            current_X, prototype, self.isCategorical, self.numericFeaturesRanges
        )

        threshold = np.median(similaritiesToPrototype)

        leftMask_local = similaritiesToPrototype <= threshold
        rightMask_local = ~leftMask_local 

        leftIndices = indices[leftMask_local]
        rightIndices = indices[rightMask_local]

        if len(leftIndices) < self.min_samples_leaf or len(rightIndices) < self.min_samples_leaf:
            if self.isClassifier:
                 unique_classes, counts = np.unique(current_y, return_counts=True)
                 return unique_classes[np.argmax(counts)]
            else:
                 return np.mean(current_y)

        if n_current_samples * self._X_fit.shape[1] >= self.par and self.n_jobs != 1:
             results = Parallel(n_jobs=self.n_jobs)(
                 delayed(self._build_tree)(child_indices, depth + 1)
                 for child_indices in [leftIndices, rightIndices]
             )
             left_child = results[0]
             right_child = results[1]
        else:
             left_child = self._build_tree(leftIndices, depth + 1)
             right_child = self._build_tree(rightIndices, depth + 1)

        return {"prototype_idx": prototype_global_idx,
                "prototype": prototype,
                "threshold": threshold,
                "left": left_child,
                "right": right_child}

    def predict(self, X):

        y_pred = np.empty(X.shape[0], dtype=np.float64)
        initial_indices = np.arange(X.shape[0])
        
        self._traverse_tree(X, self.tree, initial_indices, y_pred)

        return y_pred

    def _traverse_tree(self, X, node, indices, y_pred):

        if not isinstance(node, dict):
            y_pred[indices] = node
            return

        # --- Node Splitting Logic (for prediction) ---
        # Select the subset of X corresponding to current indices
        current_X_subset = X[indices]

        similaritiesToPrototype = gower_similarity_to_prototype_numba(
            current_X_subset, node["prototype"], self.isCategorical, self.numericFeaturesRanges
        )

        leftMask_local = similaritiesToPrototype <= node["threshold"]
        rightMask_local = ~leftMask_local

        leftIndices = indices[leftMask_local]
        rightIndices = indices[rightMask_local]

        # Recursively traverse
        if len(leftIndices) > 0:
             self._traverse_tree(X, node["left"], leftIndices, y_pred)
        if len(rightIndices) > 0:
             self._traverse_tree(X, node["right"], rightIndices, y_pred)

    def compute_numeric_ranges(self, X):
        
        n_features = X.shape[1]
        all_ranges = np.zeros(n_features)
        numeric_mask = ~self.isCategorical

        numeric_cols = X[:, numeric_mask]

        min_vals = np.nanmin(numeric_cols, axis=0)
        max_vals = np.nanmax(numeric_cols, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges < 1e-9] = 1.0 
        all_ranges[numeric_mask] = ranges

        self.numericFeaturesRanges = all_ranges[numeric_mask]


    def compute_categorical_mask(self, n_features):
        self.isCategorical = np.zeros(n_features, dtype=bool)
        if self.categoricalFeatures is not None:
             valid_indices = [idx for idx in self.categoricalFeatures if 0 <= idx < n_features]
             self.isCategorical[valid_indices] = True