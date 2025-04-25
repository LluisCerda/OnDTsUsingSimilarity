# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed

from numba import njit

@njit
def gower_similarity_to_prototype_numba(X, prototype, isCategorical, numericFeaturesRanges):
    n_samples, n_features = X.shape
    similarities = np.empty(n_samples, dtype=np.float64)
    num_numeric = len(numericFeaturesRanges)
    num_categorical = np.sum(isCategorical)
    
    numeric_indices = np.where(~isCategorical)[0]
    categorical_indices = np.where(isCategorical)[0]
    
    # Create sliced arrays inside the Numba function if needed
    # This might depend on how the original Numba function was structured
    # Here, we access columns directly using indices.
    
    for i in range(n_samples):
        sim_num = 0.0
        range_idx = 0
        for k_idx in range(len(numeric_indices)):
            k = numeric_indices[k_idx]
            # Ensure range is not zero before division
            range_val = numericFeaturesRanges[range_idx]
            if range_val > 1e-9: # Use a small epsilon for floating point comparison
                    sim_num += 1.0 - (np.abs(X[i, k] - prototype[k]) / range_val)
            elif np.abs(X[i, k] - prototype[k]) < 1e-9: # If range is zero, similarity is 1 if values are equal
                    sim_num += 1.0
            # else similarity is 0 for this feature (implicitly handled by initialization)
            range_idx += 1


        sim_cat = 0.0
        for k_idx in range(len(categorical_indices)):
                k = categorical_indices[k_idx]
                if X[i, k] == prototype[k]:
                    sim_cat += 1.0

        # Denominator should be the total number of features used in the comparison
        # If a numeric range is 0, that feature might be excluded depending on exact definition
        # Assuming standard Gower: denominator is n_features unless weights are used
        total_features = n_features # Or adjust based on weighting/handling of zero-range features
        if total_features > 0:
                similarities[i] = (sim_num + sim_cat) / total_features
        else:
                similarities[i] = 1.0 # Or 0.0, depending on convention for zero features

    return similarities


'''
This class is a decision tree classifier/regressor that uses the Gower distance
to compute the similarity between samples. Splits based on median similarity
to a random prototype.

D12 optimized: Uses indices to avoid data copying during recursion.
Removed unused Python Gower implementation.
'''

class SimilarityDecisionTree_D12:

    def __init__(self, isClassifier=True, categoricalFeatures=None, max_depth=4, n_jobs=-1, par=500000, min_samples_leaf=3):
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
        # Input validation
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            
        if X.ndim == 1:
             X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        if n_samples != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self._X_fit = X
        self._y_fit = y

        self.compute_categorical_mask(n_features)
        self.compute_numeric_ranges(X) 

        initial_indices = np.arange(n_samples)
        self.tree = self._build_tree(initial_indices, depth=0)

    def _build_tree(self, indices, depth):
        
        current_y = self._y_fit[indices]
        n_current_samples = len(indices)

        is_leaf = False
        leaf_value = None

        if self.isClassifier:
            # Check for pure node or max depth or min samples
            unique_classes, counts = np.unique(current_y, return_counts=True)
            if depth >= self.max_depth or len(unique_classes) == 1 or n_current_samples < self.min_samples_leaf * 2: # Need enough samples for a potential split
                leaf_value = unique_classes[np.argmax(counts)]
                is_leaf = True
        else: # Regressor
            # Check for max depth or min samples
             if depth >= self.max_depth or n_current_samples < self.min_samples_leaf * 2:
                leaf_value = np.mean(current_y)
                is_leaf = True
                
        if is_leaf:
             return leaf_value

        # --- Node Splitting ---
        current_X = self._X_fit[indices]

        # Select random prototype *index* from current indices
        prototype_idx_in_indices = np.random.randint(0, n_current_samples)
        prototype_global_idx = indices[prototype_idx_in_indices]
        prototype = self._X_fit[prototype_global_idx]

        # Compute similarities for the current subset

        # for col in current_X.T:
        #      print(col.dtype)
        #      print(col)

        similaritiesToPrototype = gower_similarity_to_prototype_numba(
            current_X, prototype, self.isCategorical, self.numericFeaturesRanges
        )

        # Use median similarity as threshold
        threshold = np.median(similaritiesToPrototype)

        # Create masks based on the current subset
        leftMask_local = similaritiesToPrototype <= threshold
        rightMask_local = ~leftMask_local # Faster than recalculating > threshold

        # Get global indices for children
        leftIndices = indices[leftMask_local]
        rightIndices = indices[rightMask_local]

        # Check if split is valid (minimum samples in resulting children)
        if len(leftIndices) < self.min_samples_leaf or len(rightIndices) < self.min_samples_leaf:
            # Cannot split further, make this node a leaf
            if self.isClassifier:
                 unique_classes, counts = np.unique(current_y, return_counts=True)
                 return unique_classes[np.argmax(counts)]
            else:
                 return np.mean(current_y)

        # --- Recursive Calls (Parallel or Sequential) ---
        node_size = n_current_samples * self._X_fit.shape[1]

        if node_size >= self.par and self.n_jobs != 1: # Check n_jobs too
             results = Parallel(n_jobs=self.n_jobs)(
                 delayed(self._build_tree)(child_indices, depth + 1)
                 for child_indices in [leftIndices, rightIndices]
             )
             left_child = results[0]
             right_child = results[1]
        else:
             left_child = self._build_tree(leftIndices, depth + 1)
             right_child = self._build_tree(rightIndices, depth + 1)

        return {"prototype_idx": prototype_global_idx, # Store index for potential inspection
                "prototype": prototype,
                "threshold": threshold,
                "left": left_child,
                "right": right_child}

    def predict(self, X):
        # Input validation
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
             X = X.reshape(-1, 1)
        if self._X_fit is None:
            raise ValueError("Tree has not been fitted yet.")
        if X.shape[1] != self._X_fit.shape[1]:
             raise ValueError(f"Input has {X.shape[1]} features, but tree was fitted with {self._X_fit.shape[1]} features.")


        # Preallocate prediction array based on expected output type
        if self.isClassifier:
             # Determine dtype from fitted y - assume integer classes
             # More robustly, could store unique classes or use object dtype
             output_dtype = self._y_fit.dtype if np.issubdtype(self._y_fit.dtype, np.integer) else object
             y_pred = np.empty(X.shape[0], dtype=output_dtype)
        else:
             # Regressor output is float
             y_pred = np.empty(X.shape[0], dtype=np.float64)


        initial_indices = np.arange(X.shape[0])
        
        # Call traverse which now directly populates y_pred
        self._traverse_tree_predict(X, self.tree, initial_indices, y_pred)

        return y_pred

    def _traverse_tree_predict(self, X, node, indices, y_pred):
        # Base case: reached a leaf node
        if not isinstance(node, dict):
            y_pred[indices] = node # Assign leaf value to predictions for these indices
            return

        # --- Node Splitting Logic (for prediction) ---
        # Select the subset of X corresponding to current indices
        current_X_subset = X[indices]

        # Compute similarities for the current subset against this node's prototype
        similaritiesToPrototype = gower_similarity_to_prototype_numba(
            current_X_subset, node["prototype"], self.isCategorical, self.numericFeaturesRanges
        )

        # Create local masks based on threshold
        leftMask_local = similaritiesToPrototype <= node["threshold"]
        rightMask_local = ~leftMask_local

        # Map local masks back to original indices
        leftIndices = indices[leftMask_local]
        rightIndices = indices[rightMask_local]

        # Recursively traverse
        if len(leftIndices) > 0:
             self._traverse_tree_predict(X, node["left"], leftIndices, y_pred)
        if len(rightIndices) > 0:
             self._traverse_tree_predict(X, node["right"], rightIndices, y_pred)
        # No need to merge results, y_pred is modified in place


    def compute_numeric_ranges(self, X):
        n_features = X.shape[1]
        # Initialize ranges for *all* features first, then filter
        all_ranges = np.zeros(n_features)
        numeric_mask = ~self.isCategorical

        # Calculate range only for numeric columns identified by the mask
        numeric_cols = X[:, numeric_mask]
        if numeric_cols.shape[1] > 0: # Avoid errors if no numeric columns
             min_vals = np.nanmin(numeric_cols, axis=0)
             max_vals = np.nanmax(numeric_cols, axis=0)
             ranges = max_vals - min_vals
             # Handle cases where min == max (range is 0) -> set range to 1
             ranges[ranges < 1e-9] = 1.0 # Use epsilon for float comparison
             all_ranges[numeric_mask] = ranges

        # Store only the ranges for numeric features, in their original order
        self.numericFeaturesRanges = all_ranges[numeric_mask]


    def compute_categorical_mask(self, n_features):
        self.isCategorical = np.zeros(n_features, dtype=bool)
        if self.categoricalFeatures is not None:
             # Ensure indices are valid
             valid_indices = [idx for idx in self.categoricalFeatures if 0 <= idx < n_features]
             self.isCategorical[valid_indices] = True