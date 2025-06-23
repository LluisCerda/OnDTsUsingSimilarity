import numpy as np
from joblib import Parallel, delayed
from pandas import isna

'''
This class is a decision tree classifier/regressor that uses the Gower distance 
to compute the similarity between samples.
It supports n-ary splits based on quantiles of similarity scores.

D15: Original version extended to support n-ary splits using linspace for similarity thresholds.
Further refined for clarity, robustness, and to ensure input data is not modified.
'''

class SimilarityDecisionTree_D17:
    
    def __init__(self, isClassifier=True, categoricalFeatures=None, maxDepth=7, 
                 parallelizationThreshold=500000, minSamplesLeaf=1, 
                 weights=None, nChildren=2): 
        
        if nChildren < 2:
            raise ValueError("N_CHILDREN must be 2 or greater.")
        
        self.MAX_DEPTH = maxDepth
        self.MIN_BRANCHES_FOR_SPLIT = 2
        self.SIMILARITY_EPSILON = 1e-9 
        self.N_JOBS = -1
        self.N_CHILDREN = nChildren
        self.IS_CLASSIFIER = isClassifier
        self.MIN_SAMPLES_LEAF = minSamplesLeaf
        self.PARALLELIZATION_THRESHOLD = parallelizationThreshold

        self.categoricalFeatures = categoricalFeatures
        
        self.isCategorical = None
        self.isNumeric = None

        self.tree = None

        self.X_train_normalized = None 
        self.Y_train = None            
        
        self.weights = weights  

        self.numericFeaturesRanges = None
        self.minNumericFeatures = None

    def fit(self, X, y):

        self.data_preprocessing(X, y)

        initial_indices = np.arange(self.X_train_normalized.shape[0])
        self.tree = self.build_tree(initial_indices, depth=1)

    def build_tree(self, indices, depth):

        n_current_samples = len(indices)

        if self.stopping_criteria(indices, depth):
            return self._get_leaf_value(indices)
                
        current_X_node_subset = self.X_train_normalized[indices]

        prototype_subset_idx = np.random.randint(0, n_current_samples)
        prototype_vector = current_X_node_subset[prototype_subset_idx]

        similarities_to_prototype = self.gower_similarity_to_prototype(current_X_node_subset, prototype_vector)

        min_sim = np.nanmin(similarities_to_prototype)
        max_sim = np.nanmax(similarities_to_prototype)

        if max_sim - min_sim < self.SIMILARITY_EPSILON:
            return self._get_leaf_value(indices)

        threshold_edges, num_actual_bins = self.compute_threshold_selection(similarities_to_prototype, min_sim, max_sim)

        if num_actual_bins < self.MIN_BRANCHES_FOR_SPLIT :
             return self._get_leaf_value(indices)
        
        bin_assignments = np.digitize(similarities_to_prototype, bins=threshold_edges, right=True)
        
        child_indices_list = []
        for i in range(num_actual_bins):
            mask_local = (bin_assignments == i)
            child_indices_list.append(indices[mask_local])

        populated_child_indices = [child_idx_group for child_idx_group in child_indices_list if len(child_idx_group) > 0]

        if len(populated_child_indices) < self.MIN_BRANCHES_FOR_SPLIT:
            return self._get_leaf_value(indices)

        children_nodes = [None] * num_actual_bins
        
        tasks_for_parallel = []
        task_original_bin_indices = []

        for i in range(num_actual_bins):
            if len(child_indices_list[i]) > 0: 
                tasks_for_parallel.append(delayed(self.build_tree)(child_indices_list[i], depth + 1))
                task_original_bin_indices.append(i)

        use_parallel = (n_current_samples * self.X_train_normalized.shape[1] >= self.PARALLELIZATION_THRESHOLD and \
                        self.N_JOBS != 1 and \
                        len(tasks_for_parallel) > 1) 

        if use_parallel:
            results = Parallel(n_jobs=self.N_JOBS)(tasks_for_parallel)
            for i, res_node in enumerate(results):
                children_nodes[task_original_bin_indices[i]] = res_node
        else: 
            for i in range(len(tasks_for_parallel)):
                current_child_indices_for_task = child_indices_list[task_original_bin_indices[i]]
                children_nodes[task_original_bin_indices[i]] = self.build_tree(current_child_indices_for_task, depth + 1)
        
        return {"prototype": prototype_vector, 
                "threshold_edges": threshold_edges, 
                "children": children_nodes} 

    def predict(self, X):
        if self.tree is None:
            raise RuntimeError("Tree has not been fitted. Call fit() before predict().")

        X_processed = np.array(X)
        
        if self.IS_CLASSIFIER and self.Y_train is not None:
             y_pred = np.empty(X_processed.shape[0], dtype=self.Y_train.dtype)
        else: 
             y_pred = np.empty(X_processed.shape[0], dtype=np.float64)

        initial_indices = np.arange(X_processed.shape[0])

        if np.any(self.isNumeric):
            if self.minNumericFeatures is None or self.numericFeaturesRanges is None:
                raise RuntimeError("Numeric normalization parameters not found. Fit the tree first.")
            
            X_processed[:, self.isNumeric] = (X_processed[:, self.isNumeric] - self.minNumericFeatures) / self.numericFeaturesRanges
        
        self._traverse_tree(X_processed, self.tree, initial_indices, y_pred)
        return y_pred

    def _traverse_tree(self, X_normalized_full, node, indices_in_X_full, y_pred_array):
        
        if not isinstance(node, dict):
            y_pred_array[indices_in_X_full] = node
            return

        current_X_subset_for_node = X_normalized_full[indices_in_X_full]
        
        prototype_vector = node["prototype"]
        threshold_edges = node["threshold_edges"]

        similarities_to_prototype = self.gower_similarity_to_prototype(current_X_subset_for_node, prototype_vector)
        
        bin_assignments = np.digitize(similarities_to_prototype, bins=threshold_edges, right=True)
        
        for i in range(len(node["children"])): 
            child_node_definition = node["children"][i]
            if child_node_definition is not None: 

                mask_local_to_subset = (bin_assignments == i)
                child_global_indices = indices_in_X_full[mask_local_to_subset]
                
                if len(child_global_indices) > 0:
                    self._traverse_tree(X_normalized_full, child_node_definition, child_global_indices, y_pred_array)
    
    def gower_similarity_to_prototype(self, X_normalized_subset, prototype_vector_normalized):
        
        n_samples = X_normalized_subset.shape[0]
        total_similarity_accumulator = np.zeros(n_samples, dtype=np.float64)

        if np.any(self.isNumeric):
            
            X_num = X_normalized_subset[:, self.isNumeric].astype(np.float64)
            proto_num = prototype_vector_normalized[self.isNumeric].astype(np.float64)
            weights_num = self.weights[self.isNumeric]

            sim = 1.0 - np.abs(X_num - proto_num)  

            mask_X_nan = isna(X_num)
            mask_proto_nan = isna(proto_num)
            mask_proto_nan_broadcasted = np.broadcast_to(mask_proto_nan, X_num.shape)
            invalid_X_mask = mask_X_nan | mask_proto_nan_broadcasted

            sim[invalid_X_mask] = 0.0

            valid_weights = np.broadcast_to(weights_num, X_num.shape).copy()
            valid_weights[invalid_X_mask] = 0.0

            total_similarity_accumulator += np.sum(sim * valid_weights, axis=1)

        if np.any(self.isCategorical):

            X_cat = X_normalized_subset[:, self.isCategorical]
            proto_cat = prototype_vector_normalized[self.isCategorical]
            weights_cat = self.weights[self.isCategorical]

            mask_X_nan = (X_cat == None) | (X_cat != X_cat)  
            mask_proto_nan = (proto_cat == None) | (proto_cat != proto_cat)
            mask_proto_nan_broadcasted = np.broadcast_to(mask_proto_nan, X_cat.shape)
            invalid_mask = mask_X_nan | mask_proto_nan_broadcasted

            matches = (X_cat == proto_cat) & ~invalid_mask
            matches = matches.astype(float)

            valid_weights = np.broadcast_to(weights_cat, X_cat.shape).copy()
            valid_weights[invalid_mask] = 0.0

            total_similarity_accumulator += np.sum(matches * valid_weights, axis=1)

        return total_similarity_accumulator

    def compute_categorical_mask(self, n_features):
        self.isCategorical = np.zeros(n_features, dtype=bool)
        if self.categoricalFeatures is not None:
            valid_cat_features = [cf for cf in self.categoricalFeatures if 0 <= cf < n_features]
            self.isCategorical[valid_cat_features] = True 
        self.isNumeric = ~self.isCategorical

    def normalize_data(self, X_float_copy):
        
        if not np.any(self.isNumeric): 
            self.minNumericFeatures = np.array([]) 
            self.numericFeaturesRanges = np.array([])
            return X_float_copy 

        X_numeric_part = X_float_copy[:, self.isNumeric]
        
        X_min_numeric = np.nanmin(X_numeric_part, axis=0)
        X_max_numeric = np.nanmax(X_numeric_part, axis=0)

        denom_numeric = X_max_numeric - X_min_numeric
        denom_numeric[denom_numeric == 0] = 1 

        self.minNumericFeatures = X_min_numeric
        self.numericFeaturesRanges = denom_numeric

        X_float_copy[:, self.isNumeric] = (X_numeric_part - self.minNumericFeatures) / self.numericFeaturesRanges

        return X_float_copy
    
    def _get_leaf_value(self, indices):

        current_y_subset = self.Y_train[indices]
        if self.IS_CLASSIFIER:
            unique_classes, counts = np.unique(current_y_subset, return_counts=True)
            return unique_classes[np.argmax(counts)]
        else:
            return np.mean(current_y_subset)
    
    def stopping_criteria(self, indices, depth):

        n_current_samples = len(indices)
        current_y_for_node = self.Y_train[indices]

        is_pure_node = False
        if self.IS_CLASSIFIER:
            if len(np.unique(current_y_for_node)) == 1:
                is_pure_node = True
        
        return depth >= self.MAX_DEPTH or \
           is_pure_node or \
           n_current_samples < self.MIN_SAMPLES_LEAF or \
           n_current_samples < self.MIN_BRANCHES_FOR_SPLIT * self.MIN_SAMPLES_LEAF

    def compute_threshold_selection(self, similarities_to_prototype, min_sim, max_sim):
        
        if self.N_CHILDREN > 2:
            split_boundary_points = np.linspace(min_sim, max_sim, self.N_CHILDREN + 1)
            
            threshold_edges = np.unique(split_boundary_points[1:-1])
            num_actual_bins = len(threshold_edges) + 1
            
        else:
            threshold_edges = [np.nanmean(similarities_to_prototype)]
            num_actual_bins = 2
        
        return threshold_edges, num_actual_bins
    
    def data_preprocessing(self, X, y):

        self.compute_categorical_mask(X.shape[1])
        
        self.X_train_normalized = self.normalize_data(X) 
        self.Y_train = np.array(y)

        if self.weights is not None:
            sum_weights = np.sum(self.weights)
            if sum_weights == 0 or len(self.weights) == 0: 
                self.weights = None 
            else:
                self.weights = self.weights / sum_weights
        
        if self.weights is None: 
            self.weights = np.ones(self.X_train_normalized.shape[1]) / self.X_train_normalized.shape[1]
        elif len(self.weights) != self.X_train_normalized.shape[1]:
            raise ValueError(f"Length of weights ({len(self.weights)}) must match number of features ({self.X_train_normalized.shape[1]}).")

    def data_encoding(self, X, y= None):

        if y is not None:
            _, y = np.unique(y, return_inverse=True)
            y = y.astype(np.float64)

        for i in range(len(X[0])):
            if self.isCategorical[i]:
                mask = isna(X[:, i])
                
                _, X[:, i] = np.unique(X[:, i], return_inverse=True)

        X = X.astype(np.float64)
 
        return X, y