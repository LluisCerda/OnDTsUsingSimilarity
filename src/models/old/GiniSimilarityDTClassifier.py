import numpy as np
import utils as utils

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Tries all possible thresholds to find the one that minimizes the Gini impurity.
'''

class GiniSimilarityDTClassifier:
    
    def __init__(self, isCategorical, max_depth=4):
        self.max_depth = max_depth
        self.isCategorical = isCategorical
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        

        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        
        
        medoid, distances = self._compute_medoid(X)

        threshold = self._find_best_threshold(distances, y)
        if threshold is None:
            return np.bincount(y).argmax()
        
        left_indices = distances <= threshold
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.bincount(y).argmax()
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {"medoid": medoid, "threshold": threshold, "left": left_subtree, "right": right_subtree}
    
    def _compute_medoid(self, X):
        
        distances = self.gower_distance(X, cat_features=self.isCategorical)  


        medoid_idx = np.argmin(distances.sum(axis=1))  
        medoid = X[medoid_idx]  
        distances_to_medoid = distances[:, medoid_idx] 
        
        return medoid, distances_to_medoid
    
    def gower_distance(self, X, Y=None, cat_features=None):
        """
        Compute the Gower distance between rows in X and rows in Y.
        
        Parameters:
        - X: np.ndarray or list of shape (n_samples_x, n_features)
        - Y: np.ndarray or list of shape (n_samples_y, n_features) [optional]
        - cat_features: list or np.array of bool (True if categorical, False if numerical)
        
        Returns:
        - A distance matrix of shape (n_samples_x, n_samples_y)
        """
        X = np.asarray(X)
        Y = np.asarray(Y) if Y is not None else X  # If Y is not provided, compute pairwise distances in X
        
        n_samples_x, n_features = X.shape
        n_samples_y = Y.shape[0]
        
        if cat_features is None:
            cat_features = np.array([False] * n_features)  # Assume all features are numerical

        # Initialize distance matrix
        D = np.zeros((n_samples_x, n_samples_y))

        for f in range(n_features):
            if cat_features[f]:  
                # Categorical feature: 0 if same, 1 if different
                D += (X[:, f, None] != Y[:, f]).astype(float)
            else:
                # Numerical feature: Absolute difference (NO normalization)
                D += np.abs(X[:, f, None] - Y[:, f])
        
        # Average distance per feature
        D /= n_features  
        return D

    
    def _find_best_threshold(self, distances, y):
        
        unique_distances = np.unique(distances)
        
        best_threshold = None
        best_score = float('inf')
        
        for threshold in unique_distances:
            left_mask = distances <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            score = self._gini_index(y[left_mask], y[right_mask])
            
            if score < best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold

    
    def _gini_index(self, left_y, right_y):
        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)
        
        left_size, right_size = len(left_y), len(right_y)
        total_size = left_size + right_size
    
        return (left_size / total_size) * gini(left_y) + (right_size / total_size) * gini(right_y)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
    
        distance = self.gower_distance(x.reshape(1, -1), node["medoid"].reshape(1, -1))
        distance = self.gower_distance(x.reshape(1, -1), node["medoid"].reshape(1, -1), cat_features=self.isCategorical)  
        if distance[0][0] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])
