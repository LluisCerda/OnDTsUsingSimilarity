import numpy as np
import utils as utils

class MeanSimilarityDTClassifier:
    
    def __init__(self, isCategorical, max_depth=4):
        self.max_depth = max_depth
        self.isCategorical = isCategorical
        self.tree = None
        self.full_distances = None  # Store full precomputed distances
    
    def fit(self, X, y):
        """Compute full distance matrix once and build the tree."""
        self.full_distances = self.gower_distance(X, X, cat_features=self.isCategorical)
        self.tree = self._build_tree(X, y, depth=0, distances=self.full_distances)
    
    def _build_tree(self, X, y, depth, distances):
        """Recursively build the tree using precomputed distances."""
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        
        medoid_idx, distances_to_medoid, mean = self._compute_medoid(distances)
        medoid = X[medoid_idx]  
        
        left_indices = distances_to_medoid <= mean
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.bincount(y).argmax()

        # Pass only relevant distances for each subtree
        left_distances = distances[np.ix_(left_indices, left_indices)]
        right_distances = distances[np.ix_(right_indices, right_indices)]
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1, left_distances)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1, right_distances)
        
        return {"medoid": medoid, "threshold": mean, "left": left_subtree, "right": right_subtree}
    
    def _compute_medoid(self, distances):
        """
        np.sum(distances, axis=1) -> row_sums = distances @ np.ones(distances.shape[1])
        Matrix multiplication is faster than np.sum for large matrices.

        np.argmin() -> np.arpartition(row_sums, 1)[0]
        avoids sorting the entire array, only finds the index of the smallest element.
        
        """
        row_sums = distances @ np.ones(distances.shape[1])  # Fast summation
        medoid_idx = np.argpartition(row_sums, 1)[0]  # Faster min search

        distances_to_medoid = distances[:, medoid_idx]  
        mean_distances = np.mean(distances_to_medoid)
        
        return medoid_idx, distances_to_medoid, mean_distances
    
    def gower_distance(self, X, Y=None, cat_features=None):
        """Compute the Gower distance matrix."""
        X = np.asarray(X)
        Y = np.asarray(Y) if Y is not None else X  
        
        n_samples_x, n_features = X.shape
        n_samples_y = Y.shape[0]
        
        if cat_features is None:
            cat_features = np.array([False] * n_features)  

        D = np.zeros((n_samples_x, n_samples_y))

        for f in range(n_features):
            if cat_features[f]:  
                D += (X[:, f, None] != Y[:, f]).astype(float)
            else:
                D += np.abs(X[:, f, None] - Y[:, f])
        
        D /= n_features  
        return D
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
    
        distance = self.gower_distance(x.reshape(1, -1), node["medoid"].reshape(1, -1), cat_features=self.isCategorical)
        if distance[0][0] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])
