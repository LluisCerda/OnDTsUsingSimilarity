import numpy as np
from pyvis.network import Network

class STDecisionTreeRegression:
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        
        # Regression: Stop if depth limit reached, return mean value
        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(y)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y) 
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}
    
    def _best_split(self, X, y):

        _, num_features = X.shape
        best_score = float('inf')
        best_feature, best_threshold = None, None
        
        for feature in range(num_features):
            
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Use Gini for classification, MSE for regression
                score = self._mse(y[left_indices], y[right_indices])
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _mse(self, left_y, right_y):
        """Compute Mean Squared Error for regression."""
        def mse(y):
            return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0
        
        left_size, right_size = len(left_y), len(right_y)
        total_size = left_size + right_size

        return (left_size / total_size) * mse(left_y) + (right_size / total_size) * mse(right_y)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])

