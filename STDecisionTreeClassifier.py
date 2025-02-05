import numpy as np

class STDecisionTreeClassifier:
    
    def __init__(self, isCategorical, max_depth=4):
        self.max_depth = max_depth
        self.tree = None
        self.isCategorical=isCategorical
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        
        unique_labels = np.unique(y)

        if len(unique_labels) == 1:
            return unique_labels[0]  # Pure class

        if self.max_depth is not None and depth >= self.max_depth:
            return np.bincount(y).argmax() 
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return np.bincount(y).argmax()
        
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return np.bincount(y).argmax()  # Si no hay datos en una de las ramas, devuelve la clase mayoritaria

        
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
                
                if self.isCategorical[feature]:
                    left_indices = X[:, feature] == threshold
                    right_indices = ~left_indices
                else:
                    left_indices = X[:, feature] <= threshold
                    right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                score = self._gini_index(y[left_indices], y[right_indices]) 

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
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
        
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])