import numpy as np
import time

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.


"Gower's" distance is computed in each node.
'''

class MeanSimilarityDTClassifier_D4:
    
    def __init__(self, categoricalFeatures, max_depth=4):
        self.max_depth = max_depth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
    
    def fit(self, X, y):

        self.isCategorical = np.zeros(X.shape[1], dtype=bool)
        if self.categoricalFeatures is not None:
            self.isCategorical[self.categoricalFeatures] = True 

        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        
        
        medoid, distances, mean = self._compute_medoid(X)
        left_indices = distances <= mean
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.bincount(y).argmax()
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {"medoid": medoid, "threshold": mean, "left": left_subtree, "right": right_subtree}
    
    def _compute_medoid(self, X):
        
        distances = self.gower_distances(X)  
    
        medoid_idx = np.argmin(distances)
        medoid = X[medoid_idx]  
        distances_to_medoid = self.gower_distances_to_medoid(X, medoid_idx) 

        mean_distances = np.median(distances_to_medoid) #mean
        
        return medoid, distances_to_medoid, mean_distances


    def gower_matrix(X, categoricalFeatures):
        X = np.array(X)
        n_samples, n_features = X.shape
        D = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                diff = np.zeros(n_features)
                
                for f in range(n_features):
                    if categoricalFeatures[f]:  # Categorical feature
                        diff[f] = 0 if X[i, f] == X[j, f] else 1
                    else:  # Numerical feature
                        max_f = np.nanmax(X[:, f])
                        min_f = np.nanmin(X[:, f])
                        range_f = max_f - min_f if max_f > min_f else 1  # Avoid division by zero
                        diff[f] = abs(X[i, f] - X[j, f]) / range_f
                
                D[i, j] = D[j, i] = np.mean(diff)
        
        return D

    
    def gower_distances(self, X):

        X = np.asarray(X)
        n_samples = X.shape[0]
        
        sum_distances = np.zeros(n_samples)  
        
        X_num = X[:, ~self.isCategorical].astype(float)
        X_cat = X[:, self.isCategorical]
        
        num_features = X_num.shape[1]  
        cat_features = X_cat.shape[1]  

        for i in range(n_samples):
            sample_time = time.time()

            num_differences = np.abs(X_num[i] - X_num) 
            num_differences = np.sum(num_differences, axis=1)

            cat_differences = np.sum(X_cat[i] != X_cat, axis=1)

            row_distances = (num_differences + cat_differences)
            
            sum_distances[i] = np.sum(row_distances) / (num_features + cat_features)

            if i % 1000 == 0: 
                sample_time = time.time() - sample_time
                remaining_time = sample_time * (n_samples - i - 1)
                print(f"Sample {i}/{n_samples} - Estimated remaining time: {remaining_time:.2f} sec ({remaining_time/60:.2f} min)")

        return sum_distances

    def gower_distances_to_medoid(self, X, medoid_idx):
        
        X = np.asarray(X)
        medoid = X[medoid_idx]  

        X_num = X[:, ~self.isCategorical].astype(float)
        X_cat = X[:, self.isCategorical]

        Medoid_num = medoid[~self.isCategorical].astype(float)
        Medoid_cat = medoid[self.isCategorical]
        
        num_features = X_num.shape[1]  
        cat_features = X_cat.shape[1]  

        
        num_differences = np.abs(X_num - Medoid_num) 
        num_differences = np.sum(num_differences, axis=1)

        cat_differences = np.sum(X_cat != Medoid_cat, axis=1)

        distances = (num_differences + cat_differences) / (num_features + cat_features)

        return distances


    def gower_distance(self, x, y):
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        distances = np.where(
            self.isCategorical[:, None],
            (x != y).astype(float),  
            np.abs(x - y)         
        )

        return np.mean(distances)  
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
    
        distance = self.gower_distance(x.reshape(1, -1), node["medoid"].reshape(1, -1))  
        if distance <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])
