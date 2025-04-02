import numpy as np
import utils as utils
import time

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.


Gower's distance is computed in each node.

Deprecated since gower's distance is more efficient in D4
'''

class MeanSimilarityDTClassifier_D2:
    
    def __init__(self, isCategorical, max_depth=4):
        self.max_depth = max_depth
        self.isCategorical = isCategorical
        self.tree = None
    
    def fit(self, X, y):
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

        mean_distances = np.mean(distances_to_medoid)
        
        return medoid, distances_to_medoid, mean_distances
    
    def gower_distances(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        sum_distances = np.zeros(n_samples) 

        for i in range(n_samples):

            sample_time = time.time()

            row_distances = np.zeros(n_samples) 

            for f in range(n_features):
                if self.isCategorical is not None and f in self.isCategorical:
                    row_distances += (X[i, f] != X[:, f]).astype(float)
                else:
                    row_distances += np.abs(X[i, f] - X[:, f]).astype(float)
            
            row_distances /= n_features 
            sum_distances[i] = np.sum(row_distances) 

            sample_time = time.time() - sample_time

            if i % 100 == 0:
                print("Sample time: ", sample_time)
                print("Estimated time: ", sample_time * (n_samples - i) / 60, " minutes")

        return sum_distances

    def gower_distances_to_medoid(self, X, medoid_idx):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        medoid = X[medoid_idx]  
        distances = np.zeros(n_samples)  

        for f in range(n_features):
            if self.isCategorical is not None and f in self.isCategorical:
                distances += (medoid[f] != X[:, f]).astype(float)  
            else:
                distances += np.abs(medoid[f] - X[:, f]).astype(float)

        distances /= n_features  

        return distances  


    def gower_distance(self, x, y):
        
        x = np.asarray(x)
        y = np.asarray(y)
        
        n_features = x.shape[0]
        
        distances = np.where(
            self.isCategorical is not None and np.arange(n_features)[:, None] in self.isCategorical,
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
