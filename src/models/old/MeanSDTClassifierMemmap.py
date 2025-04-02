import numpy as np

class MeanSDTClassifierMemmap:
    
    def __init__(self, isCategorical, max_depth=4):
        self.max_depth = max_depth
        self.isCategorical = isCategorical
        self.tree = None
        self.n_samples = None
        self.n_features = None
    
    def fit(self, X, y):

        self.gower_matrix(X)
        self.n_samples, self.n_features = X.shape
        self.tree = self._build_tree(X, y, np.arange(self.n_samples), depth=0)
    
    def _build_tree(self, X, y, indices, depth):

        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        
        medoid_idx, distances_to_medoid, mean = self._compute_medoid(indices)
        medoid = X[medoid_idx]
        
        left_indices = distances_to_medoid <= mean
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.bincount(y).argmax()
        
        left_subtree = self._build_tree(X, y[left_indices], indices[left_indices], depth + 1)
        right_subtree = self._build_tree(X, y[right_indices], indices[right_indices], depth + 1)
        
        return {"medoid": medoid, "threshold": mean, "left": left_subtree, "right": right_subtree}
    
    def _compute_medoid(self, indices):
        
        distances = np.memmap('distances.dat', dtype='float32', mode='r', shape=(self.n_samples, self.n_samples))

        row_sums = np.full(self.n_samples, np.inf)  

        for i in indices:
            row = distances[i, indices]
            row_sums[i] = np.sum(row) 

        medoid_idx = np.argmin(row_sums)

        distances_to_medoid = distances[:, medoid_idx] 
        distances_to_medoid = distances_to_medoid[indices] 
        mean_distance = np.mean(distances_to_medoid)

        return medoid_idx, distances_to_medoid, mean_distance

    def gower_matrix(self, X):
        X = np.asarray(X)
        Y = np.asarray(X)
            
        n_samples, n_features = X.shape
        distances = np.memmap('distances.dat', dtype='float32', mode='w+', shape=(n_samples, n_samples))
        
        for f in range(n_features):
            if self.isCategorical is not None and f in self.isCategorical:  
                distances += (X[:, f, None] != Y[:, f]).astype(float)
            else: 
                distances += np.abs(X[:, f, None] - Y[:, f])

        
        distances /= n_features


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
