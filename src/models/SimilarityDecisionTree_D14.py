import numpy as np

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.


D13 with gini for threshold
'''

class SimilarityDecisionTree_D14:
    
    def __init__(self, isClassifier = True, categoricalFeatures = None, max_depth = 4, n_jobs = -1, par=2000000, min_samples_leaf=3):
        self.max_depth = max_depth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
        self.numericFeaturesRanges = None
        self.n_jobs = n_jobs
        self.isClassifier = isClassifier
        self._X_fit = None
        self._y_fit = None
        self.min_samples_leaf = min_samples_leaf
        self.par = par
    
    def fit(self, X, y):

        self._X_fit = X
        self._y_fit = y

        self.compute_categorical_mask(X.shape[1])
        self.compute_numeric_ranges(X)

        initial_indices = np.arange(X.shape[0])
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

        similaritiesToPrototype = self.gower_similarity_to_prototype(current_X, prototype)

        threshold = self._find_best_threshold(similaritiesToPrototype, current_y)

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

        left_child = self._build_tree(leftIndices, depth + 1)
        right_child = self._build_tree(rightIndices, depth + 1)

        return {"prototype": prototype,"threshold": threshold,"left": left_child,"right": right_child}

    def predict(self, X):

        y_pred = np.empty(X.shape[0], dtype=np.float64)
        initial_indices = np.arange(X.shape[0])
        
        self._traverse_tree(X, self.tree, initial_indices, y_pred)

        return y_pred

    def _traverse_tree(self, X, node, indices, y_pred):

        if not isinstance(node, dict):
            y_pred[indices] = node
            return

        current_X_subset = X[indices]

        similaritiesToPrototype = self.gower_similarity_to_prototype(
            current_X_subset, node["prototype"]
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
    
    def gower_similarity_to_prototype(self, X, prototype):

        numMask = ~self.isCategorical
        catMask = self.isCategorical
        
        numericaDifferences = 1 - (np.abs( X[:,numMask] - prototype[numMask] )/ self.numericFeaturesRanges)
        numericaDifferences = np.sum(numericaDifferences, axis=1)

        categoricalDifferences = X[:, catMask] != prototype[catMask]
        categoricalDifferences = np.sum(~categoricalDifferences, axis=1)

        similarities = (numericaDifferences + categoricalDifferences) / X.shape[1]

        return similarities
    
    def compute_numeric_ranges(self, X):

        self.numericFeaturesRanges = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            if not self.isCategorical[i]:

                col = X[:, i]
                max = np.nanmax(col)
                min = np.nanmin(col)

                self.numericFeaturesRanges[i] = np.abs(max - min) if max > min else 1
        
        self.numericFeaturesRanges = self.numericFeaturesRanges[~self.isCategorical]

    def compute_categorical_mask(self, n):
        self.isCategorical = np.zeros(n, dtype=bool)
        if self.categoricalFeatures is not None:
            self.isCategorical[self.categoricalFeatures] = True 

    def _find_best_threshold(self, similarities, y):
        thresholdsList = np.linspace(similarities.min(), similarities.max(), num=10)
        
        best_threshold = None
        best_score = float('inf')
        
        for threshold in thresholdsList:
            left_mask = similarities <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            if self.isClassifier:
                score = self._gini_index(y[left_mask], y[right_mask])
            else:
                score = self._variance_reduction(y[left_mask], y[right_mask])
            
            if score < best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold

    def _gini_index(self, left_y, right_y):
        def gini(y):
            target = y.astype(int)
            proportions = np.bincount(target) / len(target)
            return 1 - np.sum(proportions ** 2)
        
        left_size, right_size = len(left_y), len(right_y)
        total_size = left_size + right_size

        return (left_size / total_size) * gini(left_y) + (right_size / total_size) * gini(right_y)

    def _variance_reduction(self, left_y, right_y):
        left_size, right_size = len(left_y), len(right_y)
        total_size = left_size + right_size

        left_var = np.var(left_y)
        right_var = np.var(right_y)

        return (left_size / total_size) * left_var + (right_size / total_size) * right_var





