import numpy as np
from joblib import Parallel, delayed

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.

D13 with balances numeric differences and with weights for both numeric and categorical features.
'''

class SimilarityDecisionTree_D15:
    
    def __init__(self, isClassifier = True, categoricalFeatures = None, maxDepth = 4, nJobs = -1, parallelizationThreshold=500000, minSamplesLeaf=1, weights=None):
        self.maxDepth = maxDepth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
        self.nJobs = nJobs
        self.isClassifier = isClassifier
        self.X = None
        self.Y = None
        self.minSamplesLeaf = minSamplesLeaf
        self.parallelizationThreshold = parallelizationThreshold
        self.weights = weights

        self.numericFeaturesRanges = None
        self.minNumericFeatures = None
    
    def fit(self, X, y):

        self.compute_categorical_mask(X.shape[1])
        self.X = self.normalize_data(X)

        self.Y = y

        initial_indices = np.arange(X.shape[0])
        self.tree = self.build_tree(initial_indices, depth=0)
    
    def build_tree(self, indices, depth):
        
        current_y = self.Y[indices]
        n_current_samples = len(indices)

        if self.isClassifier:
            unique_classes, counts = np.unique(current_y, return_counts=True)
            if depth >= self.maxDepth or len(unique_classes) == 1 or n_current_samples < self.minSamplesLeaf * 2:
                return unique_classes[np.argmax(counts)]
        else: 
             if depth >= self.maxDepth or n_current_samples < self.minSamplesLeaf * 2:
                return np.mean(current_y)
                
        current_X = self.X[indices]

        prototype_idx_in_indices = np.random.randint(0, n_current_samples)
        prototype_global_idx = indices[prototype_idx_in_indices]
        prototype = self.X[prototype_global_idx]

        similaritiesToPrototype = self.gower_similarity_to_prototype(current_X, prototype)

        threshold = np.mean(similaritiesToPrototype)

        leftMask_local = similaritiesToPrototype <= threshold
        rightMask_local = ~leftMask_local 

        leftIndices = indices[leftMask_local]
        rightIndices = indices[rightMask_local]

        if len(leftIndices) < self.minSamplesLeaf or len(rightIndices) < self.minSamplesLeaf:
            if self.isClassifier:
                 unique_classes, counts = np.unique(current_y, return_counts=True)
                 return unique_classes[np.argmax(counts)]
            else:
                 return np.mean(current_y)

        if n_current_samples * self.X.shape[1] >= self.parallelizationThreshold and self.nJobs != 1:
            results = Parallel(n_jobs=self.nJobs)(
                delayed(self.build_tree)(child_indices, depth + 1)
                for child_indices in [leftIndices, rightIndices]
            )
            left_child = results[0]
            right_child = results[1]
        else:
            left_child = self.build_tree(leftIndices, depth + 1)
            right_child = self.build_tree(rightIndices, depth + 1)

        return {"prototype": prototype,"threshold": threshold,"left": left_child,"right": right_child}

    def predict(self, X):

        y_pred = np.empty(X.shape[0], dtype=np.float64)
        initial_indices = np.arange(X.shape[0])

        numMask = ~self.isCategorical
        X[:, numMask] = (X[:, numMask] - self.minNumericFeatures) / self.numericFeaturesRanges
        
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

        if self.weights is None:
            weights = np.ones(X.shape[1])
        else:
            weights = self.weights / np.sum(self.weights)

        num_weights = weights[numMask]
        num_diffs = 1 - np.abs(X[:, numMask] - prototype[numMask])
        numericaDifferences = np.sum(num_diffs * num_weights, axis=1)

        cat_weights = weights[catMask]
        cat_diffs = X[:, catMask] == prototype[catMask]
        categoricalDifferences = np.sum(cat_diffs * cat_weights, axis=1)

        similarities = (numericaDifferences + categoricalDifferences) / np.sum(weights)

        return similarities
    
    def compute_categorical_mask(self, n):

        self.isCategorical = np.zeros(n, dtype=bool)
        if self.categoricalFeatures is not None:
            self.isCategorical[self.categoricalFeatures] = True 

    def normalize_data(self, X):

        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)

        denom = np.abs(X_max - X_min)
        denom[denom == 0] = 1

        numMask = ~self.isCategorical

        self.minNumericFeatures = X_min[numMask]
        self.numericFeaturesRanges = denom[numMask]

        X[:, numMask] = (X[:, numMask] - X_min[numMask]) / denom[numMask]

        return X


