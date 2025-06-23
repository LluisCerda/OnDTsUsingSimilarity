import numpy as np
from joblib import Parallel, delayed

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.


D10 with self.X and better prediction.
'''

class SimilarityDecisionTree_D13:
    
    def __init__(self, isClassifier = True, categoricalFeatures = None, maxDepth = 4, nJobs = -1, parallelizationThreshold=500000, minSamplesLeaf=3):
        self.maxDepth = maxDepth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
        self.numericFeaturesRanges = None
        self.nJobs = nJobs
        self.isClassifier = isClassifier
        self.X = None
        self.Y = None
        self.minSamplesLeaf = minSamplesLeaf
        self.parallelizationThreshold = parallelizationThreshold
    
    def fit(self, X, y):

        self.X = X
        self.Y = y

        self.compute_categorical_mask()
        self.compute_numeric_ranges()

        initial_indices = np.arange(X.shape[0])
        self.tree = self.build_tree(initial_indices, depth=0)
    
    def build_tree(self, indices, depth):
        
        current_y = self.Y[indices]
        n_current_samples = len(indices)

        if self.isClassifier:
            unique_classes, counts = np.unique(current_y, return_counts=True)
            if depth >= self.maxDepth or len(unique_classes) == 1 or n_current_samples < self.minSamplesLeaf * 2: # Need enough samples for a potential split
                return unique_classes[np.argmax(counts)]
        else: 
             if depth >= self.maxDepth or n_current_samples < self.minSamplesLeaf * 2:
                return np.mean(current_y)
                
        current_X = self.X[indices]

        prototype_idx_in_indices = np.random.randint(0, n_current_samples)
        prototype_global_idx = indices[prototype_idx_in_indices]
        prototype = self.X[prototype_global_idx]

        similaritiesToPrototype = self.gower_similarity_to_prototype(current_X, prototype)

        threshold = np.median(similaritiesToPrototype)

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
        
        numericaDifferences = 1 - (np.abs(X[:,numMask] - prototype[numMask]) / self.numericFeaturesRanges)
        numericaDifferences = np.sum(numericaDifferences, axis=1)

        categoricalDifferences = X[:, catMask] != prototype[catMask]
        categoricalDifferences = np.sum(categoricalDifferences, axis=1)

        similarities = (numericaDifferences + categoricalDifferences) / X.shape[1]

        return similarities
    
    def compute_numeric_ranges(self):

        X = self.X
        self.numericFeaturesRanges = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            if not self.isCategorical[i]:

                col = X[:, i]
                max = np.nanmax(col)
                min = np.nanmin(col)

                self.numericFeaturesRanges[i] = np.abs(max - min) if max > min else 1
        
        self.numericFeaturesRanges = self.numericFeaturesRanges[~self.isCategorical]

    def compute_categorical_mask(self):
        n= self.X.shape[1]
        self.isCategorical = np.zeros(n, dtype=bool)
        if self.categoricalFeatures is not None:
            self.isCategorical[self.categoricalFeatures] = True 

