import numpy as np
from joblib import Parallel, delayed

from myGower import gower_similarity_to_prototype_numba

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.


D10 optimized
'''

class SimilarityDecisionTree_D11:
    
    def __init__(self, isClassifier = True, categoricalFeatures = None, max_depth = 4, n_jobs = -1, par = 500000):
        self.max_depth = max_depth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
        self.numericFeaturesRanges = None
        self.n_jobs = n_jobs
        self.isClassifier = isClassifier
        self.par = par
    
    def fit(self, X, y):

        self.compute_categorical_mask(X.shape[1])
        self.compute_numeric_ranges(X)

        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        
        if self.isClassifier:
            if depth >= self.max_depth or len(np.unique(y)) == 1:
                return np.bincount(y).argmax()
        else:
            if depth >= self.max_depth:
                return np.mean(y)
        
        prototype_idx = np.random.randint(0, X.shape[0])
        prototype = X[prototype_idx]

        similaritiesToPrototype = gower_similarity_to_prototype_numba(X, prototype, self.isCategorical, self.numericFeaturesRanges) 

        threshold = np.median(similaritiesToPrototype) 
        
        leftMask = similaritiesToPrototype <= threshold
        rightMask = ~leftMask
        
        if np.sum(leftMask) <= 2 or np.sum(rightMask) <= 2:
            if self.isClassifier:
                return np.bincount(y).argmax()
            else:
                return np.mean(y)
        
        if X.shape[0] * X.shape[1] >= self.par:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._build_tree)(X[mask], y[mask], depth + 1)
                for mask in [leftMask, rightMask]
            )
        else:
            results = [
                self._build_tree(X[leftMask], y[leftMask], depth + 1),
                self._build_tree(X[rightMask], y[rightMask], depth + 1)
            ]
        
        return {"prototype": prototype, "threshold": threshold, "left": results[0], "right": results[1]}
    
    def predict(self, X):

        leaf_assignments = self._traverse_tree(X, self.tree, np.arange(X.shape[0]))

        y_pred = np.empty(X.shape[0], dtype=int)
        for label, indices in leaf_assignments.items():
            y_pred[indices] = label 

        return y_pred
    
    def _traverse_tree(self, X, node, indices):
        
        if not isinstance(node, dict):
            return {node: indices} 

        similaritiesToPrototype = gower_similarity_to_prototype_numba(X, node["prototype"], self.isCategorical, self.numericFeaturesRanges)
        
        leftMask = similaritiesToPrototype <= node["threshold"]
        rightMask = ~leftMask

        leftIndices = indices[leftMask]
        rightIndices = indices[rightMask]

        leftResult = self._traverse_tree(X[leftMask], node["left"], leftIndices)
        rightResult = self._traverse_tree(X[rightMask], node["right"], rightIndices)

        #Dictionary merge
        for key in rightResult:
            leftResult[key] = np.concatenate((leftResult[key], rightResult[key])) if key in leftResult else rightResult[key]

        return leftResult 
    
    def gower_similarity_to_prototype(self, X, prototype):

        numMask = ~self.isCategorical
        catMask = self.isCategorical
        numericalRanges = self.numericFeaturesRanges

        numericaDifferences = 1 - (np.abs( X[:,numMask] - prototype[numMask] ) / numericalRanges)
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


