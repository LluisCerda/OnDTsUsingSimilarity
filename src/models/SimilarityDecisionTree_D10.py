import numpy as np
from joblib import Parallel, delayed

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.


D7 with parallelization.
'''

class SimilarityDecisionTree_D10:
    
    def __init__(self, isClassifier = True, categoricalFeatures = None, max_depth = 4, n_jobs = -1):
        self.max_depth = max_depth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
        self.numericFeaturesRanges = None
        self.n_jobs = n_jobs
        self.isClassifier = isClassifier
    
    def fit(self, X, y):

        self.isCategorical = np.zeros(X.shape[1], dtype=bool)
        if self.categoricalFeatures is not None:
            self.isCategorical[self.categoricalFeatures] = True 

        self.numericFeaturesRanges = np.zeros(X.shape[1])
        self.numericFeaturesMax = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            if not self.isCategorical[i]:

                col = X[:, i]
                max_f = np.nanmax(col)
                min_f = np.nanmin(col)

                self.numericFeaturesMax[i] = max_f if max_f > min_f else 1
                self.numericFeaturesRanges[i] = np.abs(max_f - min_f) if max_f > min_f else 1
        
        self.numericFeaturesRanges = self.numericFeaturesRanges[~self.isCategorical]
        self.numericFeaturesMax = self.numericFeaturesMax[~self.isCategorical]

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

        similaritiesToPrototype = self.gower_similarity_to_prototype(X, prototype) 

        threshold = np.median(similaritiesToPrototype) 
        
        leftMask = similaritiesToPrototype <= threshold
        rightMask = ~leftMask
        
        if np.sum(leftMask) <= 2 or np.sum(rightMask) <= 2:
            if self.isClassifier:
                return np.bincount(y).argmax()
            else:
                return np.mean(y)
        
        if X.shape[0] * X.shape[1] >= 100000:
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
    
    def gower_similarity_to_prototype(self, X, prototype):

        numericaDifferences = 1 - (np.abs( X[:,~self.isCategorical] - prototype[~self.isCategorical] )/ self.numericFeaturesRanges)
        numericaDifferences = np.sum(numericaDifferences, axis=1)

        categoricalDifferences = np.count_nonzero(X[:,self.isCategorical] != prototype[self.isCategorical], axis=1)

        similarities = (numericaDifferences + categoricalDifferences) / X.shape[1]

        return similarities

    def _traverse_tree(self, X, node, indices):
        
        if not isinstance(node, dict):
            return {node: indices} 

        similaritiesToPrototype = self.gower_similarity_to_prototype(X, node["prototype"])
        
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


