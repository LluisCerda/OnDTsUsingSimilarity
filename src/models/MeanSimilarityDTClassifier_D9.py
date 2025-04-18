import numpy as np
from joblib import Parallel, delayed

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.

D7+D8 with jaccard index for binary features  
'''

class MeanSimilarityDTClassifier_D9:
    
    def __init__(self, categoricalFeatures, max_depth=4, n_jobs=-1):
        self.max_depth = max_depth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.isBinary = None
        self.isNumerical = None
        self.tree = None
        self.numericFeaturesRanges = None
        self.n_jobs = n_jobs
        self.nCategorical = None
        self.nBinary = None
        self.nNumerical = None

    def fit(self, X, y):

        self.features_to_mask(X)

        self.compute_numeric_ranges(X)

        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        
        prototype_idx = np.random.randint(0, X.shape[0])
        prototype = X[prototype_idx]

        similaritiesToPrototype = self.gower_similarity_to_prototype(X, prototype) 

        threshold = np.mean(similaritiesToPrototype) 
        
        leftMask = similaritiesToPrototype <= threshold
        rightMask = ~leftMask
        
        if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
            return np.bincount(y).argmax()
        
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
    
    def gower_similarity_to_prototype(self, X, prototype):

        total_features = X.shape[1]

        w_num = self.nNumerical / total_features
        w_cat = self.nCategorical / total_features
        w_bin = self.nBinary / total_features

        if self.nNumerical > 0:
            numericalDifferences = 1 - (np.abs(X[:,self.isNumerical] - prototype[self.isNumerical]) / self.numericFeaturesRanges)
            numericalDifferences = np.mean(numericalDifferences, axis=1)
        else: numericalDifferences = np.zeros(X.shape[0])

        if self.nCategorical > 0:
            categoricalDifferences = np.count_nonzero(X[:,self.isCategorical] != prototype[self.isCategorical], axis=1).astype(float)
            categoricalDifferences /= self.nCategorical
        else: categoricalDifferences = np.zeros(X.shape[0])

        if self.nBinary > 0:
            # binaryDifferences = np.count_nonzero(X[:,self.isBinary] != prototype[self.isBinary], axis=1).astype(float)
            # binaryDifferences /= self.nBinary

            intersection = np.logical_and(X[:, self.isBinary], prototype[self.isBinary])
            union = np.logical_or(X[:, self.isBinary], prototype[self.isBinary])
            
            intersection_count = np.sum(intersection, axis=1)
            union_count = np.sum(union, axis=1)

            union_count_safe = np.where(union_count == 0, 1, union_count)

            binaryDifferences = (union_count - intersection_count) / union_count_safe

        else: binaryDifferences = np.zeros(X.shape[0])



        similarities = numericalDifferences * w_num + categoricalDifferences * w_cat + binaryDifferences * w_bin

        return similarities
    
    def features_to_mask(self, X):
        
        nFeatures = X.shape[1]

        self.isCategorical = np.zeros(nFeatures, dtype=bool)
        self.isBinary = np.zeros(nFeatures, dtype=bool)

        if self.categoricalFeatures is not None:
            for idx in self.categoricalFeatures:
                unique_vals = np.unique(X[:, idx])
                if len(unique_vals) == 2:
                    self.isBinary[idx] = True
                else:
                    self.isCategorical[idx] = True  

        self.isNumerical = ~(self.isCategorical | self.isBinary)

        self.nCategorical = np.sum(self.isCategorical)
        self.nBinary = np.sum(self.isBinary)
        self.nNumerical = np.sum(self.isNumerical)

    def compute_numeric_ranges(self, X):
        self.numericFeaturesRanges = np.zeros(X.shape[1], dtype=float)
        for i in range(X.shape[1]):
            if self.isNumerical[i]:
                max_f = np.max(X[:, i])
                min_f = np.min(X[:, i])
                self.numericFeaturesRanges[i] = np.abs(max_f - min_f) if max_f > min_f else 1
        self.numericFeaturesRanges = self.numericFeaturesRanges[self.isNumerical]

    def predict(self, X):

        leaf_assignments = self._traverse_tree(X, self.tree, np.arange(X.shape[0]))

        y_pred = np.empty(X.shape[0], dtype=int)
        for label, indices in leaf_assignments.items():
            y_pred[indices] = label 

        return y_pred

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
