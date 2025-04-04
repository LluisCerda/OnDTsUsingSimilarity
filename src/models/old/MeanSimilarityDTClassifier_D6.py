import numpy as np
import time

'''
This class is a decision tree classifier that uses the Gower distance to compute the similarity between samples.
Treats the mean as threshold.


D5 with random prototype assignation.
'''

class MeanSimilarityDTClassifier_D6:
    
    def __init__(self, categoricalFeatures, max_depth=4):
        self.max_depth = max_depth
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = None
        self.tree = None
        self.numericFeaturesRanges = None
    
    def fit(self, X, y):

        #Compute categorical mask
        self.isCategorical = np.zeros(X.shape[1], dtype=bool)
        if self.categoricalFeatures is not None:
            self.isCategorical[self.categoricalFeatures] = True 

        #Compute range of each feature
        self.numericFeaturesRanges = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            if not self.isCategorical[i]:
                max_f = np.nanmax(X[:, i])
                min_f = np.nanmin(X[:, i])
                self.numericFeaturesRanges[i] = max_f - min_f if max_f > min_f else 1
        self.numericFeaturesRanges = self.numericFeaturesRanges[~self.isCategorical]

        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        
        prototype_idx = np.random.randint(0, X.shape[0])
        prototype = X[prototype_idx]

        similaritiesToPrototype = self.gower_similarity_to_prototype(X, prototype) 

        threshold = np.mean(similaritiesToPrototype) #mean?
        
        left_indices = similaritiesToPrototype <= threshold
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.bincount(y).argmax()
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {"prototype": prototype, "threshold": threshold, "left": left_subtree, "right": right_subtree}
    
    def gower_similarity_to_prototype(self, X, prototype):

        numericaDifferences = 1 - (np.abs(X[:,~self.isCategorical] - prototype[~self.isCategorical]) / self.numericFeaturesRanges)
        numericaDifferences = np.sum(numericaDifferences, axis=1)

        categoricalDifferences = np.count_nonzero(X[:,self.isCategorical] != prototype[self.isCategorical], axis=1)

        similarities = (numericaDifferences + categoricalDifferences) / X.shape[1]

        return similarities


    def two_samples_gower_similarity(self, x, y):
        
        numericDifferences = 1 - (np.abs(x[~self.isCategorical] - y[~self.isCategorical]) / self.numericFeaturesRanges)
        numericDifferences = np.sum(numericDifferences)
        
        categoricalDifferences = np.count_nonzero(x[self.isCategorical] != y[self.isCategorical])

        similarity = (numericDifferences + categoricalDifferences) / x.shape[0]

        return similarity

    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):

        if not isinstance(node, dict):
            return node

        similarity = self.two_samples_gower_similarity(x, node["prototype"])  

        if similarity <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])
