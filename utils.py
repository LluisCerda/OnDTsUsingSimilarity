import numpy as np

def _gini_index(self, left_y, right_y):
    def gini(y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    left_size, right_size = len(left_y), len(right_y)
    total_size = left_size + right_size
    
    return (left_size / total_size) * gini(left_y) + (right_size / total_size) * gini(right_y)