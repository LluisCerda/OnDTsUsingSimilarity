import numpy as np
from pyvis.network import Network

class StDecisionTree:
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):

        unique_labels = np.unique(y)

        if len(unique_labels) == 1:
            return unique_labels[0]  # Pure leaf node
        if self.max_depth is not None and depth >= self.max_depth:
            return np.bincount(y).argmax()  # Majority class
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}
    
    def _best_split(self, X, y):
        _, num_features = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                gini = self._gini_index(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _gini_index(self, left_y, right_y):
        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions ** 2)
        
        left_size, right_size = len(left_y), len(right_y)
        total_size = left_size + right_size
    
        return (left_size / total_size) * gini(left_y) + (right_size / total_size) * gini(right_y)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
        
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])


    def visualize_tree_pyvis(self, output_file="tree.html"):

        net = Network(notebook=True, directed=True)


        node_counter = {"count": 0}  # To ensure unique leaf nodes

        def add_edges(node, parent=None, is_root=False, edge_label=""):
            if isinstance(node, dict):
                node_label = f"Feature {node['feature']} ≤ {node['threshold']}"

                # Root node
                if is_root:
                    net.add_node(node_label, label=node_label, color="red", shape="box")
                else:
                    net.add_node(node_label, label=node_label, color="lightblue")

                if parent:
                    net.add_edge(parent, node_label, label=edge_label)

                add_edges(node["left"], node_label, edge_label="Yes")
                add_edges(node["right"], node_label, edge_label="No")
            else:
                node_counter["count"] += 1
                leaf_label = f"Class {node} - {node_counter['count']}"
                net.add_node(leaf_label, label=f"Class {node}", color="lightgreen")
                if parent:
                    net.add_edge(parent, leaf_label, label=edge_label) 


        # Start with the root node
        add_edges(self.tree, is_root=True)
        net.show(output_file)

