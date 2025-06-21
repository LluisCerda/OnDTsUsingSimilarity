from pyvis.network import Network
import pandas as pd

def visualize_tree_pyvis(tree, output_file="tree.html", feature_names=None, class_names=None, Sim_tree=False):

    net = Network(notebook=True, directed=True)
    node_counter = {"count": 0}  # To ensure unique leaf nodes

    def add_edges(node, parent=None, is_root=False, edge_label=""):
        if isinstance(node, dict):

            if feature_names is not None:
                node_label = f"{feature_names[node['feature']]} ≤ {node['threshold']}"
            elif not Sim_tree: node_label = f"Feature {node['feature']} ≤ {node['threshold']}"
            else: node_label = f"Prototype Sim ≤ {node['threshold']}"

            # Root node
            if is_root:
                net.add_node(node_label, label=node_label, color="orange", shape="box")
            else:
                net.add_node(node_label, label=node_label, color="lightblue", shape="box")

            if parent:
                net.add_edge(parent, node_label, label=edge_label)

            add_edges(node["left"], node_label, edge_label="Yes")
            add_edges(node["right"], node_label, edge_label="No")
        else:
            node_counter["count"] += 1
            leaf_label = f"Class {node} - {node_counter['count']}"

            if class_names is not None:
                net.add_node(leaf_label, label=f"Class {class_names[node]}", color="lightgreen", shape="box")
            else: net.add_node(leaf_label, label=f"Class {node}", color="lightgreen", shape="box")

            if parent:
                net.add_edge(parent, leaf_label, label=edge_label) 

    # Start with the root node
    add_edges(tree, is_root=True)
    net.show(output_file)

def visualize_dataframe(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    # print(df.iloc[:, 5:15])
    print(df)

def export_to_excel(data, filename="output/results.xlsx"):

    if not isinstance(data, list):
        data = [data]  

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)

    print(f"Results exported to {filename}")

def tree_depth_and_nodes_2(tree):
    if not isinstance(tree, dict): 
        return 1, 1 

    left_depth, left_nodes = tree_depth_and_nodes_2(tree["left"])
    right_depth, right_nodes = tree_depth_and_nodes_2(tree["right"])

    total_depth = 1 + max(left_depth, right_depth) 
    total_nodes = 1 + left_nodes + right_nodes  

    return total_depth, total_nodes

def tree_depth_and_nodes_N(tree):
    if not isinstance(tree, dict) or "children" not in tree or not tree["children"]:
        return 1, 1  

    max_child_depth = 0
    total_child_nodes = 0

    for child in tree["children"]:
        child_depth, child_nodes = tree_depth_and_nodes_N(child)
        max_child_depth = max(max_child_depth, child_depth)
        total_child_nodes += child_nodes

    total_depth = 1 + max_child_depth
    total_nodes = 1 + total_child_nodes 

    return total_depth, total_nodes


def get_categorical_indices(isCategorical):
    return [i for i, is_cat in enumerate(isCategorical) if is_cat]


