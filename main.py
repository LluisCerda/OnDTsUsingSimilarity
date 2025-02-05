from STDecisionTreeClassifier import STDecisionTreeClassifier
from STDecisionTreeRegression import STDecisionTreeRegression
from data import load_data

import pandas as pd
import time

from pyvis.network import Network

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

def visualize_dataframe(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(df)

def visualize_tree_pyvis(tree, output_file="tree.html", feature_names=None, class_names=None):

    net = Network(notebook=True, directed=True)
    node_counter = {"count": 0}  # To ensure unique leaf nodes

    def add_edges(node, parent=None, is_root=False, edge_label=""):
        if isinstance(node, dict):

            if feature_names is not None:
                node_label = f"{feature_names[node['feature']]} ≤ {node['threshold']}"
            else: node_label = f"Feature {node['feature']} ≤ {node['threshold']}"

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

def export_to_excel(data, filename="output/results.xlsx"):

    if not isinstance(data, list):
        data = [data]  

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)

    print(f"Results exported to {filename}")

def pipeline(dataset_i, depth=4):

    data_info = load_data(dataset_i)
    data = data_info["data"]

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)    

    if data_info["task"] == "classification":
        tree = STDecisionTreeClassifier(data_info["categorical"], max_depth=depth)
    else: 
        tree = STDecisionTreeRegression(max_depth=depth)
    
    fit_start_time = time.time()
    tree.fit(X_train, y_train)
    fit_end_time = time.time()
    
    predict_start_time = time.time()
    y_pred = tree.predict(X_test)
    predict_end_time = time.time()

    #if data_info["task"] == "classification":
    #    tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names, class_names=data.target_names)
    #else: 
    #    tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names)

    result = {
        "name": data_info["name"],
        "task": "classification",
        "num_samples": len(data.data),
        "num_features": len(data.feature_names),
        "depth": depth,
        "fitting_elapsed_time": fit_end_time-fit_start_time,
        "prediction_elapsed_time": predict_end_time-predict_start_time,
    }

    if data_info["task"] == "classification": 
        result["score"] = accuracy_score(y_test, y_pred)
    else:
        result["MSE"] = mean_squared_error(y_test, y_pred)
        result["MAE"] = mean_absolute_error(y_test, y_pred)
        result["R2 Score"] = r2_score(y_test, y_pred)
    
    return result

if __name__ == "__main__":

    result = []
    for dataset_i in range(1,12):

        print("STARTING DATASET: " + str(dataset_i) + "/11")
        result.append(pipeline(dataset_i))

    export_to_excel(result)


    