from models.STDecisionTreeClassifier import STDecisionTreeClassifier
from models.GiniSimilarityDTClassifier import GiniSimilarityDTClassifier
from models.MeanSimilarityDTClassifier_D4 import MeanSimilarityDTClassifier_D4
from data import load_data

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import utils

import numpy as np

def pipeline(dataset_i, depth=4):

    data_info = load_data(dataset_i)
    data = data_info["data"]

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)    

    info = {
        "dataset": data_info["name"],
        "task": "classification",
        "num_samples": len(data.data),
        "num_features": len(data.feature_names),
        "depth": depth
    }

    STtree = STDecisionTreeClassifier(data_info["categorical"], max_depth=depth)
    #giniSTree = GiniSimilarityDTClassifier(data_info["categorical"], max_depth=depth)
    meanSTree_D4 = MeanSimilarityDTClassifier_D4(data_info["categorical"], max_depth=depth)
    #tree = STDecisionTreeRegression(max_depth=depth)
    
    trees = [meanSTree_D4]
    names = ["MeanSDTD5"]

    for tree_i in range(len(trees)):
        fit_start_time = time.time()
        trees[tree_i].fit(X_train, y_train)
        fit_end_time = time.time()
        prediction_start_time = time.time()
        y_pred = trees[tree_i].predict(X_test)
        prediction_end_time = time.time()

        print(np.sum(y_pred))

        info["Fit time "+names[tree_i]] = fit_end_time - fit_start_time
        info["Prediction time "+names[tree_i]] = prediction_end_time - prediction_start_time
        info["Score "+names[tree_i]] = accuracy_score(y_test, y_pred) if data_info["task"] == "classification" else mean_squared_error(y_test, y_pred)
    

    # visualize_tree_pyvis(tree.tree,  output_file="STDT.html", Sim_tree=False)
    # visualize_tree_pyvis(Stree.tree, output_file="SIMDT.html", Sim_tree=True)

    #if data_info["task"] == "classification":
    #    tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names, class_names=data.target_names)
    #else: 
    #    tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names)

    # st_depth, st_nodes = utils.tree_depth_and_nodes(tree.tree)
    # sim_depth, sim_nodes = utils.tree_depth_and_nodes(Stree.tree)

    # "st_depth": st_depth,
    # "st_nodes": st_nodes,
    # "sim_depth": sim_depth,
    # "sim_nodes": sim_nodes

    return info

if __name__ == "__main__":

    results = []

    for dataset_i in range(1, 6):
        results.append(pipeline(dataset_i, 6))
    
    utils.export_to_excel(results)






    