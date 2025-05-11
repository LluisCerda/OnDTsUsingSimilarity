from models.SimilarityDecisionTree_D13 import SimilarityDecisionTree_D13
from models.SimilarityDecisionTree_D15 import SimilarityDecisionTree_D15
from models.SimilarityDecisionTree_D16 import SimilarityDecisionTree_D16


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from data import load_data

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import utils

import numpy as np

def pipeline(dataset_i, depth=7, par=500000, min_samples=1, splits=2):


    data_info = load_data(dataset_i)
    data = data_info["data"]

    isClassifierTask = data_info["task"] == "classification"

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)    

    SDTree_D15 = SimilarityDecisionTree_D15( isClassifier = isClassifierTask, categoricalFeatures = data_info["categorical"], maxDepth = depth, nJobs = -1, parallelizationThreshold=par, minSamplesLeaf=min_samples)
    SDTree_D16 = SimilarityDecisionTree_D16( isClassifier = isClassifierTask, categoricalFeatures = data_info["categorical"], maxDepth = depth, parallelizationThreshold=par, minSamplesLeaf=min_samples, nChildren=splits)
    #sktree = DecisionTreeClassifier(max_depth=depth)
    #sktree = DecisionTreeRegressor(max_depth=depth)


    trees = [SDTree_D16, ]
    names = ["SDTree_D16"]

    info = {
        "dataset": data_info["name"],
        "task": data_info["task"],
        "num_samples": len(data.data),
        "num_features": len(data.feature_names),
        "depth": depth
    }

    for tree_i in range(len(trees)):

        fit_start_time = time.time()
        trees[tree_i].fit(X_train, y_train)
        fit_end_time = time.time()
        prediction_start_time = time.time()
        y_pred = trees[tree_i].predict(X_test)
        prediction_end_time = time.time()

        info["Fit time "+names[tree_i]] = fit_end_time - fit_start_time
        info["Prediction time "+names[tree_i]] = prediction_end_time - prediction_start_time
        info["Score "+names[tree_i]] = accuracy_score(y_test, y_pred) if data_info["task"] == "classification" else mean_squared_error(y_test, y_pred)

        Rdepth, Rnodes = utils.tree_depth_and_nodes_N(trees[tree_i].tree)
        info["Depth" + names[tree_i]] = Rdepth
        info["Nodes" + names[tree_i]] = Rnodes

    return info

if __name__ == "__main__":

    results = []

    for depth in [6,8,10,12,14, 16]:
        for dataset_i in range(1, 13):
            print("Starting dataset " + str(dataset_i) + " with depth " + str(depth))
            results.append(pipeline(dataset_i, depth=depth, splits=2))
    
    best_results = {}

    for result in results:
        dataset = result["dataset"]
        score = result["Score SDTree_D16"]
        task = result["task"]

        if dataset not in best_results:
            best_results[dataset] = result
        else:
            best_score = best_results[dataset]["Score SDTree_D16"]
            best_task = best_results[dataset]["task"]

            if task == "classification" and score > best_score:
                best_results[dataset] = result
            elif task == "regression" and score < best_score:
                best_results[dataset] = result

    # Convert to list if needed
    best_results_list = list(best_results.values())


    utils.export_to_excel(results, filename="output/results.xlsx")
    utils.export_to_excel(best_results_list, filename="output/best_results.xlsx")

        






    