from models.old.MeanSimilarityDTClassifier_D8 import MeanSimilarityDTClassifier_D8
from models.SimilarityDecisionTree_D10 import SimilarityDecisionTree_D10
from models.SimilarityDecisionTree_D11 import SimilarityDecisionTree_D11


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from data import load_data

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import utils

def pipeline(dataset_i, depth=4, par=100000):

    data_info = load_data(dataset_i)
    data = data_info["data"]
    isClassifierTask = data_info["task"] == "classification"

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)    

    info = {
        "dataset": data_info["name"],
        "task": data_info["task"],
        "num_samples": len(data.data),
        "num_features": len(data.feature_names),
        "depth": depth
    }

    meanSTree = MeanSimilarityDTClassifier_D8(data_info["categorical"], max_depth=depth)
    SDTree_D10 = SimilarityDecisionTree_D10( isClassifier = isClassifierTask, categoricalFeatures = data_info["categorical"], max_depth = depth, n_jobs = -1)
    SDTree_D11 = SimilarityDecisionTree_D11( isClassifier = isClassifierTask, categoricalFeatures = data_info["categorical"], max_depth = depth, n_jobs = -1)
    
    #sktree = DecisionTreeClassifier(max_depth=depth)
    sktree = DecisionTreeRegressor(max_depth=depth)


    trees = [ SDTree_D11]
    names = ["SDTree_D11"]

    for tree_i in range(len(trees)):

        fit_start_time = time.time()
        trees[tree_i].fit(X_train, y_train, par)
        fit_end_time = time.time()
        prediction_start_time = time.time()
        y_pred = trees[tree_i].predict(X_test)
        prediction_end_time = time.time()

        info["Fit time "+names[tree_i]] = fit_end_time - fit_start_time
        info["Prediction time "+names[tree_i]] = prediction_end_time - prediction_start_time
        info["Score "+names[tree_i]] = accuracy_score(y_test, y_pred) if data_info["task"] == "classification" else mean_squared_error(y_test, y_pred)
    
    #utils.visualize_tree_pyvis(meanSTree_D9.tree, output_file="output/"+data_info["filename"], Sim_tree=True)
    
    # st_depth, st_nodes = utils.tree_depth_and_nodes(tree.tree)
    # sim_depth, sim_nodes = utils.tree_depth_and_nodes(Stree.tree)

    # "st_depth": st_depth,
    # "st_nodes": st_nodes,
    # "sim_depth": sim_depth,
    # "sim_nodes": sim_nodes

    return info

if __name__ == "__main__":

    results = []

    for dataset_i in range(1,13):
        for par in range(50000, 550000, 50000):
            print("Starting dataset " + str(dataset_i) + " with depth " + str(10))
            results.append(pipeline(dataset_i, 10, par))
    
    utils.export_to_excel(results)

        






    