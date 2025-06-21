from models.old.MeanSimilarityDTClassifier_D9 import MeanSimilarityDTClassifier_D9
from models.old.SimilarityDecisionTree_D10 import SimilarityDecisionTree_D10
from models.old.SimilarityDecisionTree_D11 import SimilarityDecisionTree_D11
from models.SimilarityDecisionTree_D12 import SimilarityDecisionTree_D12
from models.SimilarityDecisionTree_D14 import SimilarityDecisionTree_D14
from models.SimilarityDecisionTree_D13 import SimilarityDecisionTree_D13
from models.SimilarityDecisionTree_D15 import SimilarityDecisionTree_D15
from models.SimilarityDecisionTree_D16 import SimilarityDecisionTree_D16
from models.SimilarityDecisionTree_D17 import SimilarityDecisionTree_D17

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from data import load_data

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, mean_squared_log_error
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import utils

import numpy as np

def regression_report(y_true, y_pred):
    report = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

    # Try MSLE only if no negative values
    if np.all(y_true >= 0) and np.all(y_pred >= 0):
        report["MSLE"] = mean_squared_log_error(y_true, y_pred)

    return report


def pipeline(dataset_i, depth=7, par=500000, min_samples=1, splits=2):


    data_info = load_data(dataset_i)
    data = data_info["data"]

    isClassifierTask = data_info["task"] == "classification"

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)    

    
    #SDTree_D16 = SimilarityDecisionTree_D16( isClassifier = isClassifierTask, categoricalFeatures = data_info["categorical"], maxDepth = depth, parallelizationThreshold=par, minSamplesLeaf=min_samples, nChildren=splits)
    SDTree_D11 = SimilarityDecisionTree_D11( isClassifier = isClassifierTask, categoricalFeatures = data_info["categorical"], max_depth = depth, par=par)
    SDTree_D17 = SimilarityDecisionTree_D17( isClassifier = isClassifierTask, categoricalFeatures = data_info["categorical"], maxDepth = depth, parallelizationThreshold=par, minSamplesLeaf=min_samples, nChildren=splits)
    sktree = DecisionTreeClassifier(max_depth=depth) if isClassifierTask else DecisionTreeRegressor(max_depth=depth)
    Kneighbours = KNeighborsClassifier() if isClassifierTask else KNeighborsRegressor()
    linearRegression = LinearRegression() 
    logisticRegression = LogisticRegression(max_iter=1000)



    trees = [
        #SDTree_D16, 
        sktree # #, , Kneighbours, linearRegression, logisticRegression
    ]  

    names = [#"SDTree_D16", 
             "SDTree_D17"#, "sklearn_tree", "KNeighbors", "LinearRegression", "LogisticRegression"
             ]

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
        
        if data_info["task"] != "regression" :
            report = classification_report(y_test, y_pred, output_dict=True)

            macro_avg = report["macro avg"]
            info["Accuracy "+names[tree_i]] = report["accuracy"]
            info["Recall "+names[tree_i]] = macro_avg["recall"]
            info["Precision "+names[tree_i]] = macro_avg["precision"]
            info["F1-score "+names[tree_i]] = macro_avg["f1-score"]
        else: 
            report = regression_report(y_test, y_pred)
            info["MAE "+names[tree_i]] = report["MAE"]
            info["MSE "+names[tree_i]] = report["MSE"]
            info["RMSE "+names[tree_i]] = report["RMSE"]
            info["MedAE "+names[tree_i]] = report["MedAE"]
            info["R2 "+names[tree_i]] = report["R2"]

        # Rdepth, Rnodes = utils.tree_depth_and_nodes_N(trees[tree_i].tree)
        # info["Depth" + names[tree_i]] = Rdepth
        # info["Nodes" + names[tree_i]] = Rnodes
        info["Min samples leaf " + names[tree_i]] = min_samples
        info["Num children " + names[tree_i]] = splits

    return info

if __name__ == "__main__":

    results = []

    for min_samples in [1,2,3,4,5,6]: 
        for depth in [4, 6, 8, 10, 12, 15, 20]:
            for dataset_i in [10]: 
                print("Starting dataset " + str(dataset_i) + " with depth " + str(depth))
                results.append(pipeline(dataset_i, depth=depth, splits=2, min_samples=min_samples))

    best_results = {}

    for result in results:
        dataset = result["dataset"]
        score = result["Score SDTree_D17"]
        task = result["task"]

        if dataset not in best_results:
            best_results[dataset] = result
        else:
            best_score = best_results[dataset]["Score SDTree_D17"]
            best_task = best_results[dataset]["task"]

            if task == "classification" and score > best_score:
                best_results[dataset] = result
            elif task == "regression" and score < best_score:
                best_results[dataset] = result


    best_results_list = list(best_results.values())

    utils.export_to_excel(results, filename="output/results.xlsx")
    utils.export_to_excel(best_results_list, filename="output/best_results.xlsx")

        






    