from STDecisionTreeClassifier import STDecisionTreeClassifier
from STDecisionTreeRegression import STDecisionTreeRegression
from SimilarityDecisionTreeClassifier1 import SimilarityDecisionTreeClassifier1
from SimilarityDecisionTreeClassifier2 import SimilarityDecisionTreeClassifier2
from data import load_data

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

import utils

def pipeline(dataset_i, depth=4):

    data_info = load_data(dataset_i)
    data = data_info["data"]

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)    


    if data_info["task"] == "classification":
        tree = STDecisionTreeClassifier(data_info["categorical"], max_depth=depth)
        Stree = SimilarityDecisionTreeClassifier1(data_info["categorical"], max_depth=depth)
        S2tree = SimilarityDecisionTreeClassifier2(data_info["categorical"], max_depth=depth)
    else: 
        tree = STDecisionTreeRegression(max_depth=depth)
    
    st_fit_start_time = time.time()
    tree.fit(X_train, y_train)
    st_fit_end_time = time.time()
    
    st_predict_start_time = time.time()
    st_y_pred = tree.predict(X_test)
    st_predict_end_time = time.time()

    sim_fit_start_time = time.time()
    Stree.fit(X_train, y_train)
    sim_fit_end_time = time.time()
    
    sim_predict_start_time = time.time()
    sim_y_pred = Stree.predict(X_test)
    sim_predict_end_time = time.time()

    sim2_fit_start_time = time.time()
    S2tree.fit(X_train, y_train)
    sim2_fit_end_time = time.time()
    
    sim2_predict_start_time = time.time()
    sim2_y_pred = S2tree.predict(X_test)
    sim2_predict_end_time = time.time()

    # visualize_tree_pyvis(tree.tree,  output_file="STDT.html", Sim_tree=False)
    # visualize_tree_pyvis(Stree.tree, output_file="SIMDT.html", Sim_tree=True)

    #if data_info["task"] == "classification":
    #    tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names, class_names=data.target_names)
    #else: 
    #    tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names)

    # st_depth, st_nodes = utils.tree_depth_and_nodes(tree.tree)
    # sim_depth, sim_nodes = utils.tree_depth_and_nodes(Stree.tree)

    result = {
        "dataset": data_info["name"],
        "task": "classification",
        "num_samples": len(data.data),
        "num_features": len(data.feature_names),
        "depth": depth,
        "ST_fitting_time": st_fit_end_time-st_fit_start_time,
        "ST_prediction_time": st_predict_end_time-st_predict_start_time,
        "SIM_gini_fitting_time": sim_fit_end_time-sim_fit_start_time,
        "SIM_gini_prediction_time": sim_predict_end_time-sim_predict_start_time,
        "SIM_mean_fitting_time": sim2_fit_end_time-sim2_fit_start_time,
        "SIM_mean_prediction_time": sim2_predict_end_time-sim2_predict_start_time,
        # "st_depth": st_depth,
        # "st_nodes": st_nodes,
        # "sim_depth": sim_depth,
        # "sim_nodes": sim_nodes
    }

    if data_info["task"] == "classification": 
        result["ST_score"] = accuracy_score(y_test, st_y_pred)
        result["SIM_geany_score"] = accuracy_score(y_test, sim_y_pred)
        result["SIM_mean_score"] = accuracy_score(y_test, sim2_y_pred)
    else:
        result["MSE"] = mean_squared_error(y_test, st_y_pred)
        result["MAE"] = mean_absolute_error(y_test, st_y_pred)
        result["R2 Score"] = r2_score(y_test, st_y_pred)
    
    return result

if __name__ == "__main__":

    results = []
    for dataset_i in range(1, 5):
        results.append(pipeline(dataset_i, 8))
    
    utils.export_to_excel(results)






    