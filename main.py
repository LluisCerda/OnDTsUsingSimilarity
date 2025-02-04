from StDecisionTree import StDecisionTree

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

from sklearn.datasets import load_wine, load_iris, fetch_california_housing, fetch_openml, load_diabetes

def load_data(num):
    data_examples = {
        1: {"data": load_iris(), "name": "Iris Classification", "filename": "iris_classification.html", "task": "classification"}, 
        2: {"data": fetch_california_housing(), "name": "California Housing", "filename": "california_housing.html", "task": "regression"},
        3: {"data": fetch_openml(name="boston", as_frame=False), "name": "Boston Housing", "filename": "boston_housing.html", "task": "regression"},
        4: {"data": load_diabetes(), "name": "Diabetes Progression", "filename": "diabetes_progression.html", "task": "regression"},
        5: {"data": load_wine(), "name": "Wine Classification", "filename": "wine_classification.html", "task": "classification"}
    }
    return data_examples[num]

def visualize_dataframe(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(df)

def pipeline(num):

    data_info = load_data(num)
    data = data_info["data"]

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)    

    tree = StDecisionTree(max_depth=10, task=data_info["task"])
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_test)

    if data_info["task"] == "classification":
        tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names, class_names=data.target_names)
    else: tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"], feature_names=data.feature_names)

    if data_info["task"] == "classification":
        return accuracy_score(y_test, y_pred)
    else:
        return {
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        }


if __name__ == "__main__":

    #print(pipeline(1))

    print(pipeline(2))

    #print(pipeline(3))

    #print(pipeline(4))

    #print(pipeline(5))
    