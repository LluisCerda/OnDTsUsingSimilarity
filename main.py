from StDecisionTree import StDecisionTree

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(num):
    if num==1:
        from sklearn.datasets import load_iris
        data = {"data": load_iris(), "name": "Iris Classification", "filename": "iris_classification.html"}
    elif num==2:
        from sklearn.datasets import fetch_california_housing
        data = {"data": fetch_california_housing(), "name": "California Housing", "filename": "california_housing.html"}
    elif num==3:
        from sklearn.datasets import fetch_openml
        data = {"data": fetch_openml(name="boston", as_frame=False), "name": "Boston Housing", "filename": "boston_housing.html"}
        
    elif num==4:
        from sklearn.datasets import load_diabetes
        data = {"data": load_diabetes(), "name": "Diabetes Progression", "filename": "diabetes_progression.html"}
    else:
        from sklearn.datasets import load_wine
        data = {"data": load_wine(), "name": "Wine Classification", "filename": "wine_classification.html"}
    
    return data

def visualize_dataframe(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(df)

def pipeline(num):

    data_info = load_data(num)

    X_train, X_test, y_train, y_test = train_test_split(data_info["data"].data, data_info["data"].target, test_size=0.2, random_state=42)    

    tree = StDecisionTree()
    tree.fit(X_train, y_train)
    
    y_pred = tree.predict(X_test)

    tree.visualize_tree_pyvis(output_file="output/"+data_info["filename"])

    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":

    print(pipeline(1))

    #pipeline(2)

    #pipeline(3)

    #pipeline(4)

    print(pipeline(5))
    