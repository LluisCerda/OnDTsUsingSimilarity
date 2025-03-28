from sklearn.datasets import load_wine, load_iris, fetch_california_housing, fetch_openml, load_diabetes, load_breast_cancer, load_digits

import numpy as np


def _is_numerical(column):
    return np.issubdtype(column.dtype, np.number)

def data_preprocessing(data):

    _, y_encoded = np.unique(data.target, return_inverse=True)
    data.target = y_encoded

    isCategorical = []

    for i in range(len(data.feature_names)):
        if not _is_numerical(data.data[:, i]):
            isCategorical.append(i)  
            _, encoded_col = np.unique(data.data[:, i].astype(str), return_inverse=True)
            data.data[:, i] = encoded_col  

    if len(isCategorical) == 0:
        isCategorical = None

    return data, isCategorical

def load_data(num):

    #CLASSIFICATION ONLY NUMERICAL
    if num == 1: 
        data, isCategorical = data_preprocessing(load_iris())
        return {"data": data, "categorical": isCategorical, "name": "Iris", "filename": "iris.html", "task": "classification"}
    
    if num == 2: 
        data, isCategorical = data_preprocessing(load_wine())
        return {"data": data, "categorical": isCategorical, "name": "Wine", "filename": "wine.html", "task": "classification"}
    
    if num == 3: 
        data, isCategorical = data_preprocessing(load_breast_cancer())
        return {"data": data, "categorical": isCategorical, "name": "Breast Cancer", "filename": "breast_cancer.html", "task": "classification"}
    
    if num == 4: 
        data, isCategorical = data_preprocessing(load_digits())
        return {"data": data, "categorical": isCategorical, "name": "Digits", "filename": "digits.html", "task": "classification"}
    

    #CLASSIFICATION NUMERICAL AND CATEGORICAL
    if num == 5: 
        data, isCategorical = data_preprocessing(fetch_openml("adult", version=2, as_frame=False))
        return {"data": data, "categorical": isCategorical, "name": "Adult", "filename": "adult.html", "task": "classification"}
    

    #REGRESSION ONLY NUMERICAL
    if num == 6: 
        data, isCategorical = data_preprocessing(load_diabetes())
        return {"data": data, "categorical": isCategorical, "name": "Diabetes Progression", "filename": "diabetes_progression.html", "task": "regression"}
    
    if num == 7: 
        data, isCategorical = data_preprocessing(fetch_openml(name="boston", as_frame=False, version=1))
        return {"data": data, "categorical": isCategorical, "name": "Boston Housing", "filename": "boston_housing.html", "task": "regression"}
    
    #SLOW
    if num == 8: 
        data, isCategorical = data_preprocessing(fetch_california_housing())
        return {"data": data, "categorical": isCategorical, "name": "California Housing", "filename": "california_housing.html", "task": "regression"}
    

    #LIMPIAR
    if num == 9: 
        data, isCategorical = data_preprocessing(fetch_openml(name="banknote-authentication", as_frame=False, version=1))
        return {"data": data, "categorical": isCategorical, "name": "BankNote Authentication", "filename": "banknote_authentication.html", "task": "classification"}
    
    if num == 10: 
        data, isCategorical = data_preprocessing(fetch_openml(name="shuttle", as_frame=False, version=1))
        return {"data": data, "categorical": isCategorical, "name": "Shuttle", "filename": "shuttle.html", "task": "classification"}
    
    if num == 11: 
        data, isCategorical = data_preprocessing(fetch_openml(name="gas-drift", as_frame=False, version=1))
        return {"data": data, "categorical": isCategorical, "name": "Gas Drift", "filename": "gas_drift.html", "task": "classification"}
    
