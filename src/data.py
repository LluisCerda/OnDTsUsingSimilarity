from sklearn.datasets import load_wine, load_iris, fetch_california_housing, fetch_openml, load_diabetes, load_breast_cancer, load_digits

import numpy as np
import pandas as pd
import utils


def data_preprocessing(data, categoricalFeatures):

    _, y_encoded = np.unique(data.target, return_inverse=True)
    data.target = y_encoded

    for i in range(len(data.data[0])):
        if i in categoricalFeatures:
            mask = pd.isna(data.data[:, i]) 
            data.data[:, i][mask] = "NaN"
            data.data[:, i]  = np.unique(data.data[:, i], return_inverse=True)[1]
        else:
            col = data.data[:, i].astype(float)
    
            mean_value = np.nanmean(col)  
            
            col[np.isnan(col)] = mean_value  
            data.data[:, i] = col
    
    return data

def load_data(num):

    #CLASSIFICATION ONLY NUMERICAL
    if num == 1: 
        categoricalFeatures = None
        return {"data": load_iris(), "categorical": categoricalFeatures, "name": "Iris", "filename": "iris.html", "task": "classification"}
    
    if num == 2: 
        categoricalFeatures = None
        return {"data": load_wine(), "categorical": categoricalFeatures, "name": "Wine", "filename": "wine.html", "task": "classification"}
    
    if num == 3: 
        categoricalFeatures = None
        return {"data": load_breast_cancer(), "categorical": categoricalFeatures, "name": "Breast Cancer", "filename": "breast_cancer.html", "task": "classification"}
    
    if num == 4: 
        categoricalFeatures = None
        return {"data": load_digits(), "categorical": categoricalFeatures, "name": "Digits", "filename": "digits.html", "task": "classification"}
    
    if num == 5: 
        categoricalFeatures = None
        data = fetch_openml(name="banknote-authentication", as_frame=False, version=1, parser="auto")
        data.target = data.target.astype(int)
        return {"data": data, "categorical": categoricalFeatures, "name": "BankNote Authentication", "filename": "banknote_authentication.html", "task": "classification"}
    
    if num == 6: 
        categoricalFeatures = None
        #13910 rows, 129 columns
        data = fetch_openml(name="gas-drift", as_frame=False, version=1, parser="auto")
        data.target = data.target.astype(int)
        return {"data": data, "categorical": categoricalFeatures, "name": "Gas Drift", "filename": "gas_drift.html", "task": "classification"}

    if num == 7: 
        #58000 rows, 9 columns
        categoricalFeatures = None
        data = fetch_openml(name="shuttle", as_frame=False, version=1, parser="auto")
        data.target = data.target.astype(int)
        return {"data": data, "categorical": categoricalFeatures, "name": "Shuttle", "filename": "shuttle.html", "task": "classification"}
    

    #CLASSIFICATION NUMERICAL AND CATEGORICAL
    if num == 8: 
        categoricalFeatures = [1, 3, 5, 6, 7, 8, 9, 13]
        data = data_preprocessing(fetch_openml("adult", version=2, as_frame=False, parser="auto"), categoricalFeatures)
        return {"data": data, "categorical": categoricalFeatures, "name": "Adult", "filename": "adult.html", "task": "classification"}
    
    if num == 9: 
        categoricalFeatures = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]

        data = data_preprocessing( fetch_openml(data_id=269, as_frame=False, parser="auto"), categoricalFeatures)
        
        return {"data": data, "categorical": categoricalFeatures, "name": "Adult", "filename": "adult.html", "task": "classification"}
    

    #REGRESSION ONLY NUMERICAL
    if num == 10: 
        data, isCategorical = data_preprocessing(load_diabetes())
        return {"data": data, "categorical": isCategorical, "name": "Diabetes Progression", "filename": "diabetes_progression.html", "task": "regression"}
    
    if num == 11: 
        data, isCategorical = data_preprocessing(fetch_openml(name="boston", as_frame=False, version=1))
        return {"data": data, "categorical": isCategorical, "name": "Boston Housing", "filename": "boston_housing.html", "task": "regression"}
    
    if num == 12: 

        data, isCategorical = data_preprocessing(fetch_california_housing())
        return {"data": data, "categorical": isCategorical, "name": "California Housing", "filename": "california_housing.html", "task": "regression"}
    
    
