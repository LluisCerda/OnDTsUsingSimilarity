from sklearn.datasets import load_wine, load_iris, fetch_california_housing, fetch_openml, load_diabetes, load_breast_cancer, load_digits

import numpy as np
import pandas as pd
import utils


def data_preprocessing(data, categoricalFeatures):

    _, y_encoded = np.unique(data.target, return_inverse=True)
    data.target = y_encoded

    for i in range(len(data.data[0])):
        if categoricalFeatures is not None and i in categoricalFeatures:
            mask = pd.isna(data.data[:, i]) 
            data.data[:, i][mask] = "NaN"
            data.data[:, i]  = np.unique(data.data[:, i], return_inverse=True)[1]
            data.data[:, i] = data.data[:, i].astype(int)
        else:
            col = data.data[:, i].astype(float)
    
            mean_value = np.nanmean(col)  
            
            col[np.isnan(col)] = mean_value  
            data.data[:, i] = col

            data.data[:, i] = data.data[:, i].astype(float)
    
    return data

def load_data(num):

    categoricalFeatures = None
    #CLASSIFICATION ONLY NUMERICAL
    if num == 1: 
        return {"data": load_iris(), "categorical": categoricalFeatures, "name": "Iris", "filename": "iris.html", "task": "classification"}
    
    if num == 2: 
        return {"data": load_wine(), "categorical": categoricalFeatures, "name": "Wine", "filename": "wine.html", "task": "classification"}
    
    if num == 3: 
        return {"data": load_breast_cancer(), "categorical": categoricalFeatures, "name": "Breast Cancer", "filename": "breast_cancer.html", "task": "classification"}
    
    if num == 4: 
        return {"data": load_digits(), "categorical": categoricalFeatures, "name": "Digits", "filename": "digits.html", "task": "classification"}
    
    if num == 5: 
        data = fetch_openml(name="banknote-authentication", as_frame=False, version=1, parser="auto")
        data.target = data.target.astype(int)
        return {"data": data, "categorical": categoricalFeatures, "name": "BankNote Authentication", "filename": "banknote_authentication.html", "task": "classification"}
    
    if num == 6: 
        #13910 rows, 129 columns
        data = fetch_openml(name="gas-drift", as_frame=False, version=1, parser="auto")
        data.target = data.target.astype(int)
        return {"data": data, "categorical": categoricalFeatures, "name": "Gas Drift", "filename": "gas_drift.html", "task": "classification"}

    if num == 7: 
        #58000 rows, 9 columns
        data = fetch_openml(name="shuttle", as_frame=False, version=1, parser="auto")
        data.target = data.target.astype(int)
        return {"data": data, "categorical": categoricalFeatures, "name": "Shuttle", "filename": "shuttle.html", "task": "classification"}
    

    #CLASSIFICATION NUMERICAL AND CATEGORICAL
    if num == 8: #Has missing values
        categoricalFeatures = [1, 3, 5, 6, 7, 8, 9, 13]
        data = data_preprocessing(fetch_openml("adult", version=2, as_frame=False, parser="auto"), categoricalFeatures)
        return {"data": data, "categorical": categoricalFeatures, "name": "Adult", "filename": "adult.html", "task": "classification"}
    
    if num == 9: 

        categoricalFeatures = [1,2,3,4,5,6,7,8,9,10,11,12,18]
        data = data_preprocessing( fetch_openml(data_id=269, as_frame=False, parser="auto"), categoricalFeatures)
        return {"data": data, "categorical": categoricalFeatures, "name": "Hepatitis", "filename": "hepatitis.html", "task": "classification"}
    

    #REGRESSION ONLY NUMERICAL
    if num == 10: 
        #442 rows, 10 columns
        #Real from 25 to 346
        categoricalFeatures = None
        data = load_diabetes()
        return {"data": data, "categorical": categoricalFeatures, "name": "Diabetes Progression", "filename": "diabetes_progression.html", "task": "regression"}
    
    if num == 11: 
        #506 rows, 14 columns
        #Real from 0 to 228
        categoricalFeatures = [3, 8]
        data = data_preprocessing(fetch_openml(name="boston", as_frame=False, version=1), categoricalFeatures)
        return {"data": data, "categorical": categoricalFeatures, "name": "Boston Housing", "filename": "boston_housing.html", "task": "regression"}
    
    if num == 12: 
        #20640 rows, 8 columns
        #Real from 0.15 to 5
        categoricalFeatures = None
        data = fetch_california_housing()
        return {"data": data, "categorical": categoricalFeatures, "name": "California Housing", "filename": "california_housing.html", "task": "regression"}
    
    
