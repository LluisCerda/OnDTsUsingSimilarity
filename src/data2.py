from sklearn.datasets import load_wine, load_iris, fetch_california_housing, fetch_openml, load_diabetes, load_breast_cancer, load_digits
import utils as utils
import numpy as np
from sklearn.utils import Bunch

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
        return {"data": data, "categorical": categoricalFeatures, "name": "Gas Drift", "filename": "gas_drift.html", "task": "classification"}

    if num == 7: 
        #58000 rows, 9 columns
        data = fetch_openml(name="shuttle", as_frame=False, version=1, parser="auto")
        data.target = data.target.astype(int)
        data.data = data.data.astype(np.float64)  
        return {"data": data, "categorical": categoricalFeatures, "name": "Shuttle", "filename": "shuttle.html", "task": "classification"}
    

    #CLASSIFICATION NUMERICAL AND CATEGORICAL
    if num == 8: #Has missing values
        categoricalFeatures = [1, 3, 5, 6, 7, 8, 9, 13]
        data = fetch_openml("adult", version=2, as_frame=False, parser="auto")
        return {"data": data, "categorical": categoricalFeatures, "name": "Adult", "filename": "adult.html", "task": "classification"}
    
    if num == 9: 

        categoricalFeatures = [1,2,3,4,5,6,7,8,9,10,11,12,18]
        data = fetch_openml(data_id=269, as_frame=False, parser="auto")
        return {"data": data, "categorical": categoricalFeatures, "name": "Hepatitis", "filename": "hepatitis.html", "task": "classification"}
    
    if num == 10:
        #1000 rows , 21 columns
        # 7 numeric, 14 categorical
        categoricalFeatures = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = fetch_openml(name="default-of-credit-card-clients", as_frame=False, parser="auto")
        utils.visualize_dataframe(data)
        return {"data": data, "categorical": categoricalFeatures, "name": "credit-card", "filename": "credit_card.html", "task": "classification"}

    if num == 11:
        #270 rows, 14 columns
        categoricalFeatures = [1, 2, 3, 4, 6, 7, 8, 10, 15]
        data = fetch_openml(data_id=1461, as_frame=False, parser="auto")
        utils.visualize_dataframe(data)
        return {"data": data, "categorical": categoricalFeatures, "name": "bank-marketing", "filename": "bank_marketing.html", "task": "classification"}

    #REGRESSION
    if num == 12: 
        #442 rows, 10 columns
        #Real from 25 to 346
        categoricalFeatures = None
        data = load_diabetes()
        return {"data": data, "categorical": categoricalFeatures, "name": "Diabetes Progression", "filename": "diabetes_progression.html", "task": "regression"}
    
    if num == 13: 
        #506 rows, 14 columns
        #Real from 0 to 228
        categoricalFeatures = [3, 8]
        data = fetch_openml(name="boston", as_frame=False, version=1, parser="auto")
        return {"data": data, "categorical": categoricalFeatures, "name": "Boston Housing", "filename": "boston_housing.html", "task": "regression"}

    if num == 14: 
        #20640 rows, 8 columns
        #Real from 0.15 to 5
        categoricalFeatures = None
        data = fetch_california_housing()
        return {"data": data, "categorical": categoricalFeatures, "name": "California Housing", "filename": "california_housing.html", "task": "regression"}
    
    
    if num == 15:
        # 17379 rows, 12 columns
        # Real from 0 to 1
        # categorical 5, numeric 7
        categoricalFeatures = [0, 1, 4, 6, 7]
        data = fetch_openml(name="Bike_sharing_Demand", as_frame=False, parser="auto", version=2)
        return {"data": data, "categorical": categoricalFeatures, "name": "Bike Sharing Demand", "filename": "bike_sharing_demand.html", "task": "regression"}
    
    if num == 16:
        # 1030 rows, 9 columns
        # Real from 0 to 82.6
        # categorical 9
        categoricalFeatures = None
        data = fetch_openml(name="Concrete_Compressive_Strength", as_frame=False, parser="auto", version=7)
        return {"data": data, "categorical": categoricalFeatures, "name": "Concrete Compressive Strength", "filename": "concrete_compressive_strength.html", "task": "regression"}