import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use classifier instead of regressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def normalizing(data):
    for col in ['Recency_x', 'Frequency_x', 'Monetary_x']:
        data[f'{col}_Scaled'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data
def split_model(data):
    feature=['Amount','ProductCategory_data_bundles',
       'woe','Recency_x', 'Frequency_x', 'Monetary_x']
    #'ProductCategory_financial_services', 'ProductCategory_movies',
    #   'ProductCategory_other', 'ProductCategory_ticket',
    #   'ProductCategory_transport', 'ProductCategory_tv',
    #   'ProductCategory_utility_bill'
    target=['Risk_Level_Ordinal']
    x=data[feature]
    y=data[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return  x_train,x_test,y_train,y_test
