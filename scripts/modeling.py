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
       'ProductCategory_financial_services', 'ProductCategory_movies',
       'ProductCategory_other', 'ProductCategory_ticket',
       'ProductCategory_transport', 'ProductCategory_tv',
       'ProductCategory_utility_bill', 'ChannelId_ChannelId_2',
       'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5','woe','Recency_x_Scaled','Frequency_x_Scaled','Monetary_x_Scaled']
    target=['Risk_Level_Ordinal']
    x=data[feature]
    y=data[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return  x_train,x_test,y_train,y_test
    # Initialize RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100)

    # Train the model
    model.fit(x_train, y_train)

    # Predict on the test set
    y_pred = model.predict(x_test)
    return y_test, y_pred
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
