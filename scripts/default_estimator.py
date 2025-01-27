import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
def normalize_columns(data):
    for col in ['Recency', 'Frequency', 'Monetary']:
        data[f'{col}_Scaled'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data
def aggregate(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'], errors='coerce')
    data['DaysSinceLast']=(data['TransactionStartTime'].max()-data['TransactionStartTime']).dt.days
    recency=data.groupby('CustomerId')['DaysSinceLast'].min().reset_index(name='Recency')
    # Frequency: Number of transactions per customer
    frequency = data.groupby('CustomerId')['TransactionId'].count().reset_index(name='Frequency')
    # Monetary: Total transaction amount for each customer
    monetary = data.groupby('CustomerId')['Amount'].sum().reset_index(name='Monetary')
    # Combine RFMS components
    rfms = recency.merge(frequency, on='CustomerId').merge(monetary, on='CustomerId')
    return rfms
def calculate_rfms_score(data):
    # Compute RFMS Score
    data['RFMS_Score'] = (
        0.4 * data['Recency_Scaled'] +
        0.3 * data['Frequency_Scaled'] +
        0.3 * data['Monetary_Scaled']
    )
    # Classify users based on RFMS score
    threshold = data['RFMS_Score'].median()
    data['Classification'] = np.where(data['RFMS_Score'] >= threshold, 'Good', 'Bad')
    
    return data