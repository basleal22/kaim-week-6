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
#def classify_risk_level(data):
#    recency = float(data['Recency'])
 #   frequency = float(data['Frequency'])
#    monetary = float(data['Monetary'])
#    # Adjusted thresholds
  #  if data['Recency'] < 30 and data['Frequency'] > 10:
   #     return 'High'
    #else:
    #    return 'Low'
def classify_risk_level(row):
    recency = row['Recency']
    frequency = row['Frequency']
    monetary = row['Monetary']
    
    if recency <= 15 and frequency >= 10 and monetary >= 2000:
        return 'Low'
    elif 15 < recency <= 60 and frequency >= 5 and monetary >= 1000:
        return 'Medium'
    else:
        return 'High'
def woe_binning(data, feature, target, bins=10):
    # Create bins for the feature
    data['bin'] = pd.cut(data[feature], bins=bins, labels=False, include_lowest=True)
    
    # Group by bins and calculate necessary aggregates
    grouped = data.groupby('bin')[target].agg(['count', 'sum'])
    
    # Add a small constant to avoid division by zero
    grouped['good'] = grouped['count'] - grouped['sum']  # Non-target class
    grouped['bad'] = grouped['sum']  # Target class
    
    grouped['woe'] = np.log((grouped['good'] / grouped['good'].sum() + 1e-6) /
                            (grouped['bad'] / grouped['bad'].sum() + 1e-6))
    
    # Map the WoE back to the original data
    woe_map = grouped['woe'].to_dict()
    data['woe'] = data['bin'].map(woe_map)
    
    # Drop the 'bin' column after use (optional)
    data.drop(columns=['bin'], inplace=True)
    
    return data
