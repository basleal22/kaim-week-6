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
def classify_risk_level(row):
    # Example criteria (adjust thresholds based on your data and business understanding)
    if row['Recency'] <= 30 and row['Frequency'] >= 5 and row['Monetary'] >= 1000:
        return 'Low'  # Low risk: Recent transactions, high frequency, and high spending
    elif 30 < row['Recency'] <= 90 and row['Frequency'] >= 2 and row['Monetary'] >= 500:
        return 'Medium'  # Medium risk: Somewhat recent, moderate frequency, moderate spend
    else:
        return 'High'  # High risk: Less recent transactions, low frequency, low spending
def woe_binning(data,feature,target):
    # Create bins for the feature using pd.qcut (quantile-based binning)
    # Adjust the number of bins if necessary
    bins = pd.qcut(data[feature], q=5, labels=False, duplicates='drop')  # Using 5 bins as an example
    
    # Create a DataFrame with feature bins and the target variable
    data['bin'] = bins
    grouped = data.groupby('bin')[target].agg(['count', 'sum'])  # count of each bin and sum of target (defaults)
    
    # Calculate WoE for each bin
    grouped['dist_good'] = 1 - grouped['sum'] / grouped['sum'].sum()  # Proportion of good (not default)
    grouped['dist_bad'] = grouped['sum'] / grouped['sum'].sum()  # Proportion of bad (default)
    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])  # WoE formula
    
    # Merge the WoE values back to the data
    data = data.merge(grouped[['woe']], left_on='bin', right_index=True, how='left')
    data = data.drop(columns=['bin'])  # Drop the bin column after merging WoE
    
    return data
