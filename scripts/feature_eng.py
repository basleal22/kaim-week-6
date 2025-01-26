import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
def num_aggregate(data):
    num_cols= ['Amount', 'Value', 'PricingStrategy', 'FraudResult','TransactionYear']
    num_agg= data.groupby('CustomerId')[num_cols].agg('mean','sum','max','min').reset_index()
    return num_agg
def cat_aggregate(data):
    cat_cols=data.select_dtypes(include=['object']).columns.tolist()
    cat_agg=data.groupby('CustomerId')[cat_cols].agg(lambda x:x.mode().iloc[0] if not x.mode().empty else None)
    return cat_agg
def extract_date(data):
    # Convert TransactionStartTime to datetime
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    # Extract transaction time-related features
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour  # Hour of the day
    data['TransactionDay'] = data['TransactionStartTime'].dt.day    # Day of the month
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month  # Month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year   # Year
    return data
def onehot_encoding(data):
    # One-Hot Encoding
    Labelencoder=LabelEncoder()
    data = pd.get_dummies(data, columns=['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId'],drop_first=True)
     # Convert one-hot encoded columns to integers
    for col in data.select_dtypes(include=['bool']).columns:
        data[col] = data[col].astype(int)
    data['TransactionYear']=Labelencoder.fit_transform(data['TransactionYear'])
    data['ProviderId']=Labelencoder.fit_transform(data['ProviderId'])
    return data
def scaler(data):
    scaler=MinMaxScaler()
    data[['Amount', 'Value']]=data.fit_transform(data[['Amount', 'Value']])
    return data


