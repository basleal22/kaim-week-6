import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract(path):
    data= pd.read_csv(path)
    info=data.info()
    description=data.describe()
    null_values=data.isnull().sum()
    return data,description,null_values
def visualization(data):
    # Distribution check (using visualization)
    # Histogram for numerical features
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()
def visual_num(data):
    # Identify numerical columns
    #numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize = (15, 10), bins = 50);

# Visualize distributions
    #for col in numerical_columns:
        #plt.figure(figsize=(8, 5))
        #sns.histplot(data[col], kde=True, bins=30, color='blue')
        #plt.title(f"Distribution of {col}")
        #plt.xlabel(col)
        #plt.ylabel("Frequency")
        #plt.show()

# Boxplots for outlier detection
    #for col in numerical_columns:
        #plt.figure(figsize=(8, 5))
        #sns.boxplot(x=data[col], color='orange')
        #plt.title(f"Boxplot of {col}")
        #plt.xlabel(col)
        #plt.show()
def visual_cat(data):
        # Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns

# Visualize distributions
    for col in categorical_columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(y=data[col], order=data[col].value_counts().index, palette='viridis')
        plt.title(f"Distribution of {col}")
        plt.xlabel("Frequency")
        plt.ylabel(col)
        plt.show()
def correlation(data):
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Correlation matrix
    correlation_matrix = data[numerical_columns].corr()

    # Visualize with a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()
    