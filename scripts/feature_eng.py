import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
def num_aggregate(data):
    num_cols= data.select_dtypes(include=['int64','float64']).columns.tolist()
    num_agg= data.groupby('CustomerId')[num_cols].agg('mean','sum','max','min').reset_index()
    return num_agg
def cat_aggregate(data):
    cat_cols=data.select_dtypes(include=['object']).columns.tolist()
    cat_agg=data.groupby('CustomerId')[cat_cols].agg(lambda x:x.mode().iloc[0] if not x.mode().empty else None)
    return cat_agg
    
