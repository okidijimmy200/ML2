import pandas as pd
import numpy as np
import os

def get_column_datetime(df):
    df = df.copy()
    df = df.apply(lambda col: pd.to_datetime(col, errors='ignore', infer_datetime_format=True) if col.dtypes == object else col, axis=0) # Convert relevant col to datetime
    datetime_col = df.select_dtypes(['datetimetz', np.datetime64]).columns.tolist() # list datetime columns    
    datetime_cols = []
    
    for col in datetime_col:
        datetime_cols.append((col, df[col].dtype)) 
        
    return df, datetime_cols   


def get_column_dtype(df, threshold=15):
    categorical_col = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_col_temp = df.select_dtypes(exclude=["object"]).columns.tolist()
    integer_col = df.select_dtypes(include=['int']).columns.tolist()
    numerical_col = []
    for num_col in numerical_col_temp:
        unique_counts = df[num_col].nunique()
        if (unique_counts < threshold):
            categorical_col.append(num_col)
        else:
            numerical_col.append(num_col)
    return categorical_col, integer_col, numerical_col_temp


def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_large_categorical_columns(df, categorical_col, threshold=50):
    df = df.copy()
    removed_categorical_cols = []
    # If unique values from a categorical col is more than threshold value, remove it 
    for col in categorical_col:
        unique_counts = df[col].nunique()
        if unique_counts > threshold:
            removed_categorical_cols.append((col, unique_counts))
            df.drop(col, inplace=True, axis=1)  
    return df, removed_categorical_cols

def datetime_to_int(df, datetime_cols):
    for col in datetime_cols:  
        col_name = col[0]  
        df[col_name] = df[col_name].apply(lambda x: x.value)        
        # print(df[col_name].dtypes)
        # print(df[col_name])        
    return df

def int_to_datetime(df, datetime_cols):
    for col in datetime_cols:  
        col_name = col[0]  
        col_dtype = col[1]
        if col_dtype == 'datetime64[ns, UTC]':
            df[col_name] = df[col_name].astype('datetime64[ns, UTC]')        
        # print(df[col_name].dtypes)
        # print(df[col_name])        
    return df