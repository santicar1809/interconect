import pandas as pd
import os
import re
import numpy as np

def cancel(data):
    if data=='No':
        return '0'
    else:
        return '1'

def binary_categories(data):
    if data=='Yes':
        return 1
    elif data=='No':
        return 0

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data

def preprocess_data(data):
    '''This function will clean the data by setting removing duplicates, 
    formatting the column types, names and removing incoherent data. The datasets
    will be merged in one joined by the CustomerID'''
    df_contract = data[0] 
    df_internet = data[1] 
    df_personal = data[2] 
    df_phone    = data[3] 
    
    df_contract = columns_transformer(df_contract)
    df_internet = columns_transformer(df_internet)
    df_personal = columns_transformer(df_personal)
    df_phone = columns_transformer(df_phone)
    
    # Contratos
    df_contract['total_charges'] = df_contract['total_charges'].replace(' ',np.nan)
    df_contract['total_charges'] = pd.to_numeric(df_contract['total_charges'])
    df_contract['total_charges'].fillna(df_contract['total_charges'].mean(),inplace=True)
    #Pasamos la columna total charges a float
    df_contract['total_charges'] = df_contract['total_charges'].astype('float')
    
    # Merging data
    data = df_contract.merge(df_personal.merge(df_internet.merge(df_phone,how='left',on='customer_id'),how='left',on='customer_id'),how='left',on='customer_id')
    
    # Preprocesing merged dataset
    
    data['cancel'] = data['end_date'].apply(cancel)
    
    binary_columns=['partner', 'dependents','paperless_billing', 'online_security', 'online_backup',
       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
       'multiple_lines']
    for column in binary_columns:
           data[column]=data[column].apply(binary_categories)
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    path = './files/datasets/intermediate/'

    if not os.path.exists(path):
        os.makedirs(path)

    data.to_csv(path+'merge.csv', index=False)

    print(f'Dataframe created at route: {path}merge.csv ')

    return data