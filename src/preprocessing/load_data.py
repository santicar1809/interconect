import pandas as pd

def load_datasets():
    '''This function will upload the necessary datasets
    to perform the project.'''
    df_contract = pd.read_csv('files/datasets/input/contract.csv')
    df_internet = pd.read_csv('files/datasets/input/internet.csv')
    df_personal = pd.read_csv('files/datasets/input/personal.csv')
    df_phone = pd.read_csv('files/datasets/input/phone.csv')
    return df_contract, df_internet, df_personal, df_phone