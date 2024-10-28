import pandas as pd
import numpy as np

def get_preprocessing(df):
    
    df['type'] = df['type'].apply(lambda x: 0 if x == 'white' else 1)
    df.fillna(df.mean(), inplace=True)
    return df
