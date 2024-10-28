
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_preprocessing(df):
    
    # df['type'] = df['type'].apply(lambda x: 0 if x == 'white' else 1)
    df.fillna(df.mean(), inplace=True)
    
    sc=StandardScaler()
    scaler = sc.fit(df)
    df = scaler.transform(df)
    df = pd.DataFrame(df)
    df = df.reset_index(drop=True)
    
    return df
