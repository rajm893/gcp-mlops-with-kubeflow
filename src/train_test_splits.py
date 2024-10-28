
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.utils import shuffle

def get_train_test_splits(df, target_column, test_size):
    df = shuffle(df)
    x = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    sc=StandardScaler()
    scaler = sc.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    y_train.name = 'type'
    y_test.name = 'type'
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train = pd.concat([X_train, y_train], axis=1)
    X_test = pd.concat([X_test, y_test], axis=1)
    X_train.columns = x.columns.to_list() + [target_column]
    X_test.columns = x.columns.to_list() + [target_column]
    X_train.to_csv("gs://my-data-classification/train_data_new.csv", index=False)
    return X_train, X_test