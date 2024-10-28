
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def training_asn(X_train, X_test, target):
    y_train = X_train[target]
    y_test = X_test[target]
    X_train = X_train.drop(target, axis=1)
    X_test = X_test.drop(target, axis=1)
    
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    # Predict data test
    y_pred = rfc.predict(X_test)
    # print('Avarage Recall score', np.mean(rf_score))
    print('Test Recall score', recall_score(y_test, y_pred))
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, rfc
