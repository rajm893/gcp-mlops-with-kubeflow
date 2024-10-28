
import pandas as pd
import numpy as np


def extract_predictions(predictions):
    data = []
    for record in predictions:
        print("record: ",record)
        prediction = record["prediction"]
        data.append(prediction)
    return pd.DataFrame(data, columns=['predicition_type'])


def get_predictions(predictions, input_df):
    predictions = extract_predictions(predictions)
    result_df = pd.concat([input_df, predictions], axis=1)
    return result_df
