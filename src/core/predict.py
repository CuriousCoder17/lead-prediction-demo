import os

from catboost import CatBoostClassifier
import pandas as pd

from core.io import PATH_TO_DATA_DIR


def load_and_predict(new_data):
    # Load the model
    model = CatBoostClassifier()
    model.load_model(os.path.join(PATH_TO_DATA_DIR, "catboost_model.cbm"))

    predictions = model.predict(new_data)
    return predictions


# Example usage
if __name__ == '__main__':
    # Example new data
    new_data = pd.DataFrame({
        'Age': [25, 35, 30],
        'Gender': ['Male', 'Female', 'Female'],
        'Location': ['Islamabad', 'Lahore', 'Lahore'],
        'LeadSource': ['Email', 'Referral', 'Referral'],
        'DeviceType': ['Tablet', 'Desktop', 'Desktop'],
        'LeadStatus': ['Warm', 'Cold', 'Cold']
    })

    predictions = load_and_predict(new_data)
    print(predictions)
