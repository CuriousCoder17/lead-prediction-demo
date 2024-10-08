import os.path

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.core.io import read_main_data, PATH_TO_DATA_DIR


def train_model():
    main_data = read_main_data()

    train_data = main_data[
        [
            'Age', 'Gender', 'Location',
            'LeadSource', 'DeviceType', 'LeadStatus',
            'Conversion (Target)'
        ]
    ]
    # Assuming train_data is already defined and contains the necessary columns
    features = ['Age', 'Gender', 'Location', 'LeadSource', 'DeviceType',
                'LeadStatus']
    target = 'Conversion (Target)'

    X = train_data[features]
    y = train_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the CatBoostClassifier
    model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=6, verbose=0
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        cat_features=[
            'Gender', 'Location', 'LeadSource', 'DeviceType', 'LeadStatus'
        ]
    )

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    model.save_model(os.path.join(PATH_TO_DATA_DIR, "catboost_model.cbm"))


if __name__ == '__main__':
    train_model()
