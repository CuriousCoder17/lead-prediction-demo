import os

import pandas as pd

PATH_TO_DATA_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "data"
)


def read_main_data():
    return pd.read_csv(os.path.join(PATH_TO_DATA_DIR, "train", "train.csv"))
