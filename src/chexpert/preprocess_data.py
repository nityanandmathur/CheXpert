import argparse
import os
from typing import Text

import pandas as pd
from hydra import compose, initialize
from sklearn.model_selection import train_test_split


def fill(df):
    '''
    Filling N/A's in dataframe with zeros.
    '''
    return df.fillna(0)

def zeroing(df):
    '''
    Replacing -1's in dataframe with zeros.
    '''
    return df.replace(-1, 0)

def reorder(df):
    '''
    Reordering the columns in dataframe alphabetically.
    '''
    reordered_columns = ["Path",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "Lung Lesion",
    "Fracture",
    "Support Devices",
    "No Finding"]

    return df[reordered_columns]

def change_path(df):
    '''
    Changing path of csv from CheXpert-v1.0-small/## to data/raw/CheXpert-v1.0-small/##
    '''
    df['Path'] = df['Path'].apply(lambda x: x.replace(os.path.dirname(x),'data/raw/'+os.path.dirname(x)))
    return df

def preprocess(config_name: Text) -> None:

    '''
    Processing the data:
        1. Replacing -1's and N/A's with zeros.
        2. Removing Sex, Age, Frontal/Lateral, AP/PA, from the data.
        3. Reordering columns
        4. Changing paths to add folder name.

    train.csv(raw) -> train.csv(processed) for training + valid.csv(processed) for validation (80/20 Split)
    valid.csv(raw) -> test.csv(processed) for testing the completely trained and validated model.

    Do not use test.csv until model is ready for testing.

    Using paths meant only for devcontainers.
    If you are using any other platform, kindly update the paths as per your platform.
    '''

    print(preprocess.__doc__)

    initialize(version_base=None, config_path='../../configs')
    config = compose(config_name=config_name)

    train_data_df = pd.read_csv(config.data.raw.train)
    test_df = pd.read_csv(config.data.raw.test)

    train_df, val_df = train_test_split(train_data_df, test_size=0.2, random_state=40, shuffle=True)

    #print(fill.__doc__)
    train_df = fill(train_df)
    val_df = fill(val_df)
    test_df = fill(test_df)

    #print(zeroing.__doc__)
    train_df = zeroing(train_df)
    val_df = zeroing(val_df)
    test_df = zeroing(test_df)

    #print(reorder.__doc__)
    train_df = reorder(train_df)
    val_df = reorder(val_df)
    test_df = reorder(test_df)

    #print(change_path.__doc__)
    train_df = change_path(train_df)
    val_df = change_path(val_df)
    test_df = change_path(test_df)

    train_df.to_csv(config.data.processed.train, index=False)
    val_df.to_csv(config.data.processed.val, index=False)
    test_df.to_csv(config.data.processed.test, index=False)

    print('Data processed and saved')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', dest='config', required=True)
    args = argparser.parse_args()

    preprocess(config_name=args.config)
