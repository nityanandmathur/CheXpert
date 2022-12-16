from typing import Text

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataframe(Path: Text) -> pd.DataFrame:
    '''
    Function to create and return pandas dataframe with Path to images and diseases.
    I/P: Path to csv file.
    O/P: Pandas Dataframe
    '''
    df = pd.read_csv(filepath_or_buffer=Path, dtype={
        "Path": str,
        "Atelectasis": np.float32,
        "Cardiomegaly": np.float32,
        "Consolidation": np.float32,
        "Edema": np.float32,
        "Pleural Effusion": np.float32,
        "Pleural Other": np.float32,
        "Pneumonia": np.float32,
        "Pneumothorax": np.float32,
        "Enlarged Cardiomediastinum": np.float32,
        "Lung Opacity": np.float32,
        "Lung Lesion": np.float32,
        "Fracture": np.float32,
        "Support Devices": np.float32,
        "No Finding": np.float32
    })
    return df

def load_train_images(df: pd.DataFrame, dir: Text, batch_size: int, img_size: int) -> ImageDataGenerator:
    '''
    Function to load images from given dataframe.

    Parameters:
        1. df: pandas dataframe containing the path to images
        2. dir: directory containing images
        3. batch_size
        4. img_size: dimension of images to load

    Returns:
        Image data
    '''
    list_col = list(df.columns)
    y_cols = list_col[1::] # Dropping 'Path' column

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,
        shear_range=0.1,
        zoom_range=0.1,
        cval=0.0,
        fill_mode='constant',
        horizontal_flip=False,  # Some labels would be heavily affected by this change if it is True
        vertical_flip=False  # Not suitable for Chest X-ray images if it is True
    )

    data = datagen.flow_from_dataframe(
        dataframe=df,
        directory=dir,
        x_col='Path',
        y_col=y_cols,
        target_size=(img_size, img_size),
        class_mode='raw',
        batch_size=batch_size,
        shuffle=True,
        validate_filenames=True
    )

    return data

def load_val_images(df: pd.DataFrame, dir: Text, batch_size: int, img_size: int) -> ImageDataGenerator:
    '''
    Function to load images from given dataframe.

    Parameters:
        1. df: pandas dataframe containing the path to images
        2. dir: directory containing images
        3. batch_size
        4. img_size: dimension of images to load

    Returns:
        Image data
    '''
    list_col = list(df.columns)
    y_cols = list_col[1::] # Dropping 'Path' column

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    data = datagen.flow_from_dataframe(
        dataframe=df,
        directory=dir,
        x_col='Path',
        y_col=y_cols,
        target_size=(img_size, img_size),
        class_mode='raw',
        batch_size=batch_size,
        shuffle=True,
        validate_filenames=True
    )

    return data
