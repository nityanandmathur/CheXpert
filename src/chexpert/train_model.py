import argparse
import os
from typing import Text

import numpy as np
import tensorflow as tf
import wandb
from data_load import load_dataframe, load_train_images, load_val_images
from hydra import compose, initialize
from loss import compute_class_weights, set_binary_crossentropy_weighted_loss
from PIL import Image
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, save_model


def train(config_name: Text) -> None:
    '''
    Function to train the deep learning model.
    I/P: Configuration file path. You can select between Densenet, Resnet50 & VGG16.
    '''

    # Loading configuration file using Hydra
    initialize(version_base=None, config_path='../../configs')
    config = compose(config_name=config_name)
    wandb.init(project='chexpert')

    # Loading parameters from configuration file
    model = config.params.model
    img_size = config.params.img_size

    batch_size = config.params.batch_size
    learning_rate = config.params.learning_rate
    training_epoch = config.params.training_epoch
    no_gpu = config.params.no_gpu

    train_data_path = config.data.processed.train
    val_data_path = config.data.processed.val

    data_dir = config.data.data_dir

    # Loading training and validation dataframes
    train_df = load_dataframe(train_data_path)
    val_df = load_dataframe(val_data_path)

    #Loading image data
    train_data = load_train_images(df=train_df, dir=data_dir, batch_size=batch_size, img_size=img_size)
    val_data = load_val_images(df=val_data, dir=data_dir, batch_size=batch_size, img_size=img_size)

    #Initiallizing GPU Devices
    gpu_devices = [""]*no_gpu
    for i in range(no_gpu):
        gpu_devices[i] = f"/gpu:{i}"




# Main function
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', dest='config', required=True)
    args = argparser.parse_args()

    train(config_name=args.config)
