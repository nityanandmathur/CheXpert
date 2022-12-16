import argparse
from typing import Text

import wandb
from data_load import load_dataframe, load_train_images, load_val_images
from hydra import compose, initialize
from loss import compute_class_weights, set_binary_crossentropy_weighted_loss
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, save_model


def train(config_name: Text) -> None:
    '''
    Function to train the deep learning model.
    I/P: Configuration file path. You can select between Densenet201, Resnet50 & VGG16.
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
    val_data = load_val_images(df=val_df, dir=data_dir, batch_size=batch_size, img_size=img_size)

    #Initiallizing GPU Devices
    gpu_devices = [""]*no_gpu
    for i in range(no_gpu):
        gpu_devices[i] = f"/gpu:{i}"

    optimizer = optimizers.Adam(
    lr=learning_rate,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=0.1
    )

    positive_weights, negative_weights = compute_class_weights(labels=train_data.labels)
    print(f"\nPositive Weights: {positive_weights}")
    print(f"Negative Weights: {negative_weights}\n")

    loss = set_binary_crossentropy_weighted_loss(
        positive_weights=positive_weights,
        negative_weights=negative_weights,
        epsilon=1e-7
    )




# Main function
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', dest='config', required=True)
    args = argparser.parse_args()

    train(config_name=args.config)
