from typing import Text

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


def Build_Model(model: Text, freeze=False) -> Model:
    if model == 'densenet201':
        base_model = DenseNet201(
            include_top=True,
            weights='imagenet'
        )
    elif model == 'resnet50':
        base_model = ResNet50(
            include_top=True,
            weights='imagenet'
        )
    elif model == 'vgg16':
        base_model = VGG16(
            include_top=True,
            weights='imagenet'
        )
    else:
        raise Exception("Model cannot be loaded")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    predictions = Dense(14, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if freeze:
        for layer in base_model.layers:
            layer.trainable=False

    return model
