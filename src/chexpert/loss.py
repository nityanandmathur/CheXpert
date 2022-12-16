import numpy as np
import tensorflow.keras.backend as K


def compute_class_weights(labels):

    # total number of patients (rows).
    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = np.sum(labels == 0, axis=0) / N

    positive_weights = negative_frequencies
    negative_weights = positive_frequencies

    return positive_weights, negative_weights


def set_binary_crossentropy_weighted_loss(positive_weights, negative_weights, epsilon=1e-7):

    def binary_crossentropy_weighted_loss(y_true, y_pred):

        # initialize loss to zero
        loss = 0.0

        for i in range(len(positive_weights)):
            # for each class, add average weighted loss for that class
            loss += -1 * K.mean((positive_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) +
                                 negative_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)))
        return loss

    return binary_crossentropy_weighted_loss
