import keras
import math
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Lambda, Conv2D
from keras.models import Sequential


class SampleSequence(keras.utils.Sequence):
    """
    Convenient, thread-safe sequence generator
    """

    def __init__(self, X_set, y_set, batch_size):
        assert (len(X_set) == len(y_set))
        self.x, self.y = X_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), \
               np.array(batch_y)


def preprocess_image(image):
    """
    Crop image to (a) exclude un-constructive sections and
    (b) match model optimal input w/o scaling.
    """
    return image[60:126, 60:260]


def normalize_image(x):
    """
    Normalize image using keras utils.
    :param x: Input image.
    :return: Normalized image.
    """
    x = x - keras.backend.mean(x, (1, 2), keepdims=True)
    x_maxabs = keras.backend.max(keras.backend.abs(x), (1, 2), keepdims=True)

    return x / x_maxabs


def build_model(activation='relu', dropout=0.5):
    """
    Builds nVidia Dave2-complaint CNN.
    See: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    :param activation:  Activation function.
    :param dropout: Dropout probability.
    :return: Build model.
    """
    model = Sequential()
    # Input: Normalize, 66x200x3
    model.add(Lambda(lambda x: normalize_image(x), input_shape=(66, 200, 3)))
    # First convolution: 24x31x98, 5x5 kernel
    model.add(Conv2D(24, (5, 5), padding='valid', strides=(2, 2), activation=activation))
    # Second convolution: 36x14x47, 5x5 kernel
    model.add(Conv2D(36, (5, 5), padding='valid', strides=(2, 2), activation=activation))
    # Third convolution: 48x5x22, 5x5 kernel
    model.add(Conv2D(48, (5, 5), padding='valid', strides=(2, 2), activation=activation))
    # Fourth convolution: 64x3x20, 3x3 kernel
    model.add(Conv2D(64, (3, 3), padding='valid', activation=activation))
    # Fifth convolution: 64x1x18, 3x3 kernel
    model.add(Conv2D(64, (3, 3), padding='valid', activation=activation))
    model.add(Flatten())
    model.add(Dropout(dropout))
    # Flatten to fully-connected, 100-neuron layer
    model.add(Dense(100, activation=activation))
    model.add(Dropout(dropout))
    # Fully-connected, 50-neuron layer
    model.add(Dense(50, activation=activation))
    model.add(Dropout(dropout))
    # Fully-connected, 10-neuron layer
    model.add(Dense(10, activation=activation))
    # Fully-connected, 1-neuron (control) layer
    model.add(Dense(1))

    return model
