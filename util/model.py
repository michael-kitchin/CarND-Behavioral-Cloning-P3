import keras
import math
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Lambda, Conv2D
from keras.models import Sequential


class SampleSequence(keras.utils.Sequence):
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
    return image[60 + 0:126 + 0, 60 + 0:260 + 0]


def normalize_image(x):
    x = x - keras.backend.mean(x, (1, 2), keepdims=True)
    x_maxabs = keras.backend.max(keras.backend.abs(x), (1, 2), keepdims=True)

    return x / x_maxabs


def build_model(activation='elu', dropout=0.0):
    model = Sequential()
    model.add(Lambda(lambda x: normalize_image(x), input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), padding='valid', strides=(2, 2), activation=activation))
    model.add(Conv2D(36, (5, 5), padding='valid', strides=(2, 2), activation=activation))
    model.add(Conv2D(48, (5, 5), padding='valid', strides=(2, 2), activation=activation))
    model.add(Conv2D(64, (3, 3), padding='valid', activation=activation))
    model.add(Conv2D(64, (3, 3), padding='valid', activation=activation))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(100, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation=activation))
    model.add(Dense(1))

    return model
