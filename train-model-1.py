import math
import csv
import cv2
import keras
import numpy as np
from keras.layers import Flatten, Dense, Dropout, Lambda, Conv2D
from keras.models import Sequential
from sklearn.utils import shuffle


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


def normalize_image(x):
    x = x - keras.backend.mean(x, (1, 2), keepdims=True)
    x_maxabs = keras.backend.max(keras.backend.abs(x), (1, 2), keepdims=True)

    return x / x_maxabs


def build_model(activation='elu', dropout=0.0):
    model = Sequential()
    model.add(Lambda(lambda x: normalize_image(x), input_shape=(160, 320, 3)))
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


lines = []
with open('./input_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

base_path = './input_data/IMG/'
steering_correction = 0.2
images = []
measurements = []
for rev_ctr in range(2):
    for field_ctr in range(3):
        line_ctr = 0
        for line in lines:
            if line_ctr > 0:
                if line_ctr % 1000 == 0:
                    print ("rev: ", rev_ctr, "field: ", field_ctr, "line: ", line_ctr)
                field_path = (base_path + line[field_ctr].split('/')[-1])
                field_steering = float(line[3])

                if field_ctr == 1:
                    field_steering = (field_steering - steering_correction)
                else:
                    field_steering = (field_steering + steering_correction)

                image = cv2.imread(field_path, cv2.COLOR_BGR2RGB)
                if rev_ctr == 0:
                    images.append(image)
                    measurements.append(field_steering)
                else:
                    images.append(cv2.flip(image, 1))
                    measurements.append(field_steering * -1.0)
            line_ctr = line_ctr + 1

X_input = np.array(images)
y_input = np.array(measurements)
assert (len(X_input) == len(y_input))
X_input, y_input = shuffle(X_input, y_input)

loss = 'mse'
optimizer = 'adam'
activation = 'elu'
dropout = 0.5
epochs = 10
validation_split = 0.2
batch_size = 128
validation_index = int(len(X_input) * (1.0 - validation_split))
print ('Samples:', len(X_input),
       "Training:", validation_index,
       "Validation:", len(X_input) - validation_index)

X_train = np.array(X_input[0:validation_index])
y_train = np.array(y_input[0:validation_index])
X_validation = np.array(X_input[validation_index:len(X_input)])
y_validation = np.array(y_input[validation_index:len(y_input)])

train_generator = SampleSequence(X_train, y_train, batch_size)
validation_generator = SampleSequence(X_validation, y_validation, batch_size)

model = build_model(activation, dropout)
model.compile(loss=loss, optimizer=optimizer)
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    epochs=epochs)

model.save('model.h5')
