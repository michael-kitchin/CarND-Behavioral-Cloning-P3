from util.model import *
import math
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

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
epochs = 8
validation_split = 0.2
batch_size = 256
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
