import csv
import cv2
import scipy.ndimage
from itertools import product
from sklearn.utils import shuffle

from util.model import *

lines = []
with open('./input_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    line_ctr = 0
    for line in reader:
        if line_ctr > 0:
            lines.append(line)
        line_ctr = line_ctr + 1

base_path = './input_data/IMG/'
steering_correction = 0.1
images = []
measurements = []
for config in product(range(2), range(3)):
    rev_ctr = config[0]
    field_ctr = config[1]
    line_ctr = 0
    for line in lines:
        if line_ctr % 1000 == 0:
            print ("Reverse: ", rev_ctr, "Field: ", field_ctr, "Line: ", line_ctr)
        field_path = (base_path + line[field_ctr].split('/')[-1])
        field_steering = float(line[3])

        if field_ctr == 1:
            field_steering = (field_steering + steering_correction)
        else:
            field_steering = (field_steering - steering_correction)

        image = preprocess_image(scipy.ndimage.imread(field_path, mode='RGB'))
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
