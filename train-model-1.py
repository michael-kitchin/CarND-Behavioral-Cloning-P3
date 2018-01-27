import csv
import cv2
import numpy as np

lines = []
with open('./input_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
line_ctr = 0
for line in lines:
	if line_ctr > 0:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = './input_data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)
	line_ctr = line_ctr + 1

X_train = np.array(images)
print ('X_train',len(X_train))
y_train = np.array(measurements)
print ('y_train',len(y_train))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
# model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(120))
# model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')
