import random
import argparse
import csv
import cv2
import scipy.ndimage
from itertools import product
from sklearn.utils import shuffle

from util.model import *


def check_arg(arg, default):
    if arg is None:
        return [default]
    else:
        return arg


parser = argparse.ArgumentParser()

parser.add_argument('--input-file', action='append')
parser.add_argument('--input-dir', action='append')
parser.add_argument('--steering-correction', action='append', type=float)
parser.add_argument('--dropout-probability', action='append', type=float)
parser.add_argument('--activation-function', action='append')
parser.add_argument('--loss-function', action='append')
parser.add_argument('--optimizer-function', action='append')
parser.add_argument('--epoch-count', action='append', type=int)
parser.add_argument('--validation-split', action='append', type=float)
parser.add_argument('--batch-size', action='append', type=int)
parser.add_argument('--shuffle-images', action='append', type=bool)
parser.add_argument('--shuffle-batches', action='append', type=bool)
args = vars(parser.parse_args())
print ("Args: {}".format(args))

all_config = list(product(check_arg(args['input_file'], './input_data/driving_log.csv'),
                          check_arg(args['input_dir'], './input_data/IMG/'),
                          check_arg(args['steering_correction'], 0.1),
                          check_arg(args['dropout_probability'], 0.5),
                          check_arg(args['activation_function'], 'elu'),
                          check_arg(args['loss_function'], 'mse'),
                          check_arg(args['optimizer_function'], 'adam'),
                          check_arg(args['epoch_count'], 10),
                          check_arg(args['validation_split'], 0.2),
                          check_arg(args['batch_size'], 128),
                          check_arg(args['shuffle_images'], True),
                          check_arg(args['shuffle_batches'], False)))
random.shuffle(all_config)

config_ctr = 0
for run_config in all_config:
    print ("Config #{} of {}: {}".format((config_ctr + 1), len(all_config), run_config))
    input_file = run_config[0]
    input_dir = run_config[1]
    steering_correction = run_config[2]
    dropout_probability = run_config[3]
    activation_function = run_config[4]
    loss_function = run_config[5]
    optimizer_function = run_config[6]
    epoch_count = run_config[7]
    validation_split = run_config[8]
    batch_size = run_config[9]
    shuffle_images = run_config[10]
    shuffle_batches = run_config[11]

    lines = []
    with open(input_file) as csvfile:
        reader = csv.reader(csvfile)
        line_ctr = 0
        for line in reader:
            if line_ctr > 0:
                lines.append(line)
            line_ctr = line_ctr + 1

    images = []
    measurements = []
    for line_config in product(range(2), range(3)):
        rev_ctr = line_config[0]
        field_ctr = line_config[1]
        line_ctr = 0
        for line in lines:
            if line_ctr % 1000 == 0:
                print ("Reverse: ", rev_ctr, "Field: ", field_ctr, "Line: ", line_ctr)
            field_path = (input_dir + line[field_ctr].split('/')[-1])
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
    if shuffle_images:
        X_input, y_input = shuffle(X_input, y_input)

    validation_index = int(len(X_input) * (1.0 - validation_split))
    print ("Samples: {}, Training: {}, Validation: {}"
           .format(len(X_input), validation_index, len(X_input) - validation_index))

    X_train = np.array(X_input[0:validation_index])
    y_train = np.array(y_input[0:validation_index])
    X_validation = np.array(X_input[validation_index:len(X_input)])
    y_validation = np.array(y_input[validation_index:len(y_input)])

    train_generator = SampleSequence(X_train, y_train, batch_size)
    validation_generator = SampleSequence(X_validation, y_validation, batch_size)

    model = build_model(activation_function, dropout_probability)
    model.compile(loss=loss_function, optimizer=optimizer_function)
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator),
                        shuffle=shuffle_batches,
                        epochs=epoch_count)

    model.save('model_{}.h5'.format('_'.join([str(x) for x in run_config[2:]])).lower())
    config_ctr = config_ctr + 1
