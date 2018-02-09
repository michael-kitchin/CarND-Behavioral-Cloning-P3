# **Behavioral Cloning** 

## Writeup

### Introduction

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network using Keras that predicts steering angles from images
* Train/validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Environment

Minimum training/execution environment is undefined.

Project was developed using the following environment:

| Category | Item        |
|----------|-------------|
| OS       | Windows 10 |
| CPU      | Intel i7/6800k |
| RAM      | 64GB |
| GPU      | nVidia GTX 1060 |
| VRAM     | 6GB |
| Storage  | SATA SSD |

An updated TensorFlow/Keras configuration was required for GPU-accelerated training/validation/execution of the model on the target OS (Windows 10). The new configuration includes TensorFlow 1.5.0, Keras 2.1.3, and supporting capabilities.

`conda` environment files have been included for this configuration, as follows:
* [environment-gpu-1.yml](environments/environment-gpu-1.yml): GPU back end 
* [environment-cpu-1.yml](environments/environment-cpu-1.yml): CPU back end

---

## Rubric Points

### [Rubric Points](https://review.udacity.com/#!/rubrics/432/view) are discussed individually with respect to the implementation.

---
### 1. Required Files

#### 1.1 Are all required files submitted?

_The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4._

See: [GitHub Repo](https://github.com/michael-kitchin/https://github.com/michael-kitchin/CarND-Behavioral-Cloning-P3).

* [model.py](model.py): 
Python script for creating/training model from simulated vehicle data log.
* [drive.py](drive.py):
Python script for executing model to control simulated vehicle.
* [model.h5](model.h5):
Serialized version of best-performing model weights.
* [video.mp4](video.mp4): 
Video of best-performing model controlling simulated vehicle through multiple laps on test track #1.

### 2. Quality of Code

#### 2.1 Is the code functional?

_The model provided can be used to successfully operate the simulation._

As shown in [video.mp4](video.mp4), the simulated vehicle will navigate test track #1 correctly/indefinitely at any speed up to the configurable maximum of ~30mph.

This is made possible by the [model.h5](model.h5) model, generated by the [model.py](model.py) script.

Supported arguments (defaults are observed, best values):

| Argument | Description | Default / Best Value | Other Attempted Values |
|:-------:|-------------|----------------------|------------------|
| `--input-file` | Data log file path. | `./input_data/driving_log.csv` | |
| `--input-dir` | Hood cam image base path. | `./input_data/IMG/` | |
| `--steering-correction` | Left/right image steering correction. | `0.1` | `0.2`, `0.01`, `0.02` |
| `--dropout-probability` | Training dropout probability. | `0.5` | `0.1`, `0.25`, `0.75` |
| `--activation-function` | Activation function alias. | `elu` | `relu`, `sigmoid`, `selu` (see: [activations](https://keras.io/activations/) for options)|
| `--loss-function` | Loss function alias. | `mse` | `logcosh` (see: [losses.py](https://keras.io/losses/) for options) |
| `--optimizer-function` | Optimizer function alias. | `adam` | `adamax`, `rmsprop` (see: [optimizers.py](https://keras.io/optimizers/) for options)|
| `--epoch-count` | Epoch count. | `10` | `5`, `6`, `15`|
| `--validation-split` | Fraction of training set to use for validation. | `0.2` | `0.1`, `0.3`, `0.4` |
| `--batch-size` | Training batch size. | `128` | `32`, `64`, `256` |
| `--shuffle-images` | Shuffle all images prior to batching/training? | `True` | `False` |
| `--shuffle-batches` | Shuffle batches during training? | `False` | `True` |

The model file generated by this script is named according to the following form:
```
model_<a>_<b>_<c>_<d>_<e>_<f>_<g>_<h>_<i>_<j>.h5
```
Where:
* `<a>` = Steering correction.
* `<b>` = Dropout probability .
* `<c>` = Activation function.
* `<d>` = Loss function.
* `<e>` = Optimizer function.
* `<f>` = Epoch count.
* `<g>` = Validation split.
* `<h>` = Batch size.
* `<i>` = Shuffle images?
* `<j>` = Shuffle batches?

Example (default):
```
model_0.1_0.5_elu_mse_adam_10_0.2_128_true_false.h5
```

Example execution (Windows):
```
(C:\Tools\Anaconda3) C:\Users\mcoyo>activate bc-project-gpu-1
(bc-project-gpu-1) C:\Users\mcoyo> cd C:\Projects\CarND-Behavioral-Cloning-P3 
(bc-project-gpu-1) C:\Projects\CarND-Behavioral-Cloning-P3>python model.py
Using TensorFlow backend.
Args: {'shuffle_images': None, 'input_file': None, 'input_dir': None, 'batch_size': None, 'dropout_probability': None, 'steering_correction': None, 'activation_function': None, 'shuffle_batches': None, 'epoch_count': None, 'loss_function': None, 'validation_split': None, 'optimizer_function': None}
Config #1 of 1: ('./input_data/driving_log.csv', './input_data/IMG/', 0.1, 0.5, 'relu', 'mse', 'sgd', 10, 0.2, 128, True, False)
Reverse:  0 Field:  0 Line:  0
Reverse:  0 Field:  0 Line:  2000
Reverse:  0 Field:  0 Line:  4000
Reverse:  0 Field:  0 Line:  6000
Reverse:  0 Field:  0 Line:  8000
Reverse:  0 Field:  1 Line:  0
Reverse:  0 Field:  1 Line:  2000
Reverse:  0 Field:  1 Line:  4000
Reverse:  0 Field:  1 Line:  6000
Reverse:  0 Field:  1 Line:  8000
Reverse:  0 Field:  2 Line:  0
Reverse:  0 Field:  2 Line:  2000
Reverse:  0 Field:  2 Line:  4000
Reverse:  0 Field:  2 Line:  6000
Reverse:  0 Field:  2 Line:  8000
Reverse:  1 Field:  0 Line:  0
Reverse:  1 Field:  0 Line:  2000
Reverse:  1 Field:  0 Line:  4000
Reverse:  1 Field:  0 Line:  6000
Reverse:  1 Field:  0 Line:  8000
Reverse:  1 Field:  1 Line:  0
Reverse:  1 Field:  1 Line:  2000
Reverse:  1 Field:  1 Line:  4000
Reverse:  1 Field:  1 Line:  6000
Reverse:  1 Field:  1 Line:  8000
Reverse:  1 Field:  2 Line:  0
Reverse:  1 Field:  2 Line:  2000
Reverse:  1 Field:  2 Line:  4000
Reverse:  1 Field:  2 Line:  6000
Reverse:  1 Field:  2 Line:  8000
Samples: 48216, Training: 38572, Validation: 9644
Epoch 1/10
2018-02-04 17:39:00.544254: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2018-02-04 17:39:00.859786: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1105] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.759
pciBusID: 0000:09:00.0
totalMemory: 6.00GiB freeMemory: 4.96GiB
2018-02-04 17:39:00.859900: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:09:00.0, compute capability: 6.1)
302/302 [==============================] - 20s 66ms/step - loss: 0.0256 - val_loss: 0.0247
Epoch 2/10
302/302 [==============================] - 17s 55ms/step - loss: 0.0243 - val_loss: 0.0220
Epoch 3/10
302/302 [==============================] - 17s 55ms/step - loss: 0.0230 - val_loss: 0.0200
Epoch 4/10
302/302 [==============================] - 17s 56ms/step - loss: 0.0218 - val_loss: 0.0188
Epoch 5/10
302/302 [==============================] - 17s 55ms/step - loss: 0.0212 - val_loss: 0.0181
Epoch 6/10
302/302 [==============================] - 17s 55ms/step - loss: 0.0204 - val_loss: 0.0174
Epoch 7/10
302/302 [==============================] - 17s 55ms/step - loss: 0.0199 - val_loss: 0.0168
Epoch 8/10
302/302 [==============================] - 17s 55ms/step - loss: 0.0192 - val_loss: 0.0162
Epoch 9/10
302/302 [==============================] - 17s 55ms/step - loss: 0.0187 - val_loss: 0.0158
Epoch 10/10
302/302 [==============================] - 17s 56ms/step - loss: 0.0183 - val_loss: 0.0154
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
Model: model_0.1_0.5_elu_mse_adam_10_0.2_128_true_false.h5

(bc-project-gpu-1) C:\Projects\CarND-Behavioral-Cloning-P3>
```

Once generated, this model may be used by the the [drive.py](drive.py) script to operate the simulated vehicle.

Supported arguments (defaults are observed, best values):

| Argument | Description | Default / Best Value |
|:-------:|-------------|----------------------|
| `<model>` | Source model file. | None (required) |
| `<image_folder>` | Destination folder for recorded data log/images. | None (optional) |
| `--random-steering-interval` | How often to inject random steering magnification (in numbers of outputs). | `0` (never) |
| `--random-steering-multiplier` | Steering output multiplier for above. | `10.0` |
| `--random-throttle-interval` | How often to inject random throttle magnification (in numbers of outputs). | `0` (never) |
| `--random-throttle-multiplier` | Throttle output multiplier for above. | `10.0` |
| `--set-speed` | Target driving speed for vehicle simulation. | `35.0` |
| `--base-steering-multiplier` | Base steering output multiplier (always applied). | `1.35` |
| `--base-throttle-multiplier` | Base throttle output multiplier (always applied). | `1.0` |

Example execution (Windows; simulator running in autonomous mode):
```
(bc-project-gpu-1) C:\Projects\CarND-Behavioral-Cloning-P3>python drive.py model_0.1_0.5_elu_mse_adam_10_0.2_128_true_false.h5
Using TensorFlow backend.
Args: Namespace(base_steering_multiplier=1.35, base_throttle_multiplier=1.0, image_folder='', model='model_0.1_0.5_elu_mse_adam_10_0.2_128_true_false.h5', random_steering_interval=0, random_steering_multiplier=10.0, random_throttle_interval=0, random_throttle_multiplier=10.0, set_speed=35.0)
2018-02-08 18:09:53.303009: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2018-02-08 18:09:53.660984: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1105] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.759
pciBusID: 0000:09:00.0
totalMemory: 6.00GiB freeMemory: 4.96GiB
2018-02-08 18:09:53.661851: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:09:00.0, compute capability: 6.1)
NOT RECORDING THIS RUN ...
(83360) wsgi starting up on http://0.0.0.0:4567
(83360) accepted ('127.0.0.1', 57678)
connect  46142e3ccbb34a09ab139a3517e49e43
-0.18174397498369219 3.4359720000000005
-0.18174397498369219 3.5033440000000002
-0.18174397498369219 3.5707160000000004
-0.17246532887220384 3.772116
-0.17246532887220384 3.8225524
[...]
```

#### 2.2 Is the code usable and readable?
         
_The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed._

Generators derived from `keras.utils.Sequence` were used to reduce GPU memory consumption during training through batching. The base class was chosen for its simplicity and (prospectively) to support parallel model training.   

The training/validation image dataset is still loaded completely into memory for pre-processing and sequence construction to reduce I/O during training (and therefore execution time). 

---

### 3. Model Architecture and Training Strategy
       
#### 3.1 Has an appropriate model architecture been employed for the task?
         
_The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model._

The model architecture is based on the nVidia DAVE2 system, described in both course material and the following:
* [https://devblogs.nvidia.com/deep-learning-self-driving-cars/](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
* [https://blogs.nvidia.com/blog/2016/05/06/self-driving-cars-3/](https://blogs.nvidia.com/blog/2016/05/06/self-driving-cars-3/)

The CNN layer structure, filter sizes, etc. may be summarized as shown:
 ![dave2-architecture-1.png](media/dave2-architecture-1.png)
 
 This design was chosen based on its inclusion of most/all architectural principles advocated in course material and established use in related, real-world applications. No effort was made to further optimize structural details of this architecture.
 
 While generally applicable to challenges addressed by this project, DAVE2 has been optimized for input images (planes) with specific dimensions (200x60x3) different from those acquired by the simulated cameras (320x160x3). 
 
 Camera images are therefore cropped to ideal dimensions prior to training/validation (in [model.py](model.py)) and operation (in [drive.py](drive.py)), serving the following purposes:
 * Restricting training to the most-consequential fractions of camera images (lower-center)
 * Optimizing input to this DAVE2-based architecture
 
 Scaling camera images to the ideal resolution was also attempted with poor results (significant aliasing, compression artifacts).     

#### 3.2 Has an attempt been made to reduce overfitting of the model?
         
_Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting._

[model.py](model.py) provides command-line options for splitting the input camera images/steering inputs into training/validation sets. By default 80% of the input data is used for training and 20% for validation.

Input image data was first normalized then augmented with the following, generated variations:
* Horizontally-flipped (reversed) images
* Left/right camera images coupled with configurable steering offsets (default: +/-0.1)

Configurable dropout layers were also employed within the model architecture to reduce overfitting effects.

#### 3.3 Have the model parameters been tuned appropriately?

_Learning rate parameters are chosen with explanation, or an Adam optimizer is used._

Several optimizing functions were experimented with, but `adam` proved the most effective/comprehensible in training results.

#### 3.4 Is the training data chosen appropriately?
         
_Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track)._

Training/verification data was limited to the set included with course material ([data.zip](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip)). 

This capability should rely on additional training/verification/testing data such as from driving the track manually or via other, differently-controlled autonomous means. 

This additional data proved impossible to acquire due to simulator usage issues on the target system (Windows 10), with vehicle steering continuously locked to the left and throttle all the way open whenever started in manual mode. Post-processing of data logs to suppress the worst effects of these issues was attempted and discontinued due to time constraints.

Therefore, despite best efforts training/verification data was limited to the set included with course material ([data.zip](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip)). No test data was further split from the input data, as this did not seem productive.  

---

### 4. Architecture and Training Documentation

[README.md](README.md) and the associated code commentary comprises this documentation. 

---

### 5. Simulation
       
#### 5.1 Is the car able to navigate correctly on test data?
         
_No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle)._

As shown in [video.mp4](video.mp4), the simulated vehicle will navigate test track #1 correctly/indefinitely at any speed up to the configurable maximum of ~30mph.

Initial results were poor at the even default configured driving speed of ~9mph in the [drive.py](drive.py) script, with the model applying steering input too soon/often when approaching curves, reliably inducing the vehicle to hold shallow (more parallel), relative angles to lane edges until running off the road in harder turns (#2 and #3, especially). 

Improved results derived from a suspicion the recorded lap had been driven faster than a consistent ~9mph. 

Were the recorded lap driven faster and curves executed using typical [turn lines](https://en.wikipedia.org/wiki/Racing_line), input images should indicate steeper (more perpendicular), relative angles to lane edges at the point significant steering input was applied. With this in mind, it was theorized when this model was (a) trained on imagery acquired at higher driving speeds but (b) executed at lower speeds, it responded an approaching curve's lanes edge sooner that ideal and applied the kind of repetitive steering input necessary to induce effects described above.

It was further theorized matching/exceeding the recorded driving speed at the fixed sampling rate would more consistently place the vehicle deeper into curves, with steeper relative angles to lane edges in imagery. If successful, this input should induce more patterns/dynamics anticipated by the model and lead to more effective steering input.

This led to extensive experimentation with different driving speeds and dataset smoothing/filtering without conclusive results, but did suggest higher driving speeds and control magnification may create opportunities for steeper relative angles to be frequent/evident enough to induce significant steering input.      

The most successful/interpretable approach to this was altering the control function in [drive.py](drive.py) to linearly magnify steering input. This simplistic, configurable approach was chosen in lieu of second-guessing model capabilities through robotic techniques such as non-linear scaling, smoothing, and limiting.

The outcome was a model/control coupling that worked well at any speed, coercing the simulated vehicle deeper into curves and exiting with harder turns. This approach may not have yielded ideal turn lines but is evidently safe and effective within this simulation. 

A negative side-effect of this approach is continuous, low-magnitude/-frequency direction changes in straight sections. This does not cascade into (e.g) uncontrollable oscillations as the model remains in control, driven by image input at all times.              
