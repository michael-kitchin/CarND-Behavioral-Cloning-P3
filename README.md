# **Behavioral Cloning** 

## Writeup

---

## Rubric Points

### See: [Rubric Points](https://review.udacity.com/#!/rubrics/432/view)

---
### 1. Required Files

#### 1.1 Are all required files submitted?

_The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4._

See: [GitHub Repo](https://github.com/michael-kitchin/https://github.com/michael-kitchin/CarND-Behavioral-Cloning-P3)

* [model.py](https://github.com/michael-kitchin/CarND-Behavioral-Cloning-P3/blob/master/model.py): 
Python script for creating/training the model from a simulated vehicle data log.
* [drive.py](https://github.com/michael-kitchin/CarND-Behavioral-Cloning-P3/blob/master/drive.py):
Python script for executing the model to control a simulated vehicle.
* [model.h5](https://github.com/michael-kitchin/CarND-Behavioral-Cloning-P3/blob/master/model.h5):
Serialized version of the best-performing model weights.
* [video.mp4](https://github.com/michael-kitchin/CarND-Behavioral-Cloning-P3/blob/master/video.mp4): 
Video of the best-peforming model controlling the simulated vehicle through several laps on a test track.

![video.mp4](https://github.com/michael-kitchin/CarND-Behavioral-Cloning-P3/blob/master/video.mp4)    

---

### 2. Quality of Code

#### 2.1 Is the code functional?

_The model provided can be used to successfully operate the simulation._

#### 2.2 Is the code usable and readable?
         
_The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed._

---

### 3. Model Architecture and Training Strategy
       
#### 3.1 Has an appropriate model architecture been employed for the task?
         
_The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model._

#### 3.2 Has an attempt been made to reduce overfitting of the model?
         
_Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting._

#### 3.3 Have the model parameters been tuned appropriately?

_Learning rate parameters are chosen with explanation, or an Adam optimizer is used._

#### 3.4 Is the training data chosen appropriately?
         
_Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track)._

---

### 4. Architecture and Training Documentation
       
#### 4.1 Is the solution design documented?
         
_The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem._

#### 4.2 Is the model architecture documented?
         
_The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged._

#### 4.3 Is the creation of the training dataset and training process documented?
         
_The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included._

---

### 5. Simulation
       
#### 5.1 Is the car able to navigate correctly on test data?
         
_No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle)._
