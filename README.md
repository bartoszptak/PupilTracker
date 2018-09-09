## PupilTracker
Follow the eye with the webcam. Tracking the pupil with the help of neural networks.  
A project made in one month as part of a student internship at the Poznań University of Technology.

[![Pupil Tracker YouTube](https://img.youtube.com/vi/kZHMxFYi1rI/0.jpg)](https://www.youtube.com/watch?v=kZHMxFYi1rI)

### Requirements
* webcam (we use LOGITECH HD PRO C920)
* python >= 3.6.4
* keras >= 2.1.6
* tensorflow/tensorflow-gpu >=1.9
* numpy >= 1.11.3
* opencv-python >= 3.4.1
* dlib >= 19.15.0
* imutils >= 0.4.6

### Modules
* `pupil.py` - An application that is designed to detect eyes and save them as a photo. It is needed to create a data set.
* `click.py` - Designed to indicate pupils and cornea on each photo of the dataset. Photographs and coordinates are normalized and saved to one database.
* `Train.ipynb` - Convolutional neural network, data loading, preprocessing, training a model
* `Train_with_transfer_learning.ipynb` - Attempt to use 'transfer learning', 'InceptionV3' algorithm, 'imagenet' weights
* `main.py` - The main program of the project. 

### Credits
* [Damian Szkudlarek](https://github.com/szkudlarekdamian)
* [Bartosz Ptak](https://github.com/bartoszptak)
