# Project 3: Use Deep Learning to Clone Driving Behavior

Overview
-----
This repository is for the third project of Udacity Self Driving Car, Driving Behavior Clone. In general, we will train a deep neural network to predict for the steering angle of the car given the center image within a video stream. For the training data, Udacity provides an excellent simulator, in which we can collect data for training and test our learned models.

This `project` is based on python and keras framework. 

Bried summary of this repository:
* `README.md` : document of this repository
* model.h5 : store the final model for test, contains the structure and weights of the network trained
* drive.py : load stored model and send prediction to simulator
* video.py : create a video of images within one directory
* model.py : contains the model structure and training code
* reader.py : contains the class and preprocessing of input images

[//]: # (Image References)

[image1]: ./examples/center_origin.png "Original Center Image"
[image2]: ./examples/left_origin.png "Original Left Image"
[image3]: ./examples/right_origin.png "Original right Image"
[image4]: ./examples/placeholder.png "Grayscaling"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Recovery Image"
[image7]: ./examples/placeholder_small.png "Recovery Image"
[image8]: ./examples/placeholder_small.png "Normal Image"
[image9]: ./examples/placeholder_small.png "Flipped Image"

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Usage
For traing model, you need to create a directory, you can name it as 'data', which contains a driving log file and a directory contains all the images.
following command:
```sh
model.save(filepath)
```

For testing model, you can use the model here, model.h5, and use the drive.py provided by Udacity to test for real time simulation, also you need the simulator turned on.
```sh
python drive.py model.h5 run1
```