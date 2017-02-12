# Project 3: Use Deep Learning to Clone Driving Behavior

Overview
-----
This repository is for the third project of Udacity Self Driving Car, Driving Behavior Clone. In general, we will train a deep neural network to predict for the steering angle of the car given the center image within a video stream. For the training data, Udacity provides an excellent simulator, in which we can collect data for training and test our learned models.

This project is based on python and keras framework. 

Brief summary of this repository:
*`README.md` : document of this repository
*`model.h5` : store the final model for test, contains the structure and weights of the network trained
*`drive.py` : load stored model and send prediction to simulator
*`video.py` : create a video of images within one directory
*`model.py` : contains the model structure and training code
*`reader.py`  : contains the class and preprocessing of input images

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

###PreProcessing Strategy

For behavioral cloning task, data plays an essential rule that it can directly influence the result of out model. First, we must ensure that the label for each data is correct, only in this way can the network learns something useful. For this particular project, it is natural to treat it as a regression problem (we may also transfer it to a classification problem). The data is a 2D RGB image, the label for that image data is the steering angle at the same timestamp. But, usually it is very hard to collect smooth steering angles because of the data collecting process, so one thing we can do to alleviate this is to do exponential smoothing for the training steer angles.

![original cneter][image1] ![alt text][image2] ![alt text][image3]

There a lot of thing we can do to deal with the input images. The original image shape is 320x160, but not all of them are useful for this project, we can safely drop the upper part of the image because for predicting steer angles, we are actually look at the lane and board of the road, the upper part usually contains sky and other noisy information, which are harmful for the training. The image after croppedis of size 50x
