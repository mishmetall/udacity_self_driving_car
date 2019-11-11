# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/run_example.png "Run example"
[image2]: ./imgs/artitecture_text.png "Architecture text"
[image3]: ./imgs/architecture.png "Architecture"
[image4]: ./imgs/1.jpg "Image 1"
[image5]: ./imgs/2.jpg "Image 2"
[image6]: ./imgs/3.jpg "Image 3"
[image7]: ./imgs/feature_maps.png "Image 4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Required Files

#### 1. Are all required files submitted?
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run.mp4 video with riding at max speed 9mph (1 lap)

### Quality of Code

#### 1. Is the code functional?

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

![run examle][image1]

#### 2. Is the code usable and readable?

The code in `model.py` uses a Python generator, but I also tried to load all data in memory. While doing this I experienced slightly larger validation error, but visually vehicle drove more stable, nevertheless both of methods works fine. 


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 90-94) 

The model includes RELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 84) and cropping (line 87). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 26). Also data augmentation was done to increase stability of driving towards center of a road. I used flipping and side cameras with 0.2 shifts in steering measurement (as in lecture material). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100). However I removed last dence layer from original CNN proposed by NVidia, this doesn't affect result much.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and images from left and right cameras as well to make model robust to deviations from central line. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional layers for feature extraction and dence layers for predicting steering position. 

My first step was to use a convolution neural network model similar to the those described in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I thought this model might be appropriate because it has been proved to be working by NVidia team. Their setup was pretty similar to our, except they used real data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I firstly used LeNet architecture with a few additional convolutional layers to play with simulator, but I found that this model does a lot of mistakes, so I switched to model mentioned before. I found that my model in just a few epochs trained to minimal validation error, but then it starts to raise. This implied that the model started to overfit. 

I used early stopping technique to use model which has least validation error to minimize overfitting (line 106).

The final step was to run the simulator to see how well the car was driving around track one. Surprisingly, there weren't a single spots where the vehicle fell off the track, hovewer it oscillates on straight road a bit (moves like a sine curve). Nevertheless, it drives well, which can be seen on video (run.mp4).

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-98) consisted of a convolution neural network with the following layers and layer sizes

![Visualization text][image2]

Here is a visualization of the architecture

![Visualization image][image3]

#### 3. Creation of the Training Set & Training Process

I used data provided in project. However I olso played with simulator to record my own tracks and to play with second environment.

Here are some images from included dataset:

![Image 1][image4]
![Image 2][image5]
![Image 3][image6]

Data contains 8036 measurements and 24108 images. I preprocessed this data by scaling to -0.5..0.5 range and cropping unnecessary area at top (sky) and bottom (parts of ego-vehicle).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was chosen by early-stopping technique with patience 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.
