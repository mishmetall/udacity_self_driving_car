# **Traffic Sign Recognition**

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/examples_signs.png "Visualization"
[image2]: ./images/histogram.png "histogram"
[image3]: ./images/new_data.png "New data"
[image4]: ./images/prediction_top5.png "Prediction top 5"
[image5]: ./images/conv1_filters.png "Conv 1 filter"
[image6]: ./images/cyclist_dataset.png "Cyclist in dataset"
[image7]: ./images/misclassification_frequency_test.png "Misclassification frequenct"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mishmetall/udacity_self_driving_car/blob/master/Project_3/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3) - color images
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![Classes examaples][image1]

Here we see examples of random image for different classes. We see here that we
can encounter different lighting conditions, point of views (not only straight,
but also view from some angles). However we see also that all images has sign
approximately same size all over dataset and centered.

Distribution over images look like this:

![histogram][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I implemented LeNet architecture and tried if everything would
work good with plain data without preprocessing except scaling to 0..1 range. I've
also tried to normilize using (pixel-128) / 128 formula, but it worked slightly worth.

It is possible to generate additional data, but without it (I later see) network
works well either.

However, if it were generated, I would have chosen such modifications:

 1. Rotations up to 10-15 degrees (maybe even more, but take in mind that some
    signs can become other, like keep left and go left).
 2. Random crop [1..5] pixels from sides
 3. Changing lightness of image
 4. Plane transformation to imitate angle of view
 5. Scaling 0.8...1.2
 6. Blurring

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tested two architectures: LeNet and LeNet modified extended with dropout layers

Original LeNet:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten				|												|
| Fully connected		| 400x320 									|
| RELU					|												|
| Fully connected		| 320x240 									|
| RELU					|												|
| Fully connected		| 240x43 									|
| Softmax				|  									|

Better architecture:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Dropout	      	|  				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Dropout	      	|  				|
| Flatten				|												|
| Fully connected		| 400x320 									|
| RELU					|												|
| Dropout	      	|  				|
| Fully connected		| 320x240 									|
| RELU					|												|
| Dropout	      	|  				|
| Fully connected		| 240x43 									|
| Softmax				|  									|

Dropout plays role of regularization by zeroing randomly output of layer. It makes
network be more robust to noise and force it to extract more abstract features
instead of memorizing input data.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdaM optimizer from Tensorflow. I tested also AdaGrad,
but it optimize model much slower. Batch size was chosen to be 200 (usually it affect only train time, 
smaller batch size converges faster at the beginning, but later tend to make inaccurate gradient prediction,
for GPU it's better to choose size that fits memory, as transferring batches is expensive operation). We used 
early stopping technique (5 epochs without accuracy improvement terminates training) to train model. 
Learning rate was empirically considered to be 0.001. Validation set was used to prevent overfitting: the more
we train, the more model will tend to memorize train data. Not using fixed amount of epochs means that we also
eliminate risk of underfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.158%
* validation set accuracy of 97.868%
* test set accuracy of 97.641%

it was calculated as:
```
# Accuracy operation
self.prediction = tf.argmax(self.model, 1)  # Fully connected layer output
self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.true, 1))
self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
```

I've tried to make different preprocessing to input data (histogram qualization,
  grayscaling). But it only slows down solution and does not improve accuracy anyhow.
Also I tried to change size of convolution layers, adding more of them, however
simple change to 3x3 instead of 5x5 and adding one more convolution layer (also 3x3)
was enough to significantly improve model's
performance. Dropout was also useful, 0.7 drop rate was chosen to work good. As
we see above, model doesn't show signs of overfitting.

CNN architecture is good because it allows to focus on local features. First few
layers extract this features which are then used by fully connected layers to make
a decision. LeNet was one of them, better architectures like VGG, ImageNet, ResNet etc.
may work better due to some other improvements like skip connections, multiheaded
topology and so on.

In practice, output of classifier should be filtered over several frames, also
technique by enabling test-time dropout and then averaging prediction could be used
to minimize accidental misclassification or measure prediction uncertainty (see
  Yarin Gal thesis for more information about prediction uncertainty estimation).

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used Google Street View data from Berlin and take several screenshots from there.

![New images][image3]

First image might be easy to confuse with go straight or right, but it has good quality,
so I expect it to be classified correctly. Second image is also pretty easy, but it is 
somewhat similar to stop sign. Third sign is not presented in dataset in exact this form, 
but can be found on a road, so let's see what CNN might deside. 4th sign and other speed
signs can be difficult because numbers are hard to distinguish on low resolution images.
Last sign has low illumination, CNN might confuse it with other triangular signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![Prediction][image4]

As we see visually, everything is correct except bycicle crossing sign.
Bicycles crossing sign in dataset is defferent from new sign I found on 
street views. We see that it has a man and network confused it with sign 
"Children crossing".

Let's see how it looks in dataset:

![Dataset cyclist][image6]

We can also analize which signs are more often misclassified.

![Misclassification frequency][image7]

We see that some signs are misclassified more often, and it would be a
good idea to add more data for them.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 44th cell of the Ipython notebook.

For images network was pretty confident in its prediction, as expected. Cyclist is
wrongly detected as children crossing due to presence of man, speed limit is not
very confident due to low resolution and similarity between 8 and 5. As for turn
right ahead: it seems like CNN has some problems with catching orientation information. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is visualization of first convolution layer activations over batch of images

![Conv 1 filters][image5]

We can make assumption that network tries to detect circle (map 4, 16, 20), line
(14, 21, 22, 19), numbers (7, 11, 27 etc). Other feature maps don't activate at
all using this sign, it might activate well on other (e.g. with pedestrians or animals).

Visualization of other layers doesn't give meaningful visual information.
