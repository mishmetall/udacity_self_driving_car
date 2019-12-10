# **Extended Kalman Filter** 

**Extended Kalman Filter**

The goals / steps of this project are the following:
* Write code to estimate position and velocity of vehicle given noisy measurements from laser and RADAR


[//]: # (Image References)

[image1]: ./imgs/trajectory.png "Trajectory"
[image2]: ./imgs/artitecture_text.png "Architecture text"
[image3]: ./imgs/architecture.png "Architecture"
[image4]: ./imgs/1.jpg "Image 1"
[image5]: ./imgs/2.jpg "Image 2"
[image6]: ./imgs/3.jpg "Image 3"
[image7]: ./imgs/feature_maps.png "Feature maps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1962/view) individually and describe how I addressed each point in my implementation.  

---
### Compiling

#### 1. Your code should compile.
My project includes all necessary files for compillation. To use, follow instructions in README.md. In short:

0. run install-<linux/osx/ubuntu>.sh to install dependencies
1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

### Accuracy

#### 1. px, py, vx, vy output coordinates must have an RMSE <= [.11, .11, 0.52, 0.52] when using the file: "obj_pose-laser-radar-synthetic-input.txt" which is the same data file the simulator uses for Dataset 1.

My project met requirements with final RMSE: [0.097, 0.085, 0.451, 0.440]

I used term2 simulator to get this result. Green triangles are predicted positions

![Trajectory][image1]

### Follows the Correct Algorithm

#### 1. Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons.

My algorithm follow scheme predict-update for EKF. 
Predict step can be found in kalman_filter.cpp lines 29-33.
Update Kalman Filter (for laser sensor, linear case): kalman_filter.cpp lines 35-49.
UpdateEKF Kalman Filter (for radar sensor, non-linear case): kalman_filter.cpp lines 51-80.

also, for EKF we used normalization of error for angle (as we can't simply subtract angles, we need to make them keep in range -pi..pi):
tools.cpp 82-93

Updates of matrices can be found in FusionEKF.cpp ProcessMeasurement function

#### 2. Your Kalman Filter algorithm handles the first measurements appropriately.

The code for handling first measurement can be found in FusionEKF.cpp lines 66-108.

#### 3/4. Your Kalman Filter algorithm first predicts then updates. Your Kalman Filter can handle radar and lidar measurements.

This is done in lines FusionEKF.cpp lines 136 for Predict and 145/148 for Update.

### Code Efficiency

#### 1. Your algorithm should avoid unnecessary calculations.

I tried to optimize code as much as possible by using temporary variable to reduce same computations for matrix updates.
