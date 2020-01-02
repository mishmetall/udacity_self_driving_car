# **PID controller** 

**PID controller implementation**

The goals / steps of this project are the following:
* Write code to control vehicle to follow trajectory

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1972/view) individually and describe how I addressed each point in my implementation.  

---
### Compiling

#### 1. Your code should compile.
My project includes all necessary files for compillation. To use, follow instructions in README.md. In short:

0. run install-<linux/osx/ubuntu>.sh to install dependencies
1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./pid

### Implementation

#### 1. Describe the effect each of the P, I, D components had in your implementation.

I used algorithm provided in lessons, except I added time difference, as it appeared at some point that if I launch simulator with another graphics parameters or on another computer, it stops working correctly. So I used delta time in calculating derivative and change integral calculations from sum(errors) to sum ( (error+prev_error)/2 * delta_t )

Everything can be found in PID.h and PID.cpp

### Reflection

#### 1. Describe the effect each of the P, I, D components had in your implementation.

P (proportion) affect how large would be our responce, i.e. amplitude, the larger Kp is, the faster we'll get to zero error. However, if our
system can work only in range -1..1, like in this project, we'll need to consider using small Kp or use transfer function, like tanh, which map any value to be in range -1..1. I used small Kp.

I (integral) sums up errors observed up to last point. It reduce bias if our system has one, but usually it adds more variation if it has no bias. 

D (derivative) allows us to remove oscillations by reducing control if our error is already small, this is useful in inertial systems, where 
control can't change state immediately (changing steering wheel affect our position with delay).

I've added 4 videos:
high_Kd.mp4 - We see that high Kd lead to high oscillations near zero, it is not possible in reality or will lead to uncomfortable driving
high_Ki.mp4 - high cumulative error lead to large amplitude of oscillations
high_Kp.mp4 - leads to not smooth reaction, without transfer function can lead to out of bounds control (> 1 or < -1)
optimal_params.mp4 - optimal parameters driving

#### 2. Describe how the final hyperparameters were chosen.

I've chosen initial pretty well parameters ( 0.5, 0.1, 0.3 ) and then write routine for online optimization with following algorithm:
each n_sec (e.g. 60) seconds try to vary parameter and calculate total error. If it was better than n_sec before - leave this parameter 
and go to the next one. We'll use "online" twiddle tuning parameter, as it will take huge amount of time to optimize by restarting simulator
manually hundrets of times. Sometimes new parameter lead to high variance and thus can lead to crush - then "experiment" will be terminated 
immediately. To recover from failure I added time to recover using best parameters so far and then continue to optimize again.

After this routine I get new parameters: 0.5, 0, 0.21, thich means that our simulator is unbiased and inegrel term is not needed. We've gor PD
controller.

### Simulation

#### 1. The vehicle must successfully drive a lap around the track.

I haven't observed critical situations while driving using optimized params, so they are considered to be safe.