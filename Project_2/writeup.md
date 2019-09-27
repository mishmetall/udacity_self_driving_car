## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]:  ./images/distortion.jpg "Distorted/Undistorted"
[image2]:  ./images/undistorted_chessboard.jpg "Distorted/Undistorted chessboard"
[image3]:  ./images/mask1.png "Mask yellow lane"
[image4]:  ./images/mask2.png "Mask white lane"
[image5]:  ./images/mask3.png "Mask black removal"
[image6]:  ./images/mask.png "Final mask"
[image7]:  ./images/pt.jpg "Perspective transform"
[image8]:  ./images/polynom.png "Polyfit"
[image9]:  ./images/result.jpg "Result"


## Pipeline
### 1. Camera Calibration

Calibration is required to be able to correctly estimate distances on image.
Camera lens introduce distortions due to the fact it is hard to mount it
precisely parallel and centered to sensor, also it is working not as a
pinhole, and that's why tangential distortions are introduced.

To calibrate camera we need to assume type of distortion and take pictures
of known pattern where we expecting to have some shapes. In project directory
we can find pictures of chessboard taken from different positions. We
know that lines on this images are straight and parallel.

Code of calibration can be found in [camera_calibrate.py](camera_calibrate.py)

First of all we need to find corners on a chessboard. We are using 6x9
chessboard pattern, so we initialize array used as a mapping to have 6x9 dimentionality.
We use `objpoints` array to store points. This points are 3d coordinates
of our chessboard corners in 3D space. We use chessboard coordinate system
for simplicity, so we just assume Z coordinate as 0 and that each corner is
1 aside of other (e.g. first has coordinate (0,0,0), second (0,1,0) and so on).

Projected corners we are going to find using
 `ret, corners = cv2.findChessboardCorners(gray, (9,6),None)` function,
which means that we are trying to find corners of 6x9 chessboard. If all were
found, ret=True.

Finally, we use `ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imshape[::-1], None, None)`
to find distortion coefficients for our camera. What this function does is
try to solve equation:

 `distortion(Prj * p_3d) = p_2d`

 `p_3d` is our point in 3D space set by `objpoints`, Prj is matrix
which project 3D point on 2D plane, it is made from camera matrix `K` and
 `R,T` matrix of rotation and vector of translation, assuming rigid body
transformation. `distortion` funtion apply non-linear transformation
from undistorted image to distorted using distoriton model.

Some examples after calibration parameters were found

before/after for chessboard:  
![Undistortion example of chessboard][image2]

before/after for real image from dataset:  
![Undistortion example of real scene][image1]

### 2. Separation lanes from background

We used S-channel from HLS image to separate yellow line from background.
To do so, we converted BGR given by OpenCV to HSL colorspace. Then, to
select line, we used Sobel filter to get binary mask of points with
high gradients.

 `mask1 = sobel_x_binary(s_channel, th=th, ksize=11)`

![example][image3]

Threshold has been chosen manually using videos given in project. 

To detect right lane (white) we used L-channel with different parameters
of Sobel filter, also tuned manually.

 `mask2 = sobel_x_binary(l_channel, th=th_grad, ksize=3)`

![example][image4]

We've observed black line on a road on challenging video, but on L-channel
it has high gradient values also. To remove it we use this technique:  

```python
 def filter_color(im, th=(0,255)):
    mask = np.zeros_like(im)

    mask[(im > th[0]) & (im <= th[1])] = 1
    return mask
    
  ...
    mask3 = filter_color(l_channel, th=(np.mean(l_channel), 255))
    mask3 = cv2.GaussianBlur(mask3*255, ksize=(3,3), sigmaX=1)
    mask3[mask3 < 255] = 0
    mask3 = np.uint8(mask3/255.)
  ...
```

Here we see that points with low luminance was considered as uninteresting
and removed, then mask was blurred to remove border pixels as well. By 
applying this filter we reduced influence of black line, however it wasn't
working all the time, as "white" line in shadow under the bridge has lower
luminance than "black" line on the light side.

![example][image5]

After all, we aggregate this filters:

 `mask[((mask2 == 1) | (mask1 == 1)) & (mask3 == 1)] = 1 `

Resulting example:  

![example][image6]

### 3. Perspective transform.

The code for my perspective transform includes a function called `warp()`, 
in the file `algorithms.py`.  It uses hardcoded trapezoid location and try
to transform it's points into bird-eye view. This coefficients were used

```python
# empirically found parameters using project video
src = np.float32([[350, 670], [581, 470], [719, 470], [1000, 670]])

min_x = int(imshape[1] * 0.20)
max_x = int(imshape[1] * 0.80)
min_y = int(imshape[0] * 0.07)
max_y = int(imshape[0] * 0.93)
dst = np.float32([[min_x, max_y], [min_x, min_y], [max_x, min_y], [max_x, max_y]])

```

This resulted in the following source and destination points:

 
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 350, 670      | 256, 669      | 
| 581, 470      | 256, 50       |
| 719, 470      | 1024, 50      |
| 1000, 670     | 1024, 669     |

I verified that my perspective transform was working as expected by drawing 
the `src` and `dst` points onto a test image and its warped counterpart 
to verify that the lines appear parallel in the warped image.

![Perspective transform][image7]

### 4. Fitting polynom

#### Initialization

We assume that lanes can be modelled as a second-order polynomial equation.
To find out it's parameters we need to find points that are most probably 
lie on a lane. At first, we use histogram technique: assume that lane is 
best seen and detected near our vehicle. We use bottom 25% pixels of the 
binary mask got on previous stage (perspective transformed). Then we find 
how many pixels lie in each column, divide image into 2 parts and by hoping
that largest number of pixels in column is a part of left or right lane.

```python
# Take a histogram of the bottom part of the image
histogram = np.sum(binary_warped[binary_warped.shape[0] *3 //4 :, :], axis=0)

# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0] // 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
``` 

Code can be found in function `find_lane_pixels`  [line_polinom.py](line_polinom.py)

Then we make more precise estimation of lane line: use `leftx_base` and
`rightx_base` as a middle point of window with some size. Find mean `x` 
coordinate of all non-zero points in it and say that this x coordinate 
belongs to lane. 

Parameters:  
```python
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 250
# Set minimum number of pixels found to recenter window
minpix = 10
```

Lane start search:
```python
if len(good_left_inds) > minpix:
    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
if len(good_right_inds) > minpix:
    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```

here `good_left_inds` and `good_right_inds` are indices of non-zero points 
that lies in windows

Then we move up with another window, starting at previous
position and refine position once again. While doing all this stuff we 
save indices of lane points.

Finally we concatenate all previously found indices and find coordinates
of points belonging to left and right lanes:

```python
# Concatenate the arrays of indices (previously was a list of lists of pixels)
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
```

Here we have indices of lines. What we need to do next is to fit polynomial
equation into it. As we have almost vertical lane, it would be better to 
find equation of x(y). Code is in `window_fit_polynomial` in [line_polinom.py](line_polinom.py)

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

#### Further processing

While vehicle won't move much from frame to frame, we can assume that 
lane also would have approximately same position and equation on adjacent
frames. We exploit this idea to find lane points faster.
Code is in `search_around_poly` ([polinom2.py](polinom2.py))

We use boundaries near previously found lane to search once again for lane
pixels. We used `margin = 50` to step left and right from previous lane
detection.

```python
left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                               left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                     left_fit[1] * nonzeroy + left_fit[
                                                                         2] + margin)))
right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                       right_fit[1] * nonzeroy + right_fit[
                                                                           2] + margin)))
```

and again we fit polynom:

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

Here is how it looks like:

![Polyfit][image8]

### 5. Measuring curvature radius and the position of the vehicle with respect to center.

#### Radius estimation

Before starting with radius search, we need to convert from pixels on
warped image to meters. We made assumption that we see 30 meters ahead 
and lanes are 3.7m wide. This, in real life, should be measured better, 
but let it be so. Based on test warped image, we assume 770px between lanes and
720 pixels ahead of us function `measure_curvature_real` in the file `algorithms.py`.

```python
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 770  # meters per pixel in x dimension
```

For radius estimation we'll use formula given in course `R = (1+(2Ax+B)^2)^1.5 / | 2A | `
and estimate it at the closest to us point of lane:

```python
y_eval = np.max(ploty) # lowest point

left_curverad = np.power(1 + (2 * left_fit_cr[0] * y_eval * config.ym_per_pix + left_fit_cr[1]) ** 2, 1.5) / np.abs(
    2 * left_fit_cr[0])
right_curverad = np.power(1 + (2 * right_fit_cr[0] * y_eval * config.ym_per_pix + right_fit_cr[1]) ** 2, 1.5) / np.abs(
    2 * right_fit_cr[0])
```

We use mean between this two to print out estimation.

#### Vehicle position estimation

To find out our position with respect to road center we will use again 
closest to us points of lanes and compare center of the image with center
between lanes (function `find_shift` in the file `algorithms.py`)

```python
# Calculate vehicle center offset in pixels
ymax = imshape[0] - 1
bottom_x_left  = left_coef[0] * (ymax**2) + left_coef[1] * ymax + left_coef[2]
bottom_x_right = right_coef[0] * (ymax**2) + right_coef[1] * ymax + right_coef[2]
car_shift = imshape[1]/2 - (bottom_x_left + bottom_x_right)/2

# Convert pixel offset to meters
car_shift *= config.xm_per_pix
```
 

### 6. Results

To visualize everithing I used lane pixels as a vertices of polygon and 
used OpenCV function to draw drivable area.
`cv2.fillPoly(out, [np.concatenate((lpts, rpts[::-1]))], color=(0, 255, 0))`
`lpts` and `rpts` are points obtained from lane equations by substituting y 
in range `[0..719]`

![Result][image9]

---

## Pipeline (video)

Here's a [link to my video result on project_video.mp4](./project_video_result.mp4)
Here's a [link to my video result on challenging_video.mp4](./challenge_video_result.mp4)

Unfortunately on harder_challenging video pipeline werked very unstable,
so we haven't provided any results on it.

---

## Discussion

Some problems with solution:
1. parameters are mostly tuned to work on this particular videos and may 
fail on other data. I can propose to find out way to normilize images 
somehow to have standard conditions. However it is hard to achieve, as 
there could be any weather/lighting conditions all over the road. Another
approach is to collect large amount of real data and train neural network
to filter lanes better. However a lot of manual annotation required.
2. If there is something similar to lane on the road (in challenging video
it wes black strip) then algorithm may fail on some conditions and not
recover back. I can suggest to invent some "reset" condition or detect
if fitting failed. It could be low amount of points or equation has some
unrealistic curvature or lanes are highly non-parallel or even intersecting.
Unfortunately I haven't done any of this in my project, but it might work.
3. Sensitive to initialization success. If we'll start our algorithm under
bad conditions, then we'll probably follow incorrectly detected lanes. 
As a solution it might work to sometimes try to reinitialize lane deteciton
based on histograms even if sequential method works well.
