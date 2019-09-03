# **Finding Lane Lines on the Road** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./images/original_with_roi.png "Blurred image with ROI"
[image2]: ./images/edges.png "Edges"
[image3]: ./images/masked_edges.png "Masked edges"
[image4]: ./images/hough.png "Hough lines detected"
[image5]: ./images/hough_filtered.png "hough lines filtered"
[image6]: ./images/result.png "result"

---

### Reflection

### 1. Pipeline description

My pipeline consisted of 5 steps:
 1. Resize image to test image size (960x540). It was made to simplify
 parameters tuning steps

 2. I used blur function on image to reduce noise impact. I removed
 conversion to grayscale, as Canny edge detector worked better on color
 image when trying to test on challange.mp4 dataset. Yellow lane when
 converting to gray using standard OpenCV tools almost disappears in some
 cases, being almost indistinguishable from a road

 ![Blurred image with ROI][image1]

 3. Edge detector. I used Canny edge detector with parameters taken from
 quizzes ( 50/150 ) - they worked pretty well on all three test videos

 ![Edges][image2]

 4. Region of interest filter. To reduse search space and noise I applied
 masking. Shape was hardcoded based on test images. I also considered
 that we need to take trapezoid with less height because lanes are looking
 like lines only in a close vicinity of our vehicle.

 ROI can be seen on first image

 ![Masked edges][image3]

 5. Hough transform and line search. I tuned parameters to be able to
 detect lanes and not detect noise on test images.

 ![Hough transform lines][image4]

 6. Filtering lines. I removed lines that has slope not in range 20...70
 degrees, it was made to reduse line detection noise

 7. Averagind lines. I separated lines into left and right by slope and
 then found all slopes and shifts. Then I averaged separately slopes/shifts
 for left and for right lines and then based on maximum y values of ROI
 I found border of lanes in resulting image

 ![Lines filtered][image5]

 8. Forming output. I used helper function to draw lines over initial image

 ![Result][image6]



### 2. Shortcoming of pipeline


This pipeline has different drawbacks:
- ROI and image size is hardcoded, so if image has different FOV it will
probably fail
- Slope filter works well for shadows or noise perpendicular to the road.
If shades or other noise has edges with lane-like slopes it will impact
averaging stage


### 3. Possible improvements to the pipeline

A possible improvement would be to store lane history: we can assume that
lanes has approximately same distance between each other and parallel.
So after some time we could reason of what is noise and filter it out.

Other improvement could be to involve camera information (FOV, focal distance)
to remove hardcoded parameters of ROI selection.

Also we can detect horizon, also for smarter ROI selection.
