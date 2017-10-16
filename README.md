[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Project 3: 3D Perception in a Cluttered Space
##### Udacity Robotics Nanodegree
###### October 2017

###
###

### Overview

###### The goal of this project is to program a PR2 to identify and acquire target objects from a cluttered space and place them in a bin. The project is based on the 'Stow Task' portion of Amazon's Robotics Challenge.
###
###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/597a8731_screen-shot-2017-07-27-at-5.25.02-pm/screen-shot-2017-07-27-at-5.25.02-pm.png)

##### Specifically, the objectives of this project are to:

**1.** 
###
**2.** Maneuver the PR2
###
**3.** Grasp the object
###
**4.** 
###
**5.** 
###
**6.** Sucessfully place the object in the destination bin
###
###

If you'd like to watch the PR2 in action, click [**here.**](https://youtu.be/cTCJSNjTdo0)

Check out the Amazon Robotics Challenge [**here.**](https://www.amazonrobotics.com/#/roboticschallenge)

For those interested, [here is a great video on a highly accurate grasping robot.](https://youtu.be/MtDMn1tc_Q4)
And the dataset that powers it is now [**free and open to the public!**](https://spectrum.ieee.org/automaton/robotics/robotics-software/uc-berkeley-releases-massive-dexnet-20-dataset)

###

We operate PR2 through **ROS Kinetic** (ran through Linux Ubuntu 16.0) and commands are written in **Python**.

The code driving this project and interacting with ROS can be found at `IK_server.py`

*Quick note on naming convention:* `THIS_IS_A_CONSTANT` *and* `thisIsAVariable`

This **README** is broken into the following sections: **Environment Setup, Code Analysis, and Debugging**

###
###
###

### Environment Setup

[Filtering and Segmentation Exercises](https://github.com/udacity/RoboND-Perception-Exercises)
Using the `python-pcl` library, which can be invoked using `import pcl`

[Point Cloud Library](http://www.pointclouds.org/)



### Code Analysis

###

##### OpenCV

Although we won't be using OpenCV to calibrate our camera(we'll be using a ROS package), the process for doing so with OpenCV is outlined here:

**1.** Use `cv2.findChessboardCorners()` to find corners in chessboard images and aggregate arrays of image points (2D image plane points) and object points (3D world points) .
**2.** Use the OpenCV function `cv2.calibrateCamera()` to compute the calibration matrices and distortion coefficients.
**3.** Use `cv2.undistort()` to undistort a test image.

OpenCV functions for calibrating camera: `findChessboardCorners()` and `drawChessboardCorners()`

###

##### Filtering

###

We'll harness some filters from the [Point Cloud Library](http://www.pointclouds.org/)

###### VoxelGrid Downsampling Filter

###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/595ab978_screen-shot-2017-07-03-at-2.38.41-pm/screen-shot-2017-07-03-at-2.38.41-pm.png)

###

Downsampling is used to decrease the density of the pointcloud that is output from the RGB-D camera. This is done because the very feature rich pointclouds can be quite computationally expensive.

`RANSAC.py` contains the Voxel Downsampling Code.

###### Pass Through Filtering

###

We'll use Pass Through Filtering to trim down our point cloud space along specified axes, in order to decrease the sample size. We will allow a specific region to *Pass Through*. This is called the *Region of Interest*.

###

###### RANSAC Plane Fitting

###

We can model the table in our dataset as a plane, and remove it from the pointcloud using `Random Sample Consensus` or `RANSAC` algorithm

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/595d3ec9_screen-shot-2017-07-05-at-12.31.15-pm/screen-shot-2017-07-05-at-12.31.15-pm.png)


###### Extracting Indices - Inliers and Outliers

###

Inliers
`extracted_inliers = cloud_filtered.extract(inliers, negative=False)`

###
Outliers
`extracted_outliers = cloud_filtered.extract(inliers, negative=True)`

###

###### Outlier Removal Filter

###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/595a5ebf_statistical-outlier-removal-cropped/statistical-outlier-removal-cropped.png)

###

Used to statistically remove noise from the image.

```
# Much like the previous filters, we start by creating a filter object: 
outlier_filter = cloud_filtered.make_statistical_outlier_filter()

# Set the number of neighboring points to analyze for any given point
outlier_filter.set_mean_k(50)

# Set threshold scale factor
x = 1.0

# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
outlier_filter.set_std_dev_mul_thresh(x)

# Finally call the filter function for magic
cloud_filtered = outlier_filter.filter()
```

###
###

##### Clustering

Now that we've used shape attributes in the dataset to filter and segment the data, we'll move on to using other elements of our dataset such as: **color and spatial properties**

###### K-Means Clustering

###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/59611b02_screen-shot-2017-07-08-at-10.48.32-am/screen-shot-2017-07-08-at-10.48.32-am.png)

###


(***README IN PROGRESS***)