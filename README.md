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

##### Clustering for Segmentation

Now that we've used shape attributes in the dataset to filter and segment the data, we'll move on to using other elements of our dataset such as: **color and spatial properties**

###### K-Means Clustering

###

K-Means Clustering is an appropriate clustering algorithm if you are aware of your dataspace and have a rough idea of the number of clusters.

Remember, with K-Means, we:

**1.** Choose the number of k-means
**2.** Define the convergence / termination criteria (stability of solution / number of iterations)
**3.** Select the initial centroid locations, or randomly generate them
**4.** Calculate the distance of each datapoint to each of the centroids
**5.** Assign each of the datapoints to one of the centroids(clusters) based upon closest proximity
**6.** Recompute the centroid based on the datapoints that belong to it
**7.** Loop back to **Step 4** until convergence / termination criteria is met

###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/59611b02_screen-shot-2017-07-08-at-10.48.32-am/screen-shot-2017-07-08-at-10.48.32-am.png)

###

If you are unsure of the number of clusters, it is best to use a different clustering solution! Such as DBSCAN!

###

###### DBSCAN Algorithm *(Density-Based Spatial Clustering of Applications with Noise)*
*Sometimes called Euclidean Clustering*

###

DBSCAN is a nice alternative to k-means when you don't know how many clusters to expect in your data, but you do know something about how the points should be clustered in terms of density (distance between points in a cluster).

###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/59616bad_screen-shot-2017-07-08-at-4.32.22-pm/screen-shot-2017-07-08-at-4.32.22-pm.png)

*Original data on the left and clusters identified by the DBSCAN algorithm on the right. For DBSCAN clusters, large colored points represent core cluster members, small colored points represent cluster edge members, and small black points represent outliers.*
###

DBSCAN datapoints **do not have to be spatial data; they can be color data, intensity values, or other numerical features!** This means we can cluster not only based upon proximity, but we can cluster similarly colored objects!

##### Run the VM to test our filtering and segmentation code
Downsampling, Passthrough, RANSAC plane fitting, extract inliers/outliers
```
$ roslaunch sensor_stick robot_spawn.launch
```
`segmentation.py` contains all of our filtering code. Run it with:
```
$ ./segmentation.py
```
When you click on topics in RViz, you should be able to only see this view when the `pcl_objects` topic is selected:
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/595d4565_screen-shot-2017-07-05-at-12.24.46-pm/screen-shot-2017-07-05-at-12.24.46-pm.png)
###
###

##### Object Recognition

###
###### HSV
###
HSV can help us identify objects when the lighting conditions change on an RGB image. **Hue** is the color, **Saturation** is the color intensity, and **Value** is the brightness. A conversion from RGB to HSV can be done with `OpenCV`:
```
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
```
###

###### Color Histograms

Using color historgrams, we can identify the color pattern of an object, and not be limited to spatial data. Let's use this example of the Udacity can:
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/59713eeb_screen-shot-2017-07-20-at-4.37.45-pm/screen-shot-2017-07-20-at-4.37.45-pm.png)
###
Using `numpy`, we can create a histogram for each one of the color channels. See `color_histogram.py` to see how we do this with code. The result is:
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/59714020_screen-shot-2017-07-20-at-4.43.05-pm/screen-shot-2017-07-20-at-4.43.05-pm.png)
###

This is the RGB Signature of the blue Udacity can!

###
###### Surface Normals
###
Just as we did with plotting colors, we'll now want to plot shape. This can be done by looking at the **surface normals** of a shape in aggregate. We'll also analyze them using histograms like the ones below:

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/5972224b_screen-shot-2017-07-21-at-8.47.56-am/screen-shot-2017-07-21-at-8.47.56-am.png)
###
These **surface normal** histograms correspond to the following shapes:
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/597801d9_normals-intuiition-quiz-image/normals-intuiition-quiz-image.jpg)
###
**B** represents the cube, **A** represents the sphere, and **C** represents the pyramid
###
###### Support Vector Machines (SVM)
###
"SVM" is just a funny name for a particular *supervised* machine learning algorithm that allows you to characterize the parameter space of your dataset into discrete classes. It is a **classification** technique that uses hyperplanes to delineate between discrete classes. The ideal hyperplane for each decision boundary is one that maximizes the margin, or space, from points.
###

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/59737b6e_screen-shot-2017-07-22-at-9.20.22-am/screen-shot-2017-07-22-at-9.20.22-am.png)
###
Scikit-Learn or `sklearn.svm.SVC` will help us implement the SVM algorithm. [Check this link out for documentation on scikit-learn](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
###
`svm.py` will house our SVM code and `generate_clusters.py` will help us create a random dataset.
```
svc = svm.SVC(kernel='linear').fit(X, y)
```
 The line above is the one doing the heavy lifting. The type of delineation can be changed. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. [Read more here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

We also do some **SVM Image Classification**, and this can be found under `svm_image_classifer`.

###### Recognition Exercise
###
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/59825df9_screen-shot-2017-08-02-at-4.18.53-pm/screen-shot-2017-08-02-at-4.18.53-pm.png)
###
```
$ cd ~/catkin_ws
$ roslaunch sensor_stick training.launch
```
Once the environment is up, then:
```
$ cd ~/catkin_ws
$ rosrun sensor_stick capture_features.py
```
The `capture_features.py` script should randomly place the 7 objects infront of the RGB-D camera and capture features about the items, then output `training_set.sav`






(***README IN PROGRESS***)