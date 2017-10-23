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

The code driving this project and interacting with ROS can be found at `project_template.py`

*Quick note on naming convention:* `THIS_IS_A_CONSTANT` *and* `thisIsAVariable`

This **README** is broken into the following sections: **Environment Setup, Code Analysis, and Debugging**

###
###
###

### Environment Setup
If you don't already have the directory made:
```sh
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```
Then, clone the project repo:
```sh
$ cd ~/catkin_ws/src
$ git clone https://github.com/udacity/RoboND-Perception-Project.git
```
Install missing dependencies:
```sh
$ cd ~/catkin_ws
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
```
Build it:
```sh
$ cd ~/catkin_ws
$ catkin_make
```
Add the following to your `.bashrc` file:
```
export GAZEBO_MODEL_PATH=~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/models:$GAZEBO_MODEL_PATH
```
And if it's not there already, also add this to `.bashrc`:
```
source ~/catkin_ws/devel/setup.bash
```


[Filtering and Segmentation Exercises](https://github.com/udacity/RoboND-Perception-Exercises)
Using the `python-pcl` library, which can be invoked using `import pcl`

[Point Cloud Library](http://www.pointclouds.org/)

We'll be running the code through our linux distribution of ROS.

`Robotic VM V2.0.1` is the directory that contains the Linux Boot image:`Ubuntu 64-bit Robo V2.0.1.ova`

If prompted, the Linux system password is `robo-nd`

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


Much like the previous filters, we start by creating a filter object: 
```
outlier_filter = cloud_filtered.make_statistical_outlier_filter()
```
Set the number of neighboring points to analyze for any given point
```
outlier_filter.set_mean_k(50)
```
Set threshold scale factor
```
x = 1.0
```
Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
```
outlier_filter.set_std_dev_mul_thresh(x)
```
Finally call the filter function for magic
```
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
```sh
$ roslaunch sensor_stick robot_spawn.launch
```
`segmentation.py` contains all of our filtering code. Run it with:
```sh
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
```py
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
```py
svc = svm.SVC(kernel='linear').fit(X, y)
```
 The line above is the one doing the heavy lifting. The type of delineation can be changed. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. [Read more here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

We also do some **SVM Image Classification**, and this can be found under `svm_image_classifer`.

###### Recognition Exercise - Combining what we have learned!
###
This is a very good(almost perfect) confusion matrix. Initially, ours will not come out this good.
![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/59825df9_screen-shot-2017-08-02-at-4.18.53-pm/screen-shot-2017-08-02-at-4.18.53-pm.png)
###
```sh
$ cd ~/catkin_ws
$ roslaunch sensor_stick training.launch
```
Once the environment is up, then:
```sh
$ cd ~/catkin_ws
$ rosrun sensor_stick capture_features.py
```
The `capture_features.py` script should randomly place the 7 objects infront of the RGB-D camera and capture features about the items, then output `training_set.sav`

Then, to generate the confusion matrix:
```sh
$ rosrun sensor_stick train_svm.py
```
As you can see, this is not a good confusion matrix. That's because the functions `compute_color_histograms()` and `compute_normal_histograms()`, in the file `features.py` is not appropriately filled out. 
###
You can find `features.py` in the directory:
```
~/catkin_ws/src/sensor_stick/src/sensor_stick/
```
First up, we'll alter the `compute_color_histograms()` function, which will do RGB color analysis on each point of the point cloud.
```py
def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    nbins=32
    bins_range=(0,256)
    
    # Compute histograms
    r_hist = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    g_hist = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    b_hist = np.histogram(channel_3_vals, bins=nbins, range=bins_range)
    
    # Extract the features
    # Concatenate and normalize the histograms
    hist_features = np.concatenate((r_hist[0], g_hist[0], b_hist[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)  

    return normed_features 
```
###
###
Next, we'll add the histogram, compute features, concatenate them, and normalize them for the **surface normals** function `compute_normal_histograms`:

```py
def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])


    nbins=32
    bins_range=(0,256)
    # TODO: Compute histograms of normal values (just like with color)

    # Compute histograms
    x_hist = np.histogram(norm_x_vals, bins=nbins, range=bins_range)
    y_hist = np.histogram(norm_y_vals, bins=nbins, range=bins_range)
    z_hist = np.histogram(norm_z_vals, bins=nbins, range=bins_range)
    
    # TODO: Concatenate and normalize the histograms
    # Extract the features
    # Concatenate and normalize the histograms
    hist_features = np.concatenate((x_hist[0], y_hist[0], z_hist[0])).astype(np.float64)
    normed_features = hist_features / np.sum(hist_features)  

    return normed_features
```
###
Now, we can relaunch the Gazebo environment, re-run the training and feature capture, then generate the confusion matrix:
```sh
$ cd ~/catkin_ws
$ roslaunch sensor_stick training.launch
```
Once the environment is up, then:
```sh
$ cd ~/catkin_ws
$ rosrun sensor_stick capture_features.py
```
Give it a minute or so to go through all 7 objects. Then: 
```sh
$ rosrun sensor_stick train_svm.py
```
The outputted confusion matrix should be better than the previous one. But there are still strategies to improve:
**- Convert RGB to HSV**
**- Compute features for a larger set of random orientations of the objects**
**- Try different binning schemes with the histogram(32,64, etc)**
**- Modify the SVM parameters(kernel, regularization, etc)**
###

To modify how many times each object is spawned randomly, look for the for loop in `capture_features.py` that begins with for i in range(5):. Increase this value to increase the number of times you capture features for each object.

To use HSV, find the line in `capture_features.py` where you're calling `compute_color_histograms()` and change the flag to `using_hsv=True`.

To mess with the SVM parameters, open up `train_svm.py` and find where you're defining your classifier. Check out the sklearn.svm docs to see what your options are there.

After setting `using_hsv=true` and `bins=32`(which seems to be a sweet spot), I began playing with the other two features, **number of random orientations, and SVM kernel,** in order to improve the accuracy of my object classifier. Note, I didn't use poly, as it gave very low accuracy (17%).

My results were as follows, with `using_hsv=true` and `bins=32`:
- **64%** with SVM kernel=linear, orientations=7
- **64%** with SVM kernel=rbf, orientations=10
- **74%** with SVM kernel=linear, orientations=10
- **79%** with SVM kernel=linear, orientations=20
- **82%** with SVM kernel=sigmoid, orientations=20
- **84%** with SVM kernel=rbf, orientations=20

It seems as though our accuracy is improving with orientations increasing--which makes logical sense as we increase the sample size of the training set, our algorithm gets better. Let's continue increasing orientations exposed to the camera, to `40`:

- **88%** with SVM kernel=sigmoid, orientations=40
- **89%** with SVM kernel=linear, orientations=40
- **90%** with SVM kernel=rbf, orientations=40

And at `80` orientations per object:
- **89%** with SVM kernel=sigmoid, orientations=80
- **90%** with SVM kernel=rbf, orientations=80
- **95%** with SVM kernel=linear, orientations=80

Make sure you are in the directory where `model.sav` is located!
```
$ roslaunch sensor_stick robot_spawn.launch
$ ./object_recognition.py
```
###
###
###
###
###
#### Putting it all together for our Project's Perception Pipeline

In the previous exercises, we have built out much of our perception pipeline. Building on top of `object_recognition.py` we'll just have to add a new publisher for the RGB-D camera data, add noise filtering, and a few other items. Let's get started!

##### Filter and Segment the New Scene
Since we have a new table environment, not all of our filters will be effective. I'm going to create a new file to replace `segmentation.py` that will help us filter and segment the new environment--I'll call it `pr2_segmentation.py`. Most of the code from `segmentaion.py` can just be copied over, but note the differences in the table height. This means we'll have to adjust the ***pass through filtering*** Z-axis minimum and maximum values. 

For `axis-min` I'll try `0.2`(meters) and `1.1`(meters) for `axis-max`.

We'll also want to publish our RGB-D camera data to a `ROS-topic` named `/pr2/world/points`. Following the syntax of other publishers/subscribers should help us out with this.

Also, we'll have to deal with actual noise in this scene, so we'll apply what we learned earlier on and add in the statistical outlier remover to `pr2_segmentation.py` with:
```py
outlier_filter = cloud_filtered.make_statistical_outlier_filter()
outlier_filter.set_mean_k(50)
x = 1.0
outlier_filter.set_std_dev_mul_thresh(x)
cloud_filtered = outlier_filter.filter()
```
###

This should leave `pr2_segmentation.py` just about ready to go, but to verify, we have the following items in the `pcl_callback(pcl_msg)` function:
- **Convert ROS message to PCL** (goes from PC2 data to PCL pointXYZRGB)
- **Voxel Grid Downsampling** (essentially reduce the resolution of our Point Cloud)
- **Statistical Outlier Removal Filter** (gets rid of points that aren't apart of a group, determined by proximity)
- **Pass Through Filtering** (a vertical cropping along the Z-axis that cuts down our Point Cloud)
- **RANSAC Plane Segmentation** (helps find a plane - in this case the table)
- **Extract Inliers and Outliers** (Create two point clouds from the RANSAC analysis)
- **Euclidean Clustering/DBSCAN** (Calculates the distance to other points, and based on thresholds of proximity, combines groups of points into clusters that we assume are objects)
- **Cluster Mask** (color each object differently so they are easily distinguished from one another)

We'll then want to make sure this new point cloud is published with:
```py
pcl_cluster_pub = rospy.Publisher("/pcl_world", PointCloud2, queue_size=1)
```
###
To test everything out, open a new terminal in the ROS system, and type:
```sh
$ cd ~/catkin_ws/
$ roslaunch pr2_robot pick_place_project.launch
```
Give the PR2 a moment to visualize and map its surroundings. Once it's finished, type:
```sh
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/
$ ./pr2_segmentation.py
```
You should now be able to go into RViz and click on the new ROS topics:`pcl_world` which is our filtered, segmented, and colored group of objects; and `pr2/world/points` which is the data coming from our PR2's RGB-D camera.

With this done, and once you are happy with the results, you can transfer the code. You'll find `project_template.py` in the `/pr2_robot/scripts` directory. This will be our new location for our **primary project code**. Port in the code from `pr2_segmentation.py`  and`object_recognition.py` into the respective `TO DOs`.
###
##### Capture Features of the New Objects
Just like we did in the lecture exercise, we need to capture the features of the objects that we will need to find in the project (essentially photograph them with the RGB-D camera from many different angles). 

There will be 3 different "worlds". These can be found in the `config` directory of `pr2_robot`, and are named `pick_list_1.yaml`, `pick_list_2.yaml`, and `pick_list_3.yaml`.

Pick out the models from there (should be 8 models)and put their names into the `models` dictionary of `capture_features.py`. Then, set the number of random orientations each model will be exposed to in the `for loop`. I used `80` for higher accuracy. Lastly, change the very last line of the script to make sure my new feature set is not confused with previous ones. Set it to: `training_set_worlds123.sav`

We're now ready to capture our objects' features. Launch our handy `sensor_stick` Gazebo environment with:
```sh
$ roslaunch sensor_stick training.launch
```
Now, change the directory to where you want the feature set `training_set_worlds123.sav` to save. I put mine here:
```sh
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/
```
Then, run the feature capture script:
```sh
rosrun sensor_stick capture_features.py
```
This will take some time. Mine took ~20 minutes. 

Once finished, you will have your features saved in that directory. We'll be using them in the next step when we train our classifier!

###
##### Train the SVM Classifier
We're now going to train our classifier on the features we've extracted from our objects. `train_svm.py` is going to do this for us, and is located where your directory should still be pointing:
```sh
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/
```
Open `train_svm.py` and make sure line ~39 or so is loading the correct feature data we just produced:
```py
training_set = pickle.load(open('training_set_worlds123.sav', 'rb'))
```
Then, we're going to check our SVC kernel is set to `linear`:
```py
clf = svm.SVC(kernel='linear')
```
`rbf` and `sigmoid` also work well, but with 80 iterations, I've found linear to have the highest accuracy (93%).

Now, from this same directory, run the script:
```sh
$ rosrunh sensor_stick train_svm.py
```
Two confusion matrices will be generated--one will be a normalized version. Prior to any training, I had a very confused confusion matrix that looked like this:
###
![](https://github.com/chriswernst/Perception-Udacity-RoboticsND-Project3/blob/master/images/initial_confusion_matrix.png?raw=true)

###
***If you did the above steps correctly, it should look closer to this:***
###
###
![](https://github.com/chriswernst/Perception-Udacity-RoboticsND-Project3/blob/master/images/linear_conf_matrix_n80_accur93.png?raw=true)
***This is a better looking confusion matrix*** 

###
###
##### Check the Labeling in RViz
###
Now, we can check the labeling of objects in RViz. Exit out of any existing Gazebo or RViz sessions, and type in a terminal:
```sh
roslaunch pr2_robot pick_place_project.launch
```
Give it a moment to boot up, then change to our favorite directory, and run our `project_template.py` code:
```sh
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/
$ rosrun pr2_robot project_template.py
```
It's important to run `project_template.py` from this directory because that is where `model.sav` lives; which is the output of our training of the Support Vector Machine classifier.




(***README IN PROGRESS***)