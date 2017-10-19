import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

os.chdir('/Users/ChrisErnst/Development/Perception-Challenge-Udacity-RoboticsND-Project3')

# Read in an image
image = mpimg.imread('Udacican.jpg')


# Define a function to compute color histogram features  
def rgb_hist(img, nbins=32, bins_range=(0, 256)):
    # Take histograms in R, G, and B
    r_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    g_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    b_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generate bin centers(they are all same size, so we'll use Red
    bin_edges = r_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Plot the RGB hist
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bin_centers, r_hist[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bin_centers, g_hist[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bin_centers, b_hist[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    plt.show()
# The rgb_hist function creates a R G B histogram of the image
    
rgb_hist(image)


# The function below takes an RGB image and outputs an HSV; 
# where the x-axis corresponds to histogram bins and the y-axis corresponds 
# to what fraction of the signal comes from each bin. 
# This is now the HSV color signature of the image, where the first 32 bins along
# the x-axis (left side) correspond to the H-channel, the next 32 (center) to 
# the S-channel and the last 32 (right side) to the V-channel.

def color_hist(img, nbins=32, bins_range=(0, 256)):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Take histograms in R, G, and B
    r_hist = np.histogram(hsv_image[:,:,0], bins=nbins, range=bins_range)
    g_hist = np.histogram(hsv_image[:,:,1], bins=nbins, range=bins_range)
    b_hist = np.histogram(hsv_image[:,:,2], bins=nbins, range=bins_range)
    # Extract the features
    hist_features = np.concatenate((r_hist[0], g_hist[0], b_hist[0])).astype(np.float64)
    # Normalize
    norm_features = hist_features / np.sum(hist_features)
    return norm_features

# Call the function:
feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

fig = plt.figure(figsize=(12,6))
plt.plot(feature_vec)
plt.title('HSV Feature Vector', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
fig.tight_layout()
