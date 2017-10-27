import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


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

    # Compute histograms of color values (Color channels Range from 0 to 256(8 bit))
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

    # Compute histograms of surface normal values (Surface Normals Range from -1 to 1)
    nbins= 32
    bins_range=(-1,1)

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
