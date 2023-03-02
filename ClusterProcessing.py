"""
This python file is meant to generate human and computer readable data for
analysis based on the clusters generated via the clustering model previously
developed. The intended output is a series of images which represent the
original image and relevant overlays of the determined clusters. Additionally,
based on the clusters, data from filtered cell clusters will be compiled into
a CSV. 
"""

import numpy as np
import  cv2 as cv

def ClusterSplit(img_clust: np.ndarray, clust_count: int):
    """
    This function is designed to take an input clustered image and turn it into
    a series of masks which can be further processed by OpenCV to identify cell
    groups

    Inputs:
    -img_clust: a 3D array where two dimensions represent the x and y coordinates
        of the image and the third dimension represents RGB and asigned cluster
    -clust_count: an integer 

    Output:
    -Dictionary of binary image arrays for each cluster
    """

def ContourGenerator(img_mask: np.ndarray):
    """
    Creates contours based on an inputted mask and parameters defined herein.
    These parameters define what will be classified as likely an immune 
    cell cluster and can be varied within this code block.
    
    Input:
    -img_mask: binary 2D array where the dimensions represent the x and y
        coordinates of the relevant pixels
    
    Output: 
    -Contour object?
    """

def DataSummaryGenerator(cont_dict: dict, filepath: str):
    """
    Generates CSV file with relevant areas, intensities, and circularities
    of previously identified cell groups
    """

def ImageOverlayGenerator(img_clust: np.ndarray, clust_count: int, filepath: str):
    """
    Generates series of images equal to the number of clusters plus the
    original and saves it to the specified filepath
    """

    overlay_color=np.array(
        [205, 102, 102],
        [153, 255, 51],
        [0, 128, 255],
        [0, 255, 255],
        [178, 102, 255],
        [95, 95, 95],
        [102, 0, 0],
        [255, 0, 127])

    dims = img_clust.shape()
    final_arrays = {}
    for i in range(clust_count+1):
        temp_array=np.zeros(dims[0], dims[1],3)
            for j in range(dims[0]):
                for k in range(dims[1]):
                    if img_clust[j][k][4] == i:
                        temp_array[j][k][:] = overlay_color[i][:]
                    else: 
                        temp_array[j][k][:] = KeepOriginal(img_clust[j][k], 3)
        if i <= clust_count:
            final_arrays[f"Cluster {i}"] = temp_array
        else:
            final_arrays["Original"] = temp_array

def KeepOriginal(array: np.ndarray, num_chan):
    temp_list=np.zeros(num_chan)
    for m in range(num_chan):
        temp_list[m]=array[m]
    return temp_list


        
