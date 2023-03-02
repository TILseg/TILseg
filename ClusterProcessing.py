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
import os 
#from PIL import Image

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

'''
#This is code I used to append two array but not required everytime and 
#relatively slow
cluster=np.load("/home/bradyr18/cluster.npy")
patch=np.load("/home/bradyr18/patch.npy")
test_array=np.zeros([3000,4000,4])
for i in range(3000):
    for j in range(4000):
        for k in range(2):
            test_array[i][j][k]=patch[i][j][k]
        test_array[i][j][3]=cluster[i][j]
np.save("/home/bradyr18/both.npy", test_array)
'''
original_image = np.load("/home/bradyr18/patch.npy")
test_array = np.load("/home/bradyr18/both.npy")

def ImageOverlayGenerator(img_clust: np.ndarray, original_image: np.ndarray, clust_count: int, filepath: str):
    """
    Generates series of images equal to the number of clusters plus the
    original and saves it to the specified filepath
    """
    #Colors that will become associated with each cluster on overlays
    OverlayColor=np.array(
        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])

    #Making a dictionary of the original images that will be overwriten
    dims = img_clust.shape
    FourDImage=np.expand_dims(original_image, 0)
    FinalArrays=FourDImage
    for i in range(clust_count):
        FinalArrays=np.vstack((FinalArrays, FourDImage))

    for j in range(dims[0]):
        for k in range(dims[1]):
            key=int(img_clust[j][k][3])
            FinalArrays[key+1][j][k]=OverlayColor[key]

    path=os.path.join(filepath, "ImageStack1")
    os.mkdir(path)
    os.chdir(path)
    for i in range(clust_count+1):
        cv.imwrite(f"Image{i}.jpg", FinalArrays[i][:][:][:])
    return FinalArrays


final_arrays=ImageOverlayGenerator(test_array, original_image, 8, "/home/bradyr18")

