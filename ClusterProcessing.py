"""
This python file is meant to generate human and computer readable data for
analysis based on the clusters generated via the clustering model previously
developed. The intended output is a series of images which represent the
original image and relevant overlays of the determined clusters. Additionally,
based on the clusters, data from filtered cell clusters will be compiled into
a CSV.
"""
import os

import numpy as np
import  cv2 as cv
#from PIL import Image

def contour_generator(img_mask: np.ndarray):
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

def data_summary_generator(cont_dict: dict, filepath: str):
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

def gen_base_arrays(ori_image: np.ndarray, num_clust: int, array_dims: list):
    """
    Generates set of two arrays which will be overlaid with the cluster data.
    The first array contains the number of clusters+1 images which will recieve
    masks based on the associated cluster. The second is a binary array for use
    in contour generation.
    """
    four_dim_array=np.expand_dims(ori_image, 0)
    binary_array=np.full((num_clust, array_dims[0], array_dims[1]),False)
    final_array=four_dim_array
    for _ in range(num_clust):
        final_array=np.vstack((final_array, four_dim_array))
    return final_array, binary_array

def generate_images(image_array: np.ndarray, filepath: str, num_clust: int):
    """
    This takes in an array of image values and generates a directory of
    jpg images in a specified file location.
    """
    path=os.path.join(filepath, "Overlaid Images")
    os.mkdir(path)
    os.chdir(path)
    for i in range(num_clust+1):
        if i !=0:
            cv.imwrite(f"Image{i}.jpg", image_array[i][:][:][:])
        else:
            cv.imwrite("Original.jpg", image_array[i][:][:][:])
    return None

def image_overlay_generator(img_clust: np.ndarray, original_image: np.ndarray,
                             clust_count: int, filepath: str):
    """
    Generates series of images equal to the number of clusters plus the
    original and saves it to the specified filepath
    """
    #Colors that will become associated with each cluster on overlays
    overlay_color=np.array(
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

    final_arrays, binary_arrays = gen_base_arrays(original_image, clust_count, dims)

    for j in range(dims[0]):
        for k in range(dims[1]):
            key=int(img_clust[j][k][3])
            final_arrays[key][j][k]=overlay_color[key]
            binary_arrays[key][j][k]=True

    generate_images(final_arrays, filepath, clust_count)

    return final_arrays, binary_arrays



original_image1 = np.load("/home/bradyr18/patch.npy")
test_array = np.load("/home/bradyr18/both.npy")

final, binary=image_overlay_generator(test_array, original_image1, 8, "/home/bradyr18")
