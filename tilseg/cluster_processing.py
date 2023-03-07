"""
This python file is meant to generate human and computer readable data for
analysis based on the clusters generated via the clustering model previously
developed. The intended output is a series of images which represent the
original image and relevant overlays of the determined clusters. Additionally,
based on the clusters, data from filtered cell clusters will be compiled into
a CSV.
"""
import os
import time

import numpy as np
import cv2 as cv

start = time.time()

def immune_cluster_generator(masks: list, filepath: str):
    """
    This function will generate the contours, identify the relevant cluster
    that contains the immune cells and export the data as a CSV

    Inputs: 
    masks - a list of 2D arrays which are binary representations of the 
    clusters
    filepath - string of where the CSV will be saved
    """
    contour_list = []
    count_list = []
    for ele in enumerate(masks):
        contour_temp, contour_count = contour_generator(masks[ele[0]])
        contour_list.append(contour_temp)
    


def contour_generator(img_mask: np.ndarray):
    """
    Creates contours based on an inputted mask and parameters defined herein.
    These parameters define what will be classified as likely an immune
    cell cluster and can be varied within this code block.

    Input:
    -img_mask: binary 2D array where the dimensions represent the x and y
        coordinates of the relevant pixels

    Output:
    -Contour: list of arrays of points defining the contour
    """

    contours, hierarchy = cv.findContours(img_mask.astype(np.int32),
                                          cv.RETR_FLOODFILL,
                                          cv.CHAIN_APPROX_NONE)
    contours_mod = []
    for ele in enumerate(contours):
        if filter_bool(contours[ele[0]]):
            contours_mod.append(contours[ele[0]])
    return contours_mod, len(contours_mod)


def filter_bool(contour: np.ndarray):
    meets_crit = False
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    if area != 0 and perimeter != 0:
        Roundness = perimeter**2 / (4 * np.pi * area)
        meets_crit = bool(area > 200
                          and area < 2000
                          and Roundness < 3.0)
    else:
        pass
    return meets_crit


def data_summary_generator(cont_list: list, filepath: str):
    """
    Generates CSV file with relevant areas, intensities, and circularities
    of previously identified cell groups

    Input:
    -list of arrays of points corresponding to generated contours
    """
    data_sum = np.ndarray()
    # for ele in enumerate(cont_list):
        

'''
#This is code I used to append two array but not required everytime and
#relatively slow
cluster=np.load("/home/bradyr18/cluster2.npy")
cluster=np.reshape(cluster, (3000, 4000))
patch=np.load("/home/bradyr18/patch.npy")
print(cluster.shape)
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
    four_dim_array = np.expand_dims(ori_image, 0)
    binary_array = np.zeros((num_clust, array_dims[0], array_dims[1]))
    final_array = four_dim_array
    for _ in range(num_clust):
        final_array = np.vstack((final_array, four_dim_array))
    return final_array, binary_array


def generate_images(image_array: np.ndarray, filepath: str, num_clust: int):
    """
    This takes in an array of image values and generates a directory of
    jpg images in a specified file location.
    """
    path = os.path.join(filepath, "Overlaid Images")
    if not os.path.exists(filepath):
        os.mkdir(path)
    else:
        pass
    os.chdir(path)
    for m in range(num_clust+1):
        if m != num_clust:
            cv.imwrite(f"Image{m + 1}.jpg", image_array[m][:][:][:])
        else:
            cv.imwrite("Original.jpg", image_array[m][:][:][:])
    return None


def image_overlay_generator(img_clust: np.ndarray, original_image: np.ndarray,
                            clust_count: int, filepath: str):
    """
    Generates series of images equal to the number of clusters plus the
    original and saves it to the specified filepath
    """
    # Colors that will become associated with each cluster on overlays
    overlay_color = np.array([0, 0, 0])

    # Making a dictionary of the original images that will be overwriten
    dims = img_clust.shape

    final_arrays, binary_arrays = gen_base_arrays(original_image, clust_count,
                                                  dims)

    for j in range(dims[0]):
        for k in range(dims[1]):
            key = int(img_clust[j][k][3])
            final_arrays[key][j][k] = overlay_color
            binary_arrays[key][j][k] = 1

    # Commented out temporarilly to look at code
    generate_images(final_arrays, filepath, clust_count)

    return final_arrays, binary_arrays


def image_postprocessing(clusters: np.ndarray, ori_img: np.ndarray, filepath:str,
                         gen_overlays: bool = True, gen_csv: bool = True):
    """
    This is a wrapper function that will be used to group all postprocessing
    together.

    Inputs: 
    ori_img - 3D array with dimensions X, Y, and color with three color
    channels as RGB
    clusters - 2D array with dimensions X, Y and values as the cluster
    identified via the model
    gen_overlays - boolean to determine if overlay images will be generated
    gen_csv - boolean to determine if CSV of contours will be generated
    """
    
    dim = clusters.max + 1
    if gen_overlays == True:
        masked_images, masks = image_overlay_generator(clusters, ori_img,
                                                       dim, filepath)
    else: 
        pass

    if gen_csv == True:
            image_overlay_generator(clusters, ori_img, dim, filepath)

    

original_image1 = np.load("/home/bradyr18/patch.npy")
test_array = np.load("/home/bradyr18/both.npy")

final, binary = image_overlay_generator(test_array, original_image1,
                                        4, "/home/bradyr18")

mid = time.time()

for i in range(4):
    area = contour_generator(binary[i])
