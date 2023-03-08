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
import pandas as pd

# pylint: disable=locally-disabled, no-member, pointless-string-statement
# pylint: disable=locally-disabled, too-many-arguments


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
    count_index = 0
    for ele in enumerate(masks):
        contour_temp, contour_count = contour_generator(masks[ele[0]])
        contour_list.append(contour_temp)
        count_list.append(contour_count)
        if contour_count > count_list[count_index]:
            count_index = ele[0]
        else:
            pass

    data_summary_generator(masks[count_index], filepath)


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

    contours, _ = cv.findContours(img_mask.astype(np.int32),
                                  cv.RETR_FLOODFILL,
                                  cv.CHAIN_APPROX_NONE)
    contours_mod = []
    for ele in enumerate(contours):
        if filter_bool(contours[ele[0]]):
            contours_mod.append(contours[ele[0]])
    return contours_mod, len(contours_mod)


def filter_bool(contour: np.ndarray):
    """
    Determines if a given contour meets the filters that
    have been defined for TILs
    """
    meets_crit = False
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    if area != 0 and perimeter != 0:
        roundness = perimeter**2 / (4 * np.pi * area)
        meets_crit = all(area > 200, area < 2000,
                         roundness < 3.0)
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
    data_sum = np.zeros(len(cont_list), 4)
    for ele in enumerate(cont_list):
        temp_area = cv.contourArea(cont_list[ele[0]])
        temp_per = cv.arcLength(cont_list[ele[0]])
        _, temp_radius = cv.minEnclosingCircle(cont_list[ele[0]])
        temp_roundness = temp_per**2 / (4 * np.pi * temp_area)
        temp_circle_area = np.pi * temp_radius**2
        data_sum[ele[0]][:] = [temp_area, temp_per, temp_roundness,
                               temp_circle_area]
    dataframe = pd.DataFrame(data_sum, columns=["Area", "Perimeter",
                                                "Roundness",
                                                "Bounding Circle Area"])

    path = os.path.join(filepath, "Compiled_Data")
    dataframe.to_csv(path, index=False)


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


def gen_base_arrays(ori_image: np.ndarray, num_clusts: int):
    """
    Generates set of two arrays which will be overlaid with the cluster data.
    The first array contains the number of clusters+1 images which will recieve
    masks based on the associated cluster. The second is a binary array for use
    in contour generation.
    """
    dims = ori_image.shape
    four_dim_array = np.expand_dims(ori_image, 0)
    binary_array = np.zeros((num_clusts, dims[0], dims[1]))
    all_mask_array = np.zeros((3, dims[0], dims[1]))
    final_array = four_dim_array
    for _ in range(num_clusts):
        final_array = np.vstack((final_array, four_dim_array))
    return final_array, binary_array, all_mask_array


def generate_image_series(image_array: np.ndarray, filepath: str, prefix: str):
    """
    This takes in an array of image values and generates a directory of
    jpg images in a specified file location.
    """
    dims = image_array.shape
    path = os.path.join(filepath, prefix)
    if not os.path.exists(filepath):
        os.mkdir(path)
    else:
        pass
    os.chdir(path)
    for count in range(dims[2]+1):
        cv.imwrite(f"Image{count + 1}.jpg", image_array[count][:][:][:])


def image_overlay_generator(img_clust: np.ndarray, original_image: np.ndarray):
    """
    Generates series of images equal to the number of clusters plus the
    original and saves it to the specified filepath
    """
    # Colors that will become associated with each cluster on overlays
    black = np.array([0, 0, 0])

    colors_overlay = np.array([0, 0, 0], [255, 0, 0],
                              [0, 255, 0], [0, 0, 255])

    # Making a dictionary of the original images that will be overwriten
    dims = img_clust.shape
    num_clust = img_clust.max()

    final_arrays, binary_arrays, all_masks = gen_base_arrays(original_image,
                                                             num_clust)

    for j in range(dims[0]):
        for k in range(dims[1]):
            key = int(img_clust[j][k][3])
            final_arrays[key][j][k] = black
            binary_arrays[key][j][k] = 1
            all_masks[:][j][k] = colors_overlay[key]

    return final_arrays, binary_arrays, all_masks


def image_postprocessing(clusters: np.ndarray, ori_img: np.ndarray,
                         filepath: str, gen_overlays: bool = True,
                         gen_masks: bool = False, gen_csv: bool = True):
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

    masked_images, masks, all_masks = image_overlay_generator(clusters,
                                                              ori_img)

    cv.imwrite("Original.jpg", ori_img)
    cv.imwrite("AllClusters.jpg", all_masks)

    if gen_overlays:
        generate_image_series(masked_images, filepath, "Overlaid Images")
    else:
        pass

    if gen_masks:
        generate_image_series(masks, filepath, "Masks")
    else:
        pass

    if gen_csv:
        immune_cluster_generator(masks, filepath)
    else:
        pass
    return masked_images, masks


original_image1 = np.load("/home/bradyr18/patch.npy")
test_array = np.load("/home/bradyr18/both.npy")

final, binary, full_masks = image_overlay_generator(test_array,
                                                    original_image1)

mid = time.time()

for i in range(4):
    area1 = contour_generator(binary[i])
