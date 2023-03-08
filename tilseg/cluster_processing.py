"""
This python file is meant to generate human and computer readable data for
analysis based on the clusters generated via the clustering model previously
developed. The output is a series of images which represent the
original image and relevant overlays of the determined clusters. Additionally,
based on the clusters, data from filtered cell clusters will be compiled into
a CSV. Immune cell groups are identified using the contour functionality from
OpenCV. The implemented filters are based on area and roundness of the derived
contours.
"""
import os
import time

import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt


# pylint: disable=locally-disabled, no-member, too-many-arguments
# pylint: disable=locally-disabled, no-else-raise


def image_series_exceptions(image_array: np.ndarray, rgb_bool: bool = True):
    """
    This function is used by generate_image_series in order to throw
    exceptions from recieving incorrect array types.

     Parameters
    -----
    image_array: np.ndarray
        a 4 dimensional array where the dimensions are image number, X, Y,
        color from which RGB images are generated
    rgb_bool: bool
        is the image being passed in color or grayscale
    """

    if rgb_bool:
        if image_array.ndim != 4:
            raise ValueError(f"RGB images should has 4 dimensions but "
                             f"{image_array.ndim} were input")
        else:
            pass

        image_array_shape = image_array.shape
        if image_array_shape[3] != 3:
            raise ValueError("Image should have 3 channels for RGB")
        else:
            pass
    else:
        if image_array.ndim != 3:
            raise ValueError(f"Grayscale images should has 3 dimensions but "
                             f"{image_array.ndim} were input")
        else:
            pass

        if image_array.max() > 255 or image_array.min() < 0:
            raise ValueError("Grayscale images should have pixel values"
                             "between 0 and 255")
        else:
            pass


def generate_image_series(image_array: np.ndarray, filepath: str,
                          prefix: str, rgb_bool: bool = True):
    """
    This takes in an array of image values and generates a directory of
    .jpg images in the specified file location

    Parameters
    -----
    image_array: np.ndarray
        a 4 dimensional array where the dimensions are image number, X, Y,
        color from which RGB images are generated
    filepath: str
        the filepath (relative or absolute) in which the directory of images
        is generated
    prefix: str
        the name of the directory created to store the generated images
    rgb_bool: bool
        is the image being passed in color or grayscale
    """

    image_series_exceptions(image_array, rgb_bool)

    dims = image_array.shape
    path = os.path.join(filepath, prefix)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass
    os.chdir(path)

    if rgb_bool:
        for count in range(dims[0]):
            plt.imsave(f"Image{count + 1}.jpg", image_array[count][:][:][:])
    else:
        for count in range(dims[0]):
            cv.imwrite(f"Image{count + 1}.jpg", image_array[count][:][:])


def gen_base_arrays(ori_image: np.ndarray, num_clusts: int):
    """
    Generates three arrays as the basis of cluster assignment. The first array
    contains the original image and will be used to make overlaid images. The
    second is all zeros and will be used to generate boolean masks. The third
    also contains 0 but with different dimensions for use to generate an all
    cluster image.

    Parameters
    -----
    ori_image: np.ndarray
        the original image as a 3 dimensional array with dimensions of X, Y,
        color
    num_clusts: int
        number of clusters which defines length of added dimension in overlaid
        and masks arrays

    Returns
    -----
    final_array: np.ndarray
        4 dimensional array best thought of as a series of 3D arrays where
        each 3D array is the original image and the 4th dimension will
        correspond to cluster after value assignment
    binary_array: np.ndarray
        3 dimensional array where the dimensions correspond to cluster,
        X and Y. This will be used for generation of binary masks for each
        cluster.
    all_mask_array: np.ndarray
        3 dimensional array where the dimensions correspond to X, Y, color.
        This will be used to generate an image with all clusters shown.
    """
    dims = ori_image.shape
    four_dim_array = np.expand_dims(ori_image, 0)
    binary_array = np.zeros((num_clusts, dims[0], dims[1]))
    all_mask_array = np.zeros((dims[0], dims[1], 3), np.uint8)
    final_array = four_dim_array
    for _ in range(num_clusts - 1):
        final_array = np.vstack((final_array, four_dim_array))
    return final_array, binary_array, all_mask_array


def result_image_generator(img_clust: np.ndarray, original_image: np.ndarray):
    """
    Generates 3 arrays from clusters. The first is the each cluster
    individually overlaid on the original image. The second is a binary mask
    from each cluster. The third is an array with each pixel colored based on
    the associated cluster.

    Parameters
    -----
    img_clust: np.ndarray
        a 2D array where the dimensions correspond to X and Y, and the values
        correspond to the cluster assigned to that pixel
    original_image: np.ndarray
        the original image as a 3 dimensional array where dimensions
        correspond to X, Y, and color

    Returns
    -----
    final_arrays: np.ndarray
        a 4 dimensional array where dimensions correspond to cluster, X, Y,
        and color. This can be thought of as a list of images with one for
        each cluster. The images are the original image with cluster pixels
        labeled black
    binary_arrays: np.ndarray
        a 3 dimensional array where dimensions correspond to cluster, X, and
        Y. This can be thought of as a list of images with one for each
        cluster. The images will contain 1s in pixels associated with the
        cluster and 0s everywhere else.
    all_masks: np.ndarray
        a 3 dimensional array where dimensions correspond to X, Y and color.
        The pixels in the array have various colors associated for each
        cluster.
    """

    if img_clust.ndim != 2:
        raise ValueError(f"Cluster array has 2 dimensions but "
                         f"{img_clust.ndim} were input")
    else:
        pass

    if original_image.ndim != 3:
        raise ValueError(f"All cluster image has 3 dimensions but "
                         f"{original_image.ndim} were input")
    else:
        pass

    # Colors that will become associated with each cluster on overlays
    black = np.array([0, 0, 0], np.uint8)

    colors_overlay = np.array(([0, 0, 0], [255, 0, 0],
                              [0, 255, 0], [0, 0, 255]), np.uint8)

    # Making a dictionary of the original images that will be overwriten
    dims = img_clust.shape
    num_clust = int(img_clust.max() + 1)

    final_arrays, binary_arrays, all_masks = gen_base_arrays(original_image,
                                                             num_clust)

    for j in range(dims[0]):
        for k in range(dims[1]):
            key = int(img_clust[j][k])
            final_arrays[key][j][k] = black
            binary_arrays[key][j][k] = 1
            for count in range(3):
                all_masks[j][k][count] = colors_overlay[key][count]

    return final_arrays, binary_arrays, all_masks


def filter_boolean(contour: np.ndarray):
    """
    Determines if a given contour meets the filters that
    have been defined for TILs

    Parameter
    -----
    contour: np.ndarray
        an array of points corresponding to an individual contour

    Returns
    -----
    meets_crit: bool
        boolean that is true if the contour meets the filter and false
        otherwise
    """

    meets_crit = False
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    if area != 0 and perimeter != 0:
        roundness = perimeter**2 / (4 * np.pi * area)
        meets_crit = all([area > 200, area < 2000,
                         roundness < 3.0])
    else:
        pass
    return meets_crit


def contour_generator(img_mask: np.ndarray):
    """
    Creates contours based on an inputted mask and parameters defined here and
    in the filter_boolean function. These parameters define what will be
    classified as likely an immune cell cluster and can be varied within
    filter_bool.

    Parameter
    -----
    img_mask: np.ndarray
        binary 2D array where the dimensions represent X, and Y and values are
        either 0 or 1 based on if the point is contained in the cluster

    Returns
    -----
    contours_mod: list
        list of arrays of points which defines all filtered contours
    contours_count: int
        number of contours that met the determined filters
    """

    if img_mask.max() > 1 or img_mask.min() < 0:
        raise ValueError("Mask should only have values of 0 or 1")
    else:
        pass

    contours, _ = cv.findContours(img_mask.astype(np.int32),
                                  cv.RETR_FLOODFILL,
                                  cv.CHAIN_APPROX_NONE)
    contours_mod = []
    for ele in enumerate(contours):
        if filter_boolean(contours[ele[0]]):
            contours_mod.append(contours[ele[0]])
    contours_count = len(contours_mod)
    return contours_mod, contours_count


def csv_results_compiler(cont_list: list, filepath: str):
    """
    Generates CSV file with relevant areas, perimeters, and circularities
    of filtered contours thought to contain TILs

    Parameters
    -----
    cont_list: list
        list of arrays of points corresponding to contours
    filepath: str
        the filepath where the CSV file will be saved
    """

    data_sum = np.zeros((len(cont_list), 4))
    for ele in enumerate(cont_list):
        temp_area = cv.contourArea(cont_list[ele[0]])
        temp_per = cv.arcLength(cont_list[ele[0]], True)
        _, temp_radius = cv.minEnclosingCircle(cont_list[ele[0]])
        temp_roundness = temp_per**2 / (4 * np.pi * temp_area)
        temp_circle_area = np.pi * temp_radius**2
        data_sum[ele[0]][:] = [temp_area, temp_per, temp_roundness,
                               temp_circle_area]
    dataframe = pd.DataFrame(data_sum, columns=["Area", "Perimeter",
                                                "Roundness",
                                                "Bounding Circle Area"])

    path = os.path.join(filepath, "Compiled_Data.csv")
    dataframe.to_csv(path, index=False)


def immune_cluster_analyzer(masks: list, original_image: np.ndarray,
                            filepath: str):
    """
    This function will generate the contours, identify the relevant cluster
    that contains the immune cells and export the data as a CSV. It will also
    generate an image of the contours overlaid on the image.

    Parameters
    -----
    masks: list
        list of masks which are arrays of 0s and 1s corresponding to cluster
        location

    Returns
    -----
    TIL_contour: list
        list of arrays that correspond to the contours of the filtered TILs
    max_contour_count

    """

    contour_list = []
    count_list = []
    count_index = 0
    max_contour_count = 0
    for ele in enumerate(masks):
        contour_temp, contour_count = contour_generator(masks[ele[0]])
        contour_list.append(contour_temp)
        count_list.append(contour_count)
        if contour_count > count_list[count_index]:
            count_index = ele[0]
            max_contour_count = contour_count
        else:
            pass
    TIL_contour = contour_list[count_index]
    return TIL_contour, max_contour_count


def image_postprocessing(clusters: np.ndarray, ori_img: np.ndarray,
                         filepath: str, gen_all_clusters: bool = True,
                         gen_overlays: bool = True, gen_tils: bool = True,
                         gen_masks: bool = False, gen_csv: bool = True):
    """
    This is a wrapper function that will be used to group all postprocessing
    together. In general postprocessing will generate series of images as well
    as a CSV with general data derived from contour determination using OpenCV.
    Also prints the time taken for post-processing.

    Parameters
    -----
    clusters: np.ndarray
        2D array with dimensions X, and Y and values as the cluster identified
        via the model
    ori_img: np.ndarray
        3D array with dimensions X, Y, and color with three color channels
        as RGB. This is the original image clustering was performed on
    gen_all_clusters: bool
        determines if image with all clusters visualized will be generated
    gen_overlays: bool
        determines if overlaid images will be generated
    gen_tils: bool
        determines if overlaid and mask of TILs will be generated
    gen_masks: bool
        determines if masks will be generated
    gen_csv: bool
        determines if CSV of contours will be generated
    """

    if clusters.ndim != 2:
        raise ValueError(f"Cluster array has 2 dimensions but "
                         f"{clusters.ndim} were input")
    else:
        pass

    if ori_img.ndim != 3:
        raise ValueError(f"Original image has 3 dimensions but "
                         f"{ori_img.ndim} were input")

    ori_shape = ori_img.shape
    if ori_shape[2] != 3:
        raise ValueError("Images should have 3 channels for RGB")
    else:
        pass
   
    intial_time = time.time()
    masked_images, masks, all_masks = result_image_generator(clusters,
                                                             ori_img)

    mod_filepath = os.path.join(filepath, "Clustering Results")

    if not os.path.exists(mod_filepath):
        os.mkdir(mod_filepath)
    else:
        pass
    
    if any([gen_overlays, gen_masks, gen_tils, gen_all_clusters]):
        ori_filepath = os.path.join(mod_filepath, "Original.jpg")
        plt.imsave(ori_filepath, ori_img)
    else: 
        pass
    
    if gen_tils or gen_csv:
        TIL_list, TIL_count = immune_cluster_analyzer(masks, ori_img, mod_filepath)
    else:
        pass

    if gen_all_clusters:
        all_clust_filepath = os.path.join(mod_filepath, "AllClusters.jpg")
        plt.imsave(all_clust_filepath, all_masks)
    else:
        pass

    if gen_overlays:
        generate_image_series(masked_images, mod_filepath, "Overlaid Images",
                              True)
    else:
        pass

    if gen_tils:
        dims = ori_img.shape
        tils_mask = np.zeros((dims[0], dims[1], 3), np.uint8)        
        cv.drawContours(tils_mask, TIL_list, -1, (255, 255, 255), 3)
        contour_mask_filepath = os.path.join(mod_filepath, "ContourMask.jpg")
        plt.imsave(contour_mask_filepath, tils_mask)
        cv.drawContours(ori_img, TIL_list, -1, (0, 255, 0), 3)
        contour_img_filepath = os.path.join(mod_filepath, "ContourOverlay.jpg")
        plt.imsave(contour_img_filepath, ori_img)
    else: 
        pass


    if gen_masks:
        masks_imgs = masks * 255
        generate_image_series(masks_imgs, mod_filepath, "Masks", False)
    else:
        pass

    if gen_csv:
        csv_results_compiler(TIL_list, mod_filepath)
    else:
        pass

    print(f"Time to process image: {time.time()-intial_time:.3f}")

    return TIL_count
