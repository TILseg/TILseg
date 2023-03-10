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

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    # two sets of tests based on if image is color or grayscale
    if rgb_bool:
        # checks the number of dimensions in the image array to ensure its 4
        if image_array.ndim != 4:
            raise ValueError(f"RGB images should has 4 dimensions but "
                             f"{image_array.ndim} were input")
        else:
            pass

        # checks the fourth dimension to ensure it has 3 color channels RGB
        image_array_shape = image_array.shape
        if image_array_shape[3] != 3:
            raise ValueError("Image should have 3 channels for RGB")
        else:
            pass
    else:
        # checks the number of dimensions in the image array to ensure its 3
        if image_array.ndim != 3:
            raise ValueError(f"Grayscale images should has 3 dimensions but "
                             f"{image_array.ndim} were input")
        else:
            pass
        # checks pixel values to ensure they are between 0 and 255
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
    # call the function that generates all exceptions for this function
    image_series_exceptions(image_array, rgb_bool)

    # find dimensions of array as first dimension is number of images
    dims = image_array.shape
    # save current directory and define filepath of new directory then create
    # if it does not exist
    home = os.getcwd()
    path = os.path.join(filepath, prefix)
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass
    # go to created directory
    os.chdir(path)

    # save dims[0] images in folder from image_array using imsave function
    if rgb_bool:
        for count in range(dims[0]):
            plt.imsave(f"Image{count + 1}.jpg", image_array[count][:][:][:])
    else:
        for count in range(dims[0]):
            cv.imwrite(f"Image{count + 1}.jpg", image_array[count][:][:])
    # go back to the original directory
    os.chdir(home)


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
    # get image dimensions
    dims = ori_image.shape

    # make an empty array as the basis for masks
    binary_array = np.zeros((num_clusts, dims[0], dims[1]))
    # make an empty array as the basis for the all cluster image
    all_mask_array = np.zeros((dims[0], dims[1], 3), np.uint8)

    #  image into 4 dimensions for appending into multiple layers
    four_dim_array = np.expand_dims(ori_image, 0)
    # initalize final_array and append until the first dimension has the
    # same length as the number of clusters
    final_array = four_dim_array
    for _ in range(num_clusts - 1):
        final_array = np.vstack((final_array, four_dim_array))
    # return all generated arrays
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
    # generate errors if the clusters or image are the wrong dimensions
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

    # colors that will become associated with each cluster on overlays
    black = np.array([0, 0, 0], np.uint8)

    # array that holds colors used for generation of all cluster overlay
    colors_overlay = np.array(([0, 0, 0], [255, 0, 0],
                              [0, 255, 0], [0, 0, 255]), np.uint8)

    # making a dictionary of the original images that will be overwriten
    dims = img_clust.shape
    num_clust = int(img_clust.max() + 1)

    # call function that generates arrays for reassignment
    final_arrays, binary_arrays, all_masks = gen_base_arrays(original_image,
                                                             num_clust)

    # itterate over every pixel in the image
    for j in range(dims[0]):
        for k in range(dims[1]):
            # get the cluster and assign it to key
            key = int(img_clust[j][k])
            # recolor the pixel in the overlaid image layer corresponding
            # to the specified cluster
            final_arrays[key][j][k] = black
            # reassign pixel in mask layer based on relevant cluster
            binary_arrays[key][j][k] = 1
            # recolor every pixel with the cluster based on the cluster
            all_masks[j][k] = colors_overlay[key]

    return final_arrays, binary_arrays, all_masks


def mask_only_generator(img_clust: np.ndarray):
    """
    Generates 1 array from cluster. It is a binary mask from each cluster.

    Parameters
    -----
    img_clust: np.ndarray
        a 2D array where the dimensions correspond to X and Y, and the values
        correspond to the cluster assigned to that pixel


    Returns
    -----
    binary_arrays: np.ndarray
        a 3 dimensional array where dimensions correspond to cluster, X, and
        Y. This can be thought of as a list of images with one for each
        cluster. The images will contain 1s in pixels associated with the
        cluster and 0s everywhere else.
    """
    # generate error if the cluster is the wrong dimensions
    if img_clust.ndim != 2:
        raise ValueError(f"Cluster array has 2 dimensions but "
                         f"{img_clust.ndim} were input")
    else:
        pass

    # making a dictionary of the original images that will be overwriten
    dims = img_clust.shape
    num_clust = int(img_clust.max() + 1)

    # make empty array based on dimensions and number of clusters
    binary_arrays = np.zeros((num_clust, dims[0], dims[1]), np.uint8)

    # itterate over every pixel in the image
    for j in range(dims[0]):
        for k in range(dims[1]):
            # get the cluster and assign it to key
            key = int(img_clust[j][k])
            # reassign pixel in mask layer based on relevant cluster
            binary_arrays[key][j][k] = 1
    return binary_arrays


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

    # initalize boolean as false
    meets_crit = False
    # get characteristics of contour
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)

    # filter the contours containing more than one pixel based on area and
    # roundness
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
    # raise an error if the image mask is not binary
    vals = np.unique(img_mask)
    if not(all(vals == [0, 1]) or all(vals == [False, True])):
        raise ValueError("Mask should only have values of 0 or 1")
    else:
        pass

    # generate contours using OpenCV
    contours, _ = cv.findContours(img_mask.astype(np.int32),
                                  cv.RETR_FLOODFILL,
                                  cv.CHAIN_APPROX_NONE)
    # generate an empty list to store filtered contours
    contours_mod = []
    # for each contour check if it meets the filters and add to filtered list
    for ele in enumerate(contours):
        if filter_boolean(contours[ele[0]]):
            contours_mod.append(contours[ele[0]])
    # get the number of contours after filtering
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
    # make an empty array to hold contour parameters
    data_sum = np.zeros((len(cont_list), 4))
    # for each contour calculate area, perimeter and other derived parameters
    for ele in enumerate(cont_list):
        temp_area = cv.contourArea(cont_list[ele[0]])
        temp_per = cv.arcLength(cont_list[ele[0]], True)
        _, temp_radius = cv.minEnclosingCircle(cont_list[ele[0]])
        temp_roundness = temp_per**2 / (4 * np.pi * temp_area)
        temp_circle_area = np.pi * temp_radius**2
        # add contour parameters to the array
        data_sum[ele[0]][:] = [temp_area, temp_per, temp_roundness,
                               temp_circle_area]
    # turn array into dataframe to save as CSV
    dataframe = pd.DataFrame(data_sum, columns=["Area", "Perimeter",
                                                "Roundness",
                                                "Bounding Circle Area"])
    # define path and save CSV
    path = os.path.join(filepath, "Compiled_Data.csv")
    dataframe.to_csv(path, index=False)


def immune_cluster_analyzer(masks: list):
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
    # intialize lists to store contours and number of contours
    contour_list = []
    count_list = []
    # initialize variables which will define the cluster which has the most
    # contours
    count_index = 0
    max_contour_count = 0
    for ele in enumerate(masks):
        # generate contours based on for each mask on a list
        contour_temp, contour_count = contour_generator(masks[ele[0]])
        # append contours and countour count to lists
        contour_list.append(contour_temp)
        count_list.append(contour_count)
        # check if the amount of contours is the largest seen and if so
        # redefine max index and value
        if contour_count >= count_list[count_index]:
            count_index = ele[0]
            max_contour_count = contour_count
        else:
            pass
    # only get contours from the cluster with the most contours and return
    til_contour = contour_list[count_index]
    return til_contour, max_contour_count


def draw_til_images(img: np.ndarray, contours: list, filepath: str):
    """
    This function will generate the an overlaid and mask image from the
    contours.

    Parameters
    -----
    img: nd.ndarray
        3 dimensional array containing X, Y, and color data of the image that
        will be overlaid
    contours: list
        list of arrays of points defining the contours that will be overlaid
        on the images
    filepath:
        directory where the images will be saved
    """
    # get image shape and use to make an empty array
    dims = img.shape
    tils_mask = np.zeros((dims[0], dims[1], 3), np.uint8)
    # draw contours on original image in green and the blank image in white
    cv.drawContours(tils_mask, contours, -1, (255, 255, 255), 3)
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # generate relevant file paths and save overlaid image and mask
    contour_img_filepath = os.path.join(filepath, "ContourOverlay.jpg")
    contour_mask_filepath = os.path.join(filepath, "ContourMask.jpg")
    plt.imsave(contour_img_filepath, img)
    plt.imsave(contour_mask_filepath, tils_mask)


def image_postprocessing(clusters: np.ndarray, ori_img: np.ndarray,
                         filepath: str, gen_all_clusters: bool = False,
                         gen_overlays: bool = False, gen_tils: bool = False,
                         gen_masks: bool = False, gen_csv: bool = False):
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
    # get time to track function run time
    intial_time = time.time()

    # generate errors if cluster and image have incorrect dimensions
    if clusters.ndim != 2:
        raise ValueError(f"Cluster array has 2 dimensions but "
                         f"{clusters.ndim} were input")

    if ori_img.ndim != 3:
        raise ValueError(f"Original image has 3 dimensions but "
                         f"{ori_img.ndim} were input")

    # generate error if the image does not have RGB channels
    ori_shape = ori_img.shape
    clusters_shape = clusters.shape
    if ori_shape[2] != 3:
        raise ValueError("Images should have 3 channels for RGB")

    if ori_shape[0] != clusters_shape[0] or ori_shape[1] != clusters_shape[1]:
        raise ValueError("Image and cluster should have same X any Y"
                         "dimensions")

    # generate masked images, masks and all cluster image via previously
    # defined function
    if any((gen_all_clusters, gen_overlays, gen_masks)):
        masked_images, masks, all_masks = result_image_generator(clusters,
                                                                 ori_img)
    else:
        masks = mask_only_generator(clusters)

    # modify filepath and then make directory with predefined name if saving
    # any images or data
    home = os.getcwd()
    if any((gen_all_clusters, gen_overlays, gen_tils, gen_masks, gen_csv)):
        mod_filepath = os.path.join(filepath, "Clustering Results")
        if not os.path.exists(mod_filepath):
            os.mkdir(mod_filepath)
        os.chdir(mod_filepath)
        
    # if any image generation is required, generate original image
    if any([gen_overlays, gen_masks, gen_tils, gen_all_clusters]):
        ori_filepath = os.path.join(mod_filepath, "Original.jpg")
        plt.imsave(ori_filepath, ori_img)

    # generate contours if images or CSV of TILs is required
    til_list, til_count = immune_cluster_analyzer(masks)

    # save image with all clusters if specified
    if gen_all_clusters:
        all_clust_filepath = os.path.join(mod_filepath, "AllClusters.jpg")
        plt.imsave(all_clust_filepath, all_masks)

    # save overlays if specifed
    if gen_overlays:
        generate_image_series(masked_images, mod_filepath, "Overlaid Images",
                              True)

    # save TILs images if specifed
    if gen_tils:
        draw_til_images(ori_img, til_list, mod_filepath)

    # save masks if specified
    if gen_masks:
        # convert from binary to grayscale and save image series
        masks_imgs = masks * 255
        generate_image_series(masks_imgs, mod_filepath, "Masks", False)

    # generate CSV and save if specified
    if gen_csv:
        csv_results_compiler(til_list, mod_filepath)

    # go back to home directory
    os.chdir(home)

    # print time taken to process
    print(f"Time to process image: {time.time()-intial_time:.3f}")

    return til_count
