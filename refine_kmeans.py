# def (function that creates a superpatch from 3 class classified images)
#   investigate the nature of these images
# def (function that optimizes DBSCAN hyperparameters on 3 class clasified images)
# def (function that fits the model on the superpatch)

# Core library imports
import os
import pathlib

# External library imports
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError, Image
import cv2

# Local imports
from tilseg.cluster_processing import mask_only_generator, image_postprocessing
from tilseg.seg import segment_TILs
from tilseg.model_selection import opt_kmeans

# KMeans_superpatch_fit function is cpoied here for ease
def KMeans_superpatch_fit(patch_path: str,
                          hyperparameter_dict: dict = {'n_clusters: 4'}):

    """
    Fits a KMeans clustering model to a patch that will be used to cluster
    other patches
    KMeans is the only clustering algorithms that allows fitting a model to
    one patch clustering on another
    All other clustering algorithms need to be fitted on the same patch that
    needs to be clustered
    It makes sense to use this function to fit a KMeans clustering model to a
    superpatch that can capture H&E stain variation across patients and
    technologies

    Parameters
    -----
    patch_path: str
        the directory path to the patch that the model will be fitted to
        obtain cluster decision boundaries
    hyperparameter_dict: dict
        dicitonary of hyperparameters for KMeans containing 'n_clusters' as
        the only key
        this dictionary can be obtained by reading the JSON file outputted by
        tilseg.module_selection

    Returns
    -----
    model: sklearn.base.ClusterMixin
        the fitted model
    """

    # Checks that the path to the patch is a string
    if not isinstance(patch_path, str):
        raise TypeError('patch_path must be a string')

    # Checks that the patch_path actually exists
    path = pathlib.Path(patch_path)
    if not path.is_file():
        raise FileNotFoundError('Please input a path to a file that exists')

    # Checking that hyperparameter_dict is a dictionary
    if not isinstance(hyperparameter_dict, dict):
        raise TypeError('hyperparameter_dict must be a dictionary')

    # Creates a variable which references the preferred parameters for KMeans
    # clustering
    key_list = list(hyperparameter_dict.keys())
    expected_key_list = ['n_clusters','metric']
    # Checks that the expected keys are present in hyperparameters_dict
    if set(key_list) != set(expected_key_list):
        raise KeyError(
            'Please enter the appropriate keys in hyperparameter_dict')

    # Checks that n_clusters is an integer and less than 9
    if (not isinstance(hyperparameter_dict['n_clusters'], int) or
            hyperparameter_dict['n_clusters'] > 8):
        raise ValueError('Please enter an integer less than 9 for n_clusters')

    # Fits the KMeans clustering model using the optimized value for n_clusters
    model = sklearn.cluster.KMeans(**hyperparameter_dict, max_iter=20,
                                   n_init=3, tol=1e-3)

    try:
        # Reads the patch into a numpy uint8 array
        fit_patch = plt.imread(patch_path)
    # Makes sure that the fie is readable by matplotlib, which uses PIL
    except UnidentifiedImageError:
        print('Please use an image that can be opened by PIL.Image.open')
        raise

    # Linearizes the array for R, G, and B separately and normalizes
    # The result is an N X 3 array where N=height*width of the patch in pixels
    fit_patch_n = np.float32(fit_patch.reshape((-1, 3))/255.)

    # Fits the model to the linearized and normalized patch data
    model.fit(fit_patch_n)

    # Outputs KMeans model fitted to the superpatch and will be used as input
    # to clustering_score and segment_TILs functions
    return model

### Unsure where to put this function but pasted here for now for workflow
# 
def kmean_dbscan_patch_wrapper(patch_path: str,
                n_clusters: list,
                 out_dir_path: str = None,
                 save_TILs_overlay: bool = False,
                 save_cluster_masks: bool = False,
                 save_cluster_overlays: bool = False,
                 save_all_clusters_img: bool = False,
                 save_csv: bool = False):
    
    # Find Kmeans Parameters (num clusters)
    img = Image.open(patch_path)
    numpy_img = np.array(img)
    numpy_img_reshape = np.float32(numpy_img.reshape((-1, 3)) / 255.)
    opt_cluster = opt_kmeans(numpy_img_reshape, n_clusters)
    hyperparameter_dict = {'n_clusters': opt_cluster}
    kmeans_fit = KMeans_superpatch_fit(patch_path, hyperparameter_dict)
    print("Completed Kmeans fitting.")
    
    # Run Segmentation on Kmeans Model
    TIL_count_dict, kmean_labels_dict = segment_TILs(patch_path,
                 out_dir_path,
                 None,
                 'KMeans',
                 kmeans_fit,
                 save_TILs_overlay, 
                 save_cluster_masks,
                 save_cluster_overlays,
                 save_all_clusters_img,
                 save_csv,
                 False)
    
    #Feed into DBSCAN
    return TIL_count_dict, kmeans_fit
    ## changed the output to include the KMeans cluster labels

# Takes a KMeans binary TILs mask and converts it into an array to feed into DBSCAN
def km_labels_to_features(patch_path: str, n_clusters: list):
    
    # Extracting the original patch as a 3D array with dimensions X, Y, 
    # and color with three color channels as RGB.
    original_image = cv2.imread(patch_path)
    
    # obtain the KMeans labels on the patch (2D array with dimensions X, and Y 
    # and values as the cluster identified via the model)
    _, kmean_labels = kmean_dbscan_patch_wrapper(patch_path, n_clusters)

    # implement image_postprocessing function to get binary mask
    _, binary_mask = image_postprocessing(kmean_labels, original_image)

    # Use np.argwhere to find the coordinates of non-zero (1) elements in the binary mask
    tils_coords = np.argwhere(binary_mask == 1)

    # Prepare coordinates as your feature matrix
    features = tils_coords

    return features, kmean_labels
    # features (np.array) is a an array where each row corresponds to a set of 
    # coordinates (x,y) of the pixels where the binary_mask had a value of 1

# Takes the spatial coordinates from a binary mask as features to cluster with DBSCAN
# note: previous group found n=4 to be optimal, so the number of clusters used to fit 
#       KMeans is set to a default value of 4
def km_dbscan_wrapper(patch_path: str, eps: float, min_samples: int, km_clusters: int = 4):

    # Extract K-means labels and the features that be fed into DBSCAN
    # note: kmean_dbscan_patch_wrapper is from the seg.py script right now
    features, km_labels = km_labels_to_features(patch_path, km_clusters)

    # Create DBSCAN model and fit it onto the features
    dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
    dbscan_labels = dbscan.fit_predict(features)

    # Visualize the DBSCAN labels

    return dbscan_labels


## MISC FUNCTIONS

# def kmeans_apply_patch(model, patch_path: str):

#     # Load the image using OpenCV
#     patch = cv2.imread(patch_path)

#     # Check if the image loading was successful
#     if patch is None:
#         raise ValueError("Path to patch not provided. Please input desired path name.")

#     # Flatten the patch to a 1D array of pixels and normalizes it
#     patch_pixels_norm = np.float32(patch.reshape((-1, 3))/255.)

#     # Predict cluster labels using the pre-fitted KMeans model
#     km_cluster_labels = model.predict(patch_pixels_norm)

#     # Reshape the cluster labels back to the original image shape
#     km_cluster_labels = km_cluster_labels.reshape(patch.shape[:2])

#     return km_cluster_labels

# def kmeans_til_label(model, patch_path: str):
    
#     # Cluster the image based on the pre-fitted model from the superpatch
#     km_labels = kmeans_apply_patch(model, patch_path)

#     # Initialize a binary mask for TILs
#     til_mask = np.zeros_like(km_labels, dtype=np.uint8)

#     # Iterate over each unique K-Means cluster label
#     unique_labels = np.unique(km_labels)
#     for label_value in unique_labels:
#         if label_value == -1:  # Skip the background label if present
#             continue

#         # Create a binary mask for the current cluster
#         cluster_mask = np.uint8(km_labels == label_value)

#         # Use skimage's regionprops to get properties of connected regions
#         labeled_regions = label(cluster_mask)
#         regions = regionprops(labeled_regions)

#         # Filter regions based on circularity and area of a single TIL
#         for region in regions:
#             area = region.area
#             perimeter = region.perimeter
#             # Ensure that the denominator is not zero before calculating circularity
#             if area != 0 and perimeter != 0:
#                 circularity = 4 * np.pi * area / (perimeter ** 2)
#                 if 200 < region.area < 2000 and circularity > 0.3:
#                     # Add the pixels of the qualified region to the TIL mask
#                     til_mask += (labeled_regions == region.label)

#     # Ensure the shapes of the til_mask and km_labels match
#     if til_mask.shape != km_labels.shape:
#         raise ValueError("Shape mismatch between til_mask and km_cluster_labels.")

#     # Find unique cluster labels within the TILs mask
#     unique_labels, counts = np.unique(km_labels[til_mask > 0], return_counts=True)

#     # Find the label with the highest count which is most likely 
#     # the cluster that contains TILs
#     til_cluster_label = unique_labels[np.argmax(counts)]

#     # Create a binary mask for the TILs cluster from the initial kmeans clustering
#     binary_tils_mask = np.uint8(km_labels == til_cluster_label)

#     return binary_tils_mask, km_labels, til_cluster_label

# def calculate_black(image_path, threshold=50):
#     """
#     A function that takes the complete path to the image and calculates the 
#     percentage of the image that is black (i.e. background)

#     Parameters
#     -----
#     slidepath: the complete path to the slide file (.svs)

#     Returns
#     -----
#     black_percentage (float): percentage of the image that is black background
#     """
#     # Load the image
#     img = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Threshold the image to identify black regions
#     _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

#     # Calculate the total number of pixels in the image
#     total_pixels = img.shape[0] * img.shape[1]

#     # Count the number of black pixels
#     black_pixels = np.sum(thresholded == 0)

#     # Calculate the percentage of the black region
#     black_percentage = (black_pixels / total_pixels) * 100

#     return black_percentage

# def calculate_black_folder(folder_path, threshold=50):
    """
    A function that takes the complete path to the image and calculates the 
    percentage of the image that is black (i.e. background)

    Parameters
    -----
    slidepath: the complete path to the slide file (.svs)

    Returns
    -----
    black_percentages (list): S list of tuples where the first element is a string 
    (file name) and the second element is a float (percentage).
    """
    
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    black_percentages = []

    for file in files:
        image_path = os.path.join(folder_path, file)
        if file.lower().endswith(('.tif')):
            black_percentage = calculate_black(image_path, threshold)
            black_percentages.append((file, black_percentage))

    return black_percentages
    