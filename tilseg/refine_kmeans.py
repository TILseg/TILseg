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
import sklearn.cluster
from sklearn.preprocessing import StandardScaler
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError, Image
import cv2


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
    expected_key_list = ['n_clusters']
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

# def mean_shift_patch_fit(data):
#     data = np.array(data)
#     hyperparameter_dict = opt_mean_shift(data = data,
#                    bandwidth = [0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9],
#                    seeds=[0.1,0.2,0.4,0.5])
#     model = sklearn.cluster.MeanShift(**hyperparameter_dict, max_iter=20,
#                                    n_init=3, tol=1e-3)
#     model.fit_predict(data)
#     cluster_labels = model.labels_
#     cluster_centers = model.cluster_centers_
#     return model, cluster_labels, cluster_centers

def mask_to_features(binary_mask:np.ndarray):
    """
    Generates the spatial coordinates from a binary mask as features to cluster with DBSCAN
    
    Parameters
    -----
    binary_mask (np.ndarray): a binary mask with 1's corresponding to the pixels 
    involved in the cluser with the most contours and 0's for pixels not

    Returns
    -----
    features (np.array) is a an array where each row corresponds to a set of 
    coordinates (x,y) of the pixels where the binary_mask had a value of 1
    """
    
    # Use np.argwhere to find the coordinates of non-zero (1) elements in the binary mask
    tils_coords = np.argwhere(binary_mask == 1)

    # Prepare coordinates as your feature matrix
    features = tils_coords

    return features


def km_dbscan_wrapper(mask: np.ndarray, hyperparameter_dict):
    #hyper dict keys
    #eps: float, min_samples: int,
   
    #Generate Spatial Coordiantes
    features = mask_to_features(mask)

    # DBSCAN Model Fitting
    dbscan = sklearn.cluster.DBSCAN(**hyperparameter_dict)
    dbscan_labels = dbscan.fit_predict(features)

    #Generate Labels for Plot
    mask_reshape = mask.reshape(-1,1)
    all_labels = np.full(len(mask_reshape), -1)
    indices = [i for i, val in enumerate(mask_reshape) if val == 1] #indices of labels being inserted
    for index, new_label in zip(indices, dbscan_labels): #Loops through dbscan labels and adds to all_labels array in corresponding index position
        all_labels[index] = new_label
    all_labels = all_labels.reshape(mask.shape)
    
    #Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(all_labels, cmap='viridis')  # Change the colormap as needed
    plt.colorbar()
    plt.title('DBSCAN Clustering Result')
    plt.show()

    # # Save the plotted image
    # plt.savefig('dbscan_result.png')
        
    return all_labels


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

# # def calculate_black_folder(folder_path, threshold=50):
#     """
#     A function that takes the complete path to the image and calculates the 
#     percentage of the image that is black (i.e. background)

#     Parameters
#     -----
#     slidepath: the complete path to the slide file (.svs)

#     Returns
#     -----
#     black_percentages (list): S list of tuples where the first element is a string 
#     (file name) and the second element is a float (percentage).
#     """
    
#     # Get a list of all files in the folder
#     files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#     black_percentages = []

#     for file in files:
#         image_path = os.path.join(folder_path, file)
#         if file.lower().endswith(('.tif')):
#             black_percentage = calculate_black(image_path, threshold)
#             black_percentages.append((file, black_percentage))

#     return black_percentages
    