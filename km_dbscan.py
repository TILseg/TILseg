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
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError
import cv2


# SUPERPATCH CREATION FROM PATCHES

def calculate_black(image_path, threshold=50):
    """
    A function that takes the complete path to the image and calculates the 
    percentage of the image that is black (i.e. background)

    Parameters
    -----
    slidepath: the complete path to the slide file (.svs)

    Returns
    -----
    black_percentage (float): percentage of the image that is black background
    """
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to identify black regions
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Calculate the total number of pixels in the image
    total_pixels = img.shape[0] * img.shape[1]

    # Count the number of black pixels
    black_pixels = np.sum(thresholded == 0)

    # Calculate the percentage of the black region
    black_percentage = (black_pixels / total_pixels) * 100

    return black_percentage

def calculate_black_folder(folder_path, threshold=50):
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

# KMEANS

def kmeans_superpatch_fit(patch_path: str):

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

    # Creat KMeans instance (n=4)
    kmeans = KMeans(n_clusters=4, max_iter=20,
                                   n_init=3, tol=1e-3)

    try:
        # Reads the patch into a numpy uint8 array
        super_patch = cv2.imread(patch_path)
    # Makes sure that the fie is readable by matplotlib, which uses PIL
    except UnidentifiedImageError:
        print('Please use an image that can be opened by PIL.Image.open')
        raise

    # Linearizes the array for R, G, and B separately and normalizes
    # The result is an N X 3 array where N=height*width of the patch in pixels
    sp_pixels_norm = np.float32(super_patch.reshape((-1, 3))/255.)

    # Fits the model to the linearized and normalized patch data
    kmeans.fit(sp_pixels_norm)

    # Outputs KMeans model fitted to the superpatch and will be used as input
    # to clustering_score and segment_TILs functions
    return kmeans

def kmeans_apply_patch(model, patch_path: str):

    # Load the image using OpenCV
    patch = cv2.imread(patch_path)

    # Check if the image loading was successful
    if patch is None:
        raise ValueError("Path to patch not provided. Please input desired path name.")

    # Flatten the patch to a 1D array of pixels and normalizes it
    patch_pixels_norm = np.float32(patch.reshape((-1, 3))/255.)

    # Predict cluster labels using the pre-fitted KMeans model
    km_labels = model.predict(patch_pixels_norm)

    # Reshape the cluster labels back to the original image shape
    km_labels = km_labels.reshape(patch.shape[:2])

    return km_labels

# DBSCAN

def km_dbscan(km_model, patch_path: str, eps: float, min_samples: int):

    # Extract K-means labels
    km_labels = kmeans_apply_patch(km_model, patch_path)

    # Use K-Means labels as features for DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
    dbscan_labels = dbscan.fit_predict(km_labels.reshape(-1, 1))

    # Combine K-Means and DBSCAN labels
    final_labels = np.where(dbscan_labels == -1, -1, km_labels)

    # Visualization 

    return km_labels, dbscan_labels, final_labels

# Visualize or use the final_labels as needed

# Apply DBSCAN on K-Means results
#kmeans_clusters = pd.DataFrame({'KMeans_Labels': kmeans_labels, 'Feature1': your_data['Feature1'], 'Feature2': your_data['Feature2']})
#scaler = StandardScaler()
#kmeans_clusters_scaled = scaler.fit_transform(kmeans_clusters[['Feature1', 'Feature2']])

#dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#dbscan_labels = dbscan.fit_predict(kmeans_clusters_scaled)

# Combine K-Means and DBSCAN labels
#final_labels = np.where(dbscan_labels == -1, -1, kmeans_labels)