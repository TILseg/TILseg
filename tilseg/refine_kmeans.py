# Core library imports
import os
import pathlib

# External library imports
import numpy as np
import pandas as pd
import sklearn.cluster
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError, Image
import time

#Local Imports
from tilseg.seg import segment_TILs
from tilseg.model_selection import opt_kmeans


# KMeans_superpatch_fit function is cpoied here for ease
def KMeans_superpatch_fit(patch_path: str,
                          hyperparameter_dict: dict = {'n_clusters: 4'},
                          random_state = None):

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
    random_state: int
        the random state used in model creation to get reproducible model outputs

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
    if random_state == None:
        model = sklearn.cluster.KMeans(**hyperparameter_dict, max_iter=20,
                                   n_init=3, tol=1e-3)
    else:
        model = sklearn.cluster.KMeans(**hyperparameter_dict, max_iter=20,
                                   n_init=3, tol=1e-3, random_state = random_state)

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


def km_dbscan_wrapper(mask: np.ndarray, hyperparameter_dict, save_filepath: str, print_flag: bool = True):
    """
    Generates a fitted dbscan model and labels when provided a binary mask 
    2D array for the KMeans cluster with the highest contour count. A plot of 
    the dbscan clustering results is printed to the window, with a colorbar and 
    non-color bar version saved to the "ClusteringResults" directory as 
    "dbscan_result.jpg"
    
    Parameters
    -----
    binary_mask (np.ndarray): a binary mask with 1's corresponding to the pixels 
    involved in the cluser with the most contours and 0's for pixels not
    hyperparameter_dict: hyperparameters for dbscan model
    print_flag (bool): True for printing saved plot of dbscan model
    
    Returns
    -----
    all_labels (np.ndarray): labels of image after dbscan clustering for plotting
    dbscan (sklearn.cluster.DBSCAN): fitted dbscan model
    """    
    # Checking if the save directory exists
    if not os.path.exists(save_filepath) or not os.path.isdir(save_filepath):
        raise FileNotFoundError("Directory '{}' does not exist.".format(save_filepath))
    
    # Checking if the save directory is writable
    if not os.access(save_filepath, os.W_OK):
        raise PermissionError("Directory '{}' is not writable.".format(save_filepath))

    # Ensuring the mask is a 2D array
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Input 'mask' must be a 2D NumPy array.")

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
    plt.savefig(save_filepath + '/ClusteringResults/dbscan_result_colorbar.jpg')
    if print_flag == True:
        plt.show()
    else:
        plt.close()
        
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(all_labels, cmap='viridis');  # Change the colormap as needed
    plt.imsave(save_filepath + '/ClusteringResults/dbscan_result.jpg',all_labels)
    plt.close()
    
    return all_labels, dbscan


def kmean_to_spatial_model_superpatch_wrapper(superpatch_path: str,
                                            in_dir_path: str,
                                            spatial_hyperparameters: dict,
                                            n_clusters: list = [1,2,4,5,6,7,8,9],
                                            out_dir_path: str = None,
                                            save_TILs_overlay: bool = False,
                                            save_cluster_masks: bool = False,
                                            save_cluster_overlays: bool = False,
                                            save_all_clusters_img: bool = False,
                                            save_csv: bool = False,
                                            random_state: int = None):
    """
    A wrapper used to optimize a KMeans model on a superpatch to generate binary
    cluster masks for each sub-patch of the slide. These masks are 
    converted to dataframes (X pixel, Y pixel, binary mask value) and fed into a
    spatial algorithm (e.g Dbscan) to perform further segmentation on the highest 
    contour count cluster returned by segment_TILS for each path.

    Parameters
    -----
    superpatch_path: str
        filepath to superpatch image from preprocessing step (.tif)
    in_dir_path: str
        the directory path to the patches that will be clustered and have TILs
        segmented from superpatch model. This directory could be one that contains all the extracted patches
        containing significant amount of tissue using the tilseg.preprocessing module.
    spatial_hyperparameters: dict
        the spatial algorithm's optimized hyperparameters
    n_clusters: list
        a list of the number clusters to test in KMeans optimization
    out_dir: str
        the directory path where output images and CSV files will be saved
    save_TILs_overlay: bool
        generate image containing TILs overlayed on the original H&E patch
    save_cluster_masks: bool
        generate image showing binary segmentation masks of each cluster
    save_cluster_overlays: bool
        generate image containing individual clusters overlayed on the original
        patch
    save_all_clusters_img: bool
        generate image of all the clusters
    save_csv: bool
        generate CSV file containing countour information of each TIL segmented
        from the patch
    random_state: int
        random state to specify repeatable kmeans model

    Returns
    -----
    IM_labels_dict (dict): 
        labels from fitted spatial model as values and filenames as keys
    dbscan_fit (dict): 
        fitted spatial model object (sklearn.cluster.DBSCAN) as values and 
        filenames as keys
    cluster_mask_dict (dict): 
        dictionary containg the filenames of the patches without the extensions
        as the keys and the binary cluster masks from segment_TILS as the values
    cluster_index_dict (dict): 
        cluster labels from kemans that had the highest contour count in each image. 
        The keys are the filenames and the values are the cluster numbers.
    """
    # Checking if the in directory exists
    if not os.path.exists(in_dir_path) or not os.path.isdir(in_dir_path):
        raise FileNotFoundError("Directory '{}' does not exist.".format(in_dir_path))
    
    # Checking if the out directory exists
    if not os.path.exists(out_dir_path) or not os.path.isdir(out_dir_path):
        raise FileNotFoundError("Directory '{}' does not exist.".format(out_dir_path))
    
    # Checking if the out directory is writable
    if not os.access(out_dir_path, os.W_OK):
        raise PermissionError("Directory '{}' is not writable.".format(out_dir_path))    

    #Opens Superpatch Image / Retrieves Pixel Data
    img = Image.open(superpatch_path)
    numpy_img = np.array(img)
    numpy_img_reshape = np.float32(numpy_img.reshape((-1, 3))/255.)
    
    #Kmeans Optimizing
    t0 = time.time()
    hyperparameter_dict = opt_kmeans(numpy_img_reshape,n_clusters)
    tf = time.time()
    print(f"Found hyperparameters. Time took: {(tf-t0)/60} minutes.")
    
    #Kmeans Fitting
    kmeans_fit = KMeans_superpatch_fit(superpatch_path,hyperparameter_dict, random_state = random_state)
    tf2 = time.time()
    print(f"Completed Kmeans fitting. Time took: {(tf2-tf)/60} minutes.")
    
    #Run Segmentation on Kmeans Model
    TIL_count_dict, kmean_labels_dict,cluster_mask_dict, cluster_index_dict = segment_TILs(in_dir_path,
                 out_dir_path,
                 None,
                 'KMeans',
                 kmeans_fit,
                 save_TILs_overlay, 
                 save_cluster_masks,
                 save_cluster_overlays,
                 save_all_clusters_img,
                 save_csv)
    
    #Feed Resulting Cluster Mask in Spatial Model
    im_labels_dict = {}
    dbscan_fit_dict = {}
    
    for file in os.listdir(in_dir_path):
        if not file.lower().endswith(".tif"):
            continue

        #Dbcan Model Fitting
        tf3 = time.time()
        cluster_mask = cluster_mask_dict[file[:-4]]
        save_path = out_dir_path + f"/{file[:-4]}/"
        im_labels, dbscan_fit = km_dbscan_wrapper(mask = cluster_mask, hyperparameter_dict= spatial_hyperparameters,save_filepath=save_path, print_flag = False)
        
        #Addings Labels and Models to Dictionaries
        im_labels_dict[file[:-4]] = im_labels
        dbscan_fit_dict[file[:-4]] = dbscan_fit
        
        tf4 = time.time()
        print(f"Dbscan fitting time for file {file}: {tf4 - tf3} seconds.")

    return im_labels_dict, dbscan_fit_dict, cluster_mask_dict, cluster_index_dict
    
def kmean_to_spatial_model_patch_wrapper(patch_path: str,
                        spatial_hyperparameters: dict,
                        n_clusters: list = [1,2,3,4,5,6,7,8,9],
                        out_dir_path: str = None,
                        save_TILs_overlay: bool = False,
                        save_cluster_masks: bool = False,
                        save_cluster_overlays: bool = False,
                        save_all_clusters_img: bool = False,
                        save_csv: bool = False,
                        random_state: int = None):
    
    """
    A wrapper used to optimize a KMeans model on a patch to generate binary
    cluster mask. This mask is converted to a 3D array (X pixel, Y pixel, binary mask value)
    and fed into a spatial algorithm (e.g Dbscan) to perform further segmentation
    on the highest contour count cluster returned by segment_TILS. This function is used to
    generate a ground truth image for scoring (fit KMeans model to patch and predict on same patch)

    Parameters
    -----
    patch_path: str
        filepath to a single patch image from the preprocessing step (.tif)
    spatial_hyperparameters: dict
        the spatial algorithm's optimized hyperparameters
    n_clusters: list
        a list of the number clusters to test in KMeans optimization
    out_dir: str
        the directory path where output images and CSV files will be saved
    save_TILs_overlay: bool
        generate image containing TILs overlayed on the original H&E patch
    save_cluster_masks: bool
        generate image showing binary segmentation masks of each cluster
    save_cluster_overlays: bool
        generate image containing individual clusters overlayed on the original
        patch
    save_all_clusters_img: bool
        generate image of all the clusters
    save_csv: bool
        generate CSV file containing countour information of each TIL segmented
        from the patch
    random_state: int
        random state to specify repeatable kmeans model

    Returns
    -----
    IM_labels (np.ndarray): 
        labels from fitted spatial model
    dbscan_fit (sklearn.cluster.DBSCAN): 
        fitted spatial model object
    cluster_mask_dict (dict): 
        dictionary containg the filenames of the patches without the extensions
        as the keys and the binary cluster masks from segment_TILS as the values
    cluster_index (int): 
        cluster label from kemans that had the highest contour count. This is the
        cluster label that was fed into the spatial model for further classification.
    """
    
    #Opens Superpatch Image / Retrieves Pixel Data
    img = Image.open(patch_path)
    numpy_img = np.array(img)
    numpy_img_reshape = np.float32(numpy_img.reshape((-1, 3))/255.)
    
    #Kmeans Optimizing
    t0 = time.time()
    hyperparameter_dict = opt_kmeans(numpy_img_reshape,n_clusters)
    tf = time.time()
    print(f"Found hyperparameters. Time took: {(tf-t0)/60} minutes.")
    kmeans_fit = KMeans_superpatch_fit(patch_path,hyperparameter_dict, random_state)
    
    #Kmeans Fitting
    tf2 = time.time()
    print(f"Completed Kmeans fitting. Time took: {(tf2-tf)/60} minutes.")
    
    #Run Segmentation on Kmeans Model
    TIL_count_dict, kmean_labels_dict,cluster_mask_dict, cluster_index_dict = segment_TILs(patch_path,
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
        
    #Dbcan Model Fitting
    tf3 = time.time()
    file = os.path.basename(patch_path)
    cluster_mask = cluster_mask_dict[file[:-4]]
    cluster_index = cluster_index_dict[file[:-4]]
    save_path = out_dir_path + f"/{file[:-4]}/"
    im_labels, dbscan_fit = km_dbscan_wrapper(mask = cluster_mask, hyperparameter_dict= spatial_hyperparameters,save_filepath=save_path)
    tf4 = time.time()
    print(f"Script completed. Dbscan fitting time: {tf4 - tf3} seconds.")

    return im_labels, dbscan_fit, cluster_mask_dict, cluster_index


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
    