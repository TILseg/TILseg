"""
contains functions for computing the mean squared error and displaying
the image difference between two masks that have been trained on 
separate KMeans models. This is best done following superpatch creation 
(of varying patch number and size) using tilseg.preprocessing. 
"""

# imports
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tilseg.seg import KMeans_superpatch_fit, segment_TILs

def image_similarity(mask1, mask2):
    """
    calculates mean squared error and the image difference
    between two arrays

    Parameters
    ----
    mask1 (np.ndarray): array of first image
    mask2 (np.ndarray): array of second image

    Returns:
    mse (float): mean squared error
    diff (np.ndarray): image difference as numpy array
    """

    # calculate mse
    mse = mean_squared_error(mask1.flatten(), mask2.flatten())

    # calculate absolute difference 
    abs_diff = np.abs(mask1 - mask2)
    # # threshold the result, wrapped around 255s replaced by 1s
    diff = np.where(abs_diff > 1, 1, abs_diff)

    return mse, diff  

def superpatch_similarity(superpatch_folder, reference_patch, output_path, reference_array):
    """
    iterates through a folder of superpatches and calculates the mean squared
    error and plots an image of the difference between the superpatch mask 
    and reference mask

    Parameters:
    -----------
    superpatch_folder (str): path to the folder containing superpatch files

    reference_array (np.ndarray): the reference patch array

    Returns:
    None
    """

    # iterate over files in folder
    for filename in os.listdir(superpatch_folder):
        # construct the full path of the current file
        
        # necessary in order to ignore ds store files on mac
        if filename.endswith('.tif'):

            superpatch_file = os.path.join(superpatch_folder, filename)
            
            # fit KMeans model to superpatch
            model_superpatch = KMeans_superpatch_fit(
            patch_path = superpatch_file,
            hyperparameter_dict={'n_clusters': 4})
            
            # apply superpatch model to reference patch
            _, _, cluster_mask_dict_super = segment_TILs(reference_patch, 
                                                        output_path,
                                                        hyperparameter_dict=None, 
                                                        model=model_superpatch,
                                                        algorithm='KMeans',
                                                        save_TILs_overlay=False,
                                                        save_cluster_masks=False,
                                                        save_all_clusters_img=False,
                                                        save_csv=False,
                                                        multiple_images=False)
            
            # obtain first (and only) values from dictionary as an array
            super_array = next(iter(cluster_mask_dict_super.values()))

            # compute and print mse 
            mse, diff = image_similarity(reference_array, super_array)
            print(f'Mean squared error for superpatch {filename}: {round(mse, 3)}')

            # show difference image
            plt.matshow(diff, cmap='gray')
            plt.show() 