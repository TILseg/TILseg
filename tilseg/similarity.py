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
import cv2
from sklearn.metrics import mean_squared_error
from skimage.measure import regionprops, label
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
    # check if the inputs types are valid
    if(isinstance(mask1,np.ndarray)):
        pass
    else:
        raise TypeError('mask1 must be an np.ndarray')
    
    if(isinstance(mask2,np.ndarray)):
        pass
    else:
        raise TypeError('mask2 must be an np.ndarray')
    
    # check if the input arrays are in the same shape
    if(mask1.shape == mask2.shape):
        pass
    else:
        raise ValueError('mask1 and mask2 have different shapes.')
    
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
    reference_patch (str): path to the reference patch image
    output_path (str): path to the folder where images are saved
    reference_array (np.ndarray): the reference patch array

    Returns:
    None
    """
    # check if the inputs types are valid
    if(isinstance(superpatch_folder,str)):
        pass
    else:
        raise TypeError('superpatch_folder must be a string')
    
    if(isinstance(reference_patch,str)):
        pass
    else:
        raise TypeError('reference_patch must be a string')
    
    if(isinstance(output_path,str)):
        pass
    else:
        raise TypeError('output_path must be a string')
    if(isinstance(reference_array,np.ndarray)):
        pass
    else:
        raise TypeError('reference_array must be an np.ndarray')
        
    # check if the paths exist
    if os.path.exists(superpatch_folder):
        pass
    else:
        raise ValueError('System cannot find the superpatch_folder path. \
                         Please ensure os.stat() can run on your path.')
    if os.path.exists(reference_patch):
        pass
    else:
        raise ValueError('System cannot find the reference_patch. \
                         Please ensure os.stat() can run on your path.')
    if os.path.exists(output_path):
        pass
    else:
        raise ValueError('System cannot find the output_path path. \
                         Please ensure os.stat() can run on your path.')
    
    # check if the input folder is empty
    if(len(os.listdir(superpatch_folder))>0):
        pass
    else:
        raise ValueError('No superpatch found in superpatch_folder.')

    
    # iterate over files in folder
    for filename in os.listdir(superpatch_folder):
        
        # necessary in order to ignore ds store files on mac
        if filename.endswith('.tif'):
            
            # construct the full path of the current file
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
            
            # initialize empty array
            super_array_filtered = np.zeros_like(super_array)

            # use skimage's regionprops to get properties of connected regions
            labeled_regions = label(super_array)
            regions = regionprops(labeled_regions)

            # filter regions based on circularity and area of a single TIL
            for region in regions:
                area = region.area
                perimeter = region.perimeter
                # ensure that the denominator is not zero before calculating circularity
                if area != 0 and perimeter != 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if 200 < region.area < 2000 and circularity > 0.3:
                        # add the pixels of the qualified region to the TIL mask
                        super_array_filtered += (labeled_regions == region.label)

            # compute and print mse 
            mse, diff = image_similarity(reference_array, super_array_filtered)
            print(f'Mean squared error for superpatch {filename}: {round(mse, 3)}')
            
            # load original image
            original_image = cv2.imread(reference_image_path) 
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) 
            
            # find contours in the binary masks
            contours_green, _ = cv2.findContours(reference_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_red, _ = cv2.findContours(super_array_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # draw contours on the original image
            cv2.drawContours(original_image_rgb, contours_green, -1, (0, 255, 0), 2)  # green contours
            cv2.drawContours(original_image_rgb, contours_red, -1, (255, 0, 0), 2)    # red contours
            # display the result
            plt.imshow(original_image_rgb)
            plt.axis(‘off’)
            plt.show()