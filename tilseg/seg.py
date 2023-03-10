"""
Contains functions for fitting and scoring clustering models and segmenting TILs in H&E patches
This is best done following hyperparameter optimization using tilseg.model_selection
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.metrics
import sklearn.utils.validation
import pandas as pd
import cv2
import time
import PIL
import os
from skimage import io
import pathlib
from PIL import UnidentifiedImageError
from sklearn.exceptions import NotFittedError

from cluster_processing import image_postprocessing


def KMeans_superpatch_fit(patch_path: str, hyperparameter_dict: dict):

    """
    Fits a KMeans clustering model to a patch that will be used to cluster other patches
    KMeans is the only clustering algorithms that allows fitting a model to one patch clustering on another
    All other clustering algorithms need to be fitted on the same patch that needs to be clustered
    It makes sense to use this function to fit a KMeans clustering model to a superpatch that can 
    capture H&E stain variation

    Parameters
    -----
    patch_path: str
        the directory path to the patch that the model will be fitted to obtain cluster decision boundaries
    hyperparameter_dict: dict
        dicitonary of hyperparameters for KMeans containing 'n_clusters' as the only key
        this dictionary can be obtained by reading the JSON file outputted by the model_selection module

    Returns
    -----
    model: sklearn.base.ClusterMixin
        the fitted model
    """

    # Checks that the path to the patch is a string
    if type(patch_path) != str:
        raise TypeError('patch_path must be a string')
    else:
        pass

    # Checks that the patch_path actually exists
    path = pathlib.Path(patch_path)
    if not path.is_file():
        raise ValueError('Please input a path to a file that exists')
    else:
        pass

    # Checking that hyperparameter_dict is a dictionary
    if type(hyperparameter_dict) != dict:
        raise TypeError('hyperparameter_dict must be a dictionary')
    else:
        pass

    # Creates a variable which references the preferred parameters for KMeans clustering
    key_list = list(hyperparameter_dict.keys())
    expected_key_list = ['n_clusters']
    # Checks that the expected keys are present in hyperparameters_dict
    if set(key_list) != set(expected_key_list):
        raise KeyError('Please enter the appropriate keys in hyperparameter_dict')
    else:
        pass

    # Checks that n_clusters is an integer and less than 9
    if type(hyperparameter_dict['n_clusters']) != int or hyperparameter_dict['n_clusters'] > 8:
        raise ValueError('Please enter an integer less than 9 for n_clusters')
    else:
        pass

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

    # Outputs KMeans model fitted to the superpatch and will be used as input to 
    # clustering_score and segment_TILs functions
    return model


def clustering_score(patch_path: str,
                     hyperparameter_dict: dict=None,
                     algorithm: str='KMeans',
                     model: sklearn.cluster._kmeans.KMeans=None, 
                     gen_s_score: bool=False,
                     gen_ch_score: bool=True,
                     gen_db_score: bool=True):

    """
    Scores clustering models that have been fitted and predicted on a patch
    The motive of this function is to test out clustering algorithms on a single patch 
    The goal of this function is NOT to get high throughput scores from multiple patches in a whole slide image

    Parameters
    -----
    patch_path: str
        the directory path to the patch that will be fitted and/or clustered on to produce cluster labels that will be used for the scoring
    hyperparameter_dict: dict
        dicitonary of hyperparameters for the chosen algorithm
        this dictionary can be read by the JSON file outputted by the model_selection module
        for KMeans: dictionary should have 'n_clusters' key
        for DBSCAN: dictionary should have 'eps' key
        for OPTICS: dictionary should have 'min_samples' and 'max_eps' keys
        for BIRCH: dictionary should have 'threshold', 'branching_factor', and 'n_clusters' keys
    algorithm: str
        the clustering algorithm to be used: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
    model: sklearn.cluster._kmeans.KMeans
        sklearn KMeans model fitted on a superpatch
        Only enter an input for model if the chosen algorithm is KMeans and the goal is to score a model that has been fitted on a superpatch
        If no model is inputted, the clustering algorthim will fit a model on the patch found at patch_path
    gen_s_score: bool
        generate Silhouette score
    gen_ch_score: bool
        Calinski-Harabasz index
    gen_db_score: bool
        generate Davies-Bouldin score

    Returns
    -----
    s_score: float
        Silhouette score: ranges from -1 to 1 with 0 meaning indiffirent clusters, -1 meaning clustered assigned in a wrong fashion and 1 meaning far apart
        and well separated clusters
    ch_score: float
        Calinski-Harabasz index: higher value of ch_score means the clusters are dense and well separated- there is no absolute cut-off value
    db_score: float
        Davies-Bouldin score: lower values mean better clustering with zero being the minimum value
    """

    # Checks that the path to the patch is a string
    if type(patch_path) != str:
        raise TypeError('patch_path must be a string')
    else:
        pass

    # Checks that the patch_path actually exists
    path = pathlib.Path(patch_path)
    if not path.is_file():
        raise ValueError('Please input a path to a file that exists')
    else:
        pass

    # Checks that the inputted algorithm is one that can be supported by this function
    # Also checks for string as the type and any typos in the input
    if algorithm not in ['KMeans', 'DBSCAN', 'OPTICS', 'BIRCH']:
        raise ValueError('Please enter a valid clustering algorithm')
    else:
        pass

    # Checks that gen_s_score is a boolean
    if type(gen_s_score) != bool:
        raise ValueError('gen_s_score must be a boolean')
    else:
        pass

    # Checks that gen_ch_score is a boolean
    if type(gen_ch_score) != bool:
        raise ValueError('gen_ch_score must be a boolean')
    else:
        pass
    # Checks that gen_db_score is a boolean
    if type(gen_db_score) != bool:
        raise ValueError('gen_db_score must be a boolean')
    else:
        pass

    try:
        # Reads the patch into a numpy uint8 array    
        pred_patch = plt.imread(patch_path)
    # Makes sure that the fie is readable by matplotlib, which uses PIL
    except UnidentifiedImageError:
        print('Please use an image that can be opened by PIL.Image.open')
        raise

    # Linearizes the array for R, G, and B separately and normalizes
    # The result is an N X 3 array where N=height*width of the patch in pixels
    pred_patch_n = np.float32(pred_patch.reshape((-1, 3))/255.)

    # If scoring a model that has already been fitted to a superpatch
    if model != None:

        # Checks that the the algorithm is KMeans when model is not None
        # KMeans is the only sklearn model that allows fitting to one dataset and predicting
        # on another
        if algorithm != 'KMeans':
            raise ValueError('Can only receive argument for model if algorithm is KMeans')
        else:
            pass

        # Checks that the inputted model is a sklearn KMeans model
        if type(model) != sklearn.cluster._kmeans.KMeans:
            raise TypeError('Please input a valid sklearn.cluster._kmeans.KMeans model. You can fit a model to a superpatch using KMeans_superpatch_fit function.')
        else:
            pass

        try:
            sklearn.utils.validation.check_is_fitted(model)
        # Checks that the model is a sklearn estimator
        except TypeError:
            print('model is not a sklearn estimator')
            raise
        # Checks that the model has been fitted before it is called to predict
        except NotFittedError:
            print('Please fit the first using tilseg.seg.cluster_model_fitter')
            raise

        # Checks that hyperparameter_dict is None when using a model fitted on a superpatch
        # All the hyperparameters were specified during the fitting of the model
        if hyperparameter_dict != None:
            raise ValueError('hyperparameters_dict must be None since a fitted model is already inputted')
        else:
            pass

        try:
            # Predicting the index/labels of the clusters on the fitted model from 'model' function
            # The result is an N X 3 array where N=height*width of the patch in pixels
            # Each value shows the label of the cluster that pixel belongs to
            labels = model.predict(pred_patch_n)
        # Catches an exception in case the predict attribute of the model does not work on the patch
        except:
            raise ValueError('Please input a valid sklearn.cluster._kmeans.KMeans model. You can fit a model to a superpatch using tilseg.seg.KMeans_superpatch_fit function.')

    # If scoring on a model that is to be fitted on the same patch that it is supposed to cluster
    else:

        # Checks that hyperparameter_dict is a dictionary
        # Hyperparameters should be specified in this case because the fitting is also happening
        # before prediction/clustering
        if type(hyperparameter_dict) != dict:
            raise TypeError('hyperparameter_dict must be a dictionary')
        else:
            pass

        if algorithm == 'KMeans':
            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['n_clusters']
            # Checks that the expected keys are present in hyperparameter_dict for KMeans as the algorithm
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using KMeans')
            else:
                pass

            # Checks that n_clusters hyperparamter is an integer less than 9
            if type(hyperparameter_dict['n_clusters']) != int or hyperparameter_dict['n_clusters'] > 8:
                raise ValueError('Please enter an integer less than 9 for n_clusters')
            else:
                pass

            # Initializes a KMeans model with optimized, fixed, and other default hyperparameters
            # max_iter and n_init were chosen after rounds of optimization to minimize compute
            # time while not compromising on the quality of the clustering on H&E datasets
            model = sklearn.cluster.KMeans(**hyperparameter_dict, max_iter=20,
                        n_init=3, tol=1e-3)

        elif algorithm == 'DBSCAN':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['eps']
            # Checks that the expected keys are present in hyperparameter_dict for DBSCAN as the algorithm
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using DBSCAN')
            else:
                pass

            # Checking that eps is an integer or a float
            if type(hyperparameter_dict['eps']) != int or type(hyperparameter_dict['eps']) != float:
                raise ValueError('Please enter an integer or float for eps')
            else:
                pass
            # Initializes a DBSCAN model with optimized and other default hyperparameters
            model = sklearn.cluster.DBSCAN(**hyperparameter_dict)

        elif algorithm == 'OPTICS':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['min_samples', 'max_eps']
            # Checks that the expected keys are present in hyperparameter_dict for OPTICS as the algorithm
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using OPTICS')
            else:
                pass

            # Checks that min_samples is an integer or a float
            if type(hyperparameter_dict['min_samples']) != int or type(hyperparameter_dict['min_samples']) != float:
                raise ValueError('Please enter an integer or float for min_samples')
            else:
                pass

            # Checks that max_eps is an integer, float or numpy.infinity
            if type(hyperparameter_dict['max_eps']) != int or type(hyperparameter_dict['max_eps']) != float or hyperparameter_dict['max_eps'] != np.inf:
                raise ValueError('Please enter an integer, float, or numpy.inf for max_eps')
            else:
                pass

            # Initializes an OPTICS model with optimized and other default hyperparameters
            model = sklearn.cluster.OPTICS(**hyperparameter_dict)

        elif algorithm == 'BIRCH':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['threshold', 'branching_factor', 'n_clusters']
            # Checks that the expected keys are present in hyperparameter_dict for BIRCH as the algorithm
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using BIRCH')
            else:
                pass

            # Checks that threshold is an integer or a float
            if type(hyperparameter_dict['threshold']) != int or type(hyperparameter_dict['threshold']) != float:
                raise ValueError('Please enter an integer or float for threshold')
            else:
                pass

            # Checks that branching_factor is an integer
            if type(hyperparameter_dict['branching_factor']) != int:
                raise ValueError('Please enter an integer for branch_factor')
            else:
                pass

            # Checks that n_clusters is an integer or None
            if type(hyperparameter_dict['n_clusters']) != int or hyperparameter_dict['n_clusters'] != None:
                raise ValueError('Please enter an integer or None for n_clusters')
            else:
                pass

            # Initializes a BIRCH model with optimized and other default hyperparameter
            model = sklearn.cluster.Birch(**hyperparameter_dict)

        # If algorithm inputted is not one of the accepted strings
        else:
            raise ValueError('Please enter a valid clustering algorithm')

        # Fitting and clustering on the linearized and normalized RGB pixel data       
        labels = model.fit_predict(pred_patch_n)
        
    # Scores the clustering based on various metrics
    # Note that calcualting silhoutte scores can take long times so gen_s_score default is False
    # Generates Silhouette score if gen_s_score is true
    if gen_s_score:
        s_score = sklearn.metrics.silhouette_score(pred_patch.reshape((-1,3)), labels)
    else:
        s_score = None

    # Generates Calinski-Harabasz index if gen_ch_score is true
    if gen_ch_score:
        ch_score = sklearn.metrics.calinski_harabasz_score(pred_patch.reshape((-1,3)), labels)
    else:
        ch_score = None
    # Generates Davies-Bouldin score if gen_db_score is true
    if gen_db_score:
        db_score = sklearn.metrics.davies_bouldin_score(pred_patch.reshape((-1,3)), labels)
    else:
        db_score = None

    # Returns Silhouette score, Calinski-Harabasz index, and Davies-Bouldin score
    return s_score, ch_score, db_score


def segment_TILs(in_dir_path: str, 
                     out_dir_path: str=None,
                     hyperparameter_dict: dict=None,
                     algorithm: str='KMeans',
                     model: sklearn.cluster._kmeans.KMeans=None,
                     save_TILs_overlay: bool=False,
                     save_cluster_masks: bool=False,
                     save_cluster_overlays: bool=False,
                     save_all_clusters_img: bool=False,
                     save_csv: bool=False):

    """
    Applies a clustering model to patches and generates multiple files: TILs overlayed on the original 
    H&E patch, binary segmentation masks of each cluster, individual clusters overlayed on the original patch, 
    image of all the clusters, and a CSV file containing countour information of each TIL segmented from the patch

    Parameters
    -----
    in_dir_path: str
        the directory path to the patches that will clustered and have TILs segmented
        This directory could be one that contains all the extracted patches containing significant amount of 
        tissue using the tilseg.preprocessing module
    out_dir: str
        the directory path where output images and CSV files will be saved
    hyperparameter_dict: dict
        dicitonary of hyperparameters for the chosen algorithm
        this dictionary can be read by the JSON file outputted by the model_selection module
        for KMeans: dictionary should have 'n_clusters' key
        for DBSCAN: dictionary should have 'eps' key
        for OPTICS: dictionary should have 'min_samples' and 'max_eps' keys
        for BIRCH: dictionary should have 'threshold', 'branching_factor', and 'n_clusters' keys
    algorithm: str
        the clustering algorithm to be used: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
    model: sklearn.cluster._kmeans.KMeans
        sklearn KMeans model fitted on a superpatch
        Only enter an input for model if the chosen algorithm is KMeans and the goal to cluster all the patches
        using a model fitted on a superpatch
        If no model is inputted, the clustering algorthim will fit a model on the same patch that it is clustering
    save_TILs_overlay: bool
        generate image containing TILs overlayed on the original H&E patch
    save_cluster_masks: bool
        generate image showing binary segmentation masks of each cluster
    save_cluster_overlays: bool
        generate image containing individual clusters overlayed on the original patch
    save_all_clusters_img: bool
        generate image of all the clusters
    save_csv: bool
        generate CSV file containing countour information of each TIL segmented from the patch

    Returns
    -----
    TIL_count_dict: dict
        contains patch filenames without the extension as the key and TIL counts in respective pathces as the values
    """

    # Checks that the path to the input directory is a string
    if type(in_dir_path) != str:
        raise TypeError('patch_path must be a string')
    else:
        pass

    # Checks that teh input directory actually exists
    if not os.path.isdir(in_dir_path):
        raise ValueError('Please enter a valid input directory')
    else:
        pass

    # Checks that save_TILs_overlay is a boolean
    if type(save_TILs_overlay) != bool:
        raise ValueError('save_TILs_overlay must be a boolean')
    else:
        pass

    # Checks that save_cluster_masks is a boolean
    if type(save_cluster_masks) != bool:
        raise ValueError('save_cluster_masks must be a boolean')
    else:
        pass

    # Checks that save_cluster_overlays is a boolean
    if type(save_cluster_overlays) != bool:
        raise ValueError('save_cluster_overlays must be a boolean')
    else:
        pass

    # Checks that save_all_clusters_img is a boolean
    if type(save_all_clusters_img) != bool:
        raise ValueError('save_all_clusters_img must be a boolean')
    else:
        pass

    # Checks that save_csv is a boolean
    if type(save_csv) != bool:
        raise ValueError('save_csv must be a boolean')
    else:
        pass

    # Conditions when at least one boolean for files to be saved is True
    if save_TILs_overlay or save_cluster_masks or save_cluster_overlays or save_all_clusters_img or save_csv:
        # Checks that the output directory path is a string
        if type(out_dir_path) != str:
            raise TypeError('patch_path must be a string')
        else:
            pass
        
        # Checks that the output directory actually exists
        if not os.path.isdir(out_dir_path):
            raise ValueError('Please enter a valid output directory')
        else:
            pass
    else:
        pass

    # Condition for when model is not None and the user is trying to input a pre-fitted (e.g. on a superpatch) model 
    if model != None:

        # Checks that the the algorithm is KMeans when model is not None
        # KMeans is the only sklearn model that allows fitting to one dataset and predicting on another
        if algorithm != 'KMeans':
            raise ValueError('Can only receive argument for model if algorithm is KMeans')
        else:
            pass

        # Checks that the inputted model is a sklearn KMeans model
        if type(model) != sklearn.cluster._kmeans.KMeans:
            raise TypeError('Please input a valid sklearn.cluster._kmeans.KMeans model. You can fit a model to a superpatch using KMeans_superpatch_fit function.')
        else:
            pass

        try:
            sklearn.utils.validation.check_is_fitted(model)
        except TypeError:
            # Checks that the model is a sklearn estimator
            print('model is not a sklearn estimator')
            raise
            # Checks that the model has been fitted before it is called to predict
        except NotFittedError:
            print('Please fit the first using tilseg.seg.KMeans_superpatch_fit')
            raise

        # Checks that hyperparameter_dict is None when using a model fitted on a superpatch
        # All the hyperparameters were specified during the fitting of the model
        if hyperparameter_dict != None:
            raise ValueError('hyperparameters_dict must be None since a fitted model is already inputted')
        else:
            pass

        # new_model is a model that will be used to call fit_predict for models that have not already been fitted
        new_model = None

    else:

        # Checks that hyperparameter_dict is a dictionary
        if type(hyperparameter_dict) != dict:
            raise TypeError('hyperparameter_dict must be a dictionary')
        else:
            pass

        if algorithm == 'KMeans':
            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['n_clusters']
            # Checks that the expected keys are present in hyperparameter_dict for KMeans as the algorithm
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using KMeans')
            else:
                pass

            # Checks that n_clusters hyperparamter is an integer less than 9
            if type(hyperparameter_dict['n_clusters']) != int or hyperparameter_dict['n_clusters'] > 8:
                raise ValueError('Please enter an integer less than 9 for n_clusters')
            else:
                pass

            # Initializes a KMeans model with optimized, fixed, and other default hyperparameters
            # max_iter and n_init were chosen after rounds of optimization to minimize compute 
            # time while not compromising on the quality of the clustering on H&E datasets
            # new_model is a model that will be used to call fit_predict for models that have not already been fitted
            new_model = sklearn.cluster.KMeans(**hyperparameter_dict, max_iter=20,
                        n_init=3, tol=1e-3)

        elif algorithm == 'DBSCAN':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['eps']
            # Checks that the expected keys are present in hyperparameter_dict for DBSCAN as the algorithm
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using DBSCAN')
            else:
                pass

            # Checking that eps is an integer or a float
            if type(hyperparameter_dict['eps']) != int or type(hyperparameter_dict['eps']) != float:
                raise ValueError('Please enter an integer or float for eps')
            else:
                pass

            # Initializes a DBSCAN model with optimized and other default hyperparameters
            # new_model is a model that will be used to call fit_predict for models that have not already been fitted
            new_model = sklearn.cluster.DBSCAN(**hyperparameter_dict)

        elif algorithm == 'OPTICS':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['min_samples', 'max_eps']
            if set(key_list) != set(expected_key_list):
            # Checks that the expected keys are present in hyperparameter_dict for OPTICS as the algorithm
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using OPTICS')
            else:
                pass

            # Checks that min_samples is an integer or a float
            if type(hyperparameter_dict['min_samples']) != int or type(hyperparameter_dict['min_samples']) != float:
                raise ValueError('Please enter an integer or float for min_samples')
            else:
                pass

            # Checks that max_eps is an integer, float or numpy.infinity
            if type(hyperparameter_dict['max_eps']) != int or type(hyperparameter_dict['max_eps']) != float or hyperparameter_dict['max_eps'] != np.inf:
                raise ValueError('Please enter an integer, float, or numpy.inf for max_eps')
            else:
                pass

            # Initializes an OPTICS model with optimized and other default hyperparameters
            # new_model is a model that will be used to call fit_predict for models that have not already been fitted
            new_model = sklearn.cluster.OPTICS(**hyperparameter_dict)

        elif algorithm == 'BIRCH':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['threshold', 'branching_factor', 'n_clusters']
          # Checks that the expected keys are present in hyperparameter_dict for BIRCH as the algorithm
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using BIRCH')
            else:
                pass

            # Checks that threshold is an integer or a float
            if type(hyperparameter_dict['threshold']) != int or type(hyperparameter_dict['threshold']) != float:
                raise ValueError('Please enter an integer or float for threshold')
            else:
                pass

            # Checks that branching_factor is an integer
            if type(hyperparameter_dict['branching_factor']) != int:
                raise ValueError('Please enter an integer for branch_factor')
            else:
                pass

            # Checks that n_clusters is an integer or None
            if type(hyperparameter_dict['n_clusters']) != int or hyperparameter_dict['n_clusters'] != None:
                raise ValueError('Please enter an integer or None for n_clusters')
            else:
                pass

            # Initializes a BIRCH model with optimized and other default hyperparameter
            # new_model is a model that will be used to call fit_predict for models that have not already been fitted
            new_model = sklearn.cluster.Birch(**hyperparameter_dict)

        # If algorithm inputted is not one of the accepted strings
        else:
            raise ValueError('Please enter a valid clustering algorithm')

    # Initializing dicitonary with the count of the TILs in each patch in the input directory
    TIL_count_dict = {}

    # Iterating over every patch in the directory
    for file in os.listdir(in_dir_path):
        
        # Creating a directory with the same file name (without extenstion)
        # Passing if such a directory already exists
        try:
            os.mkdir(os.path.join(out_dir_path, file[:-4]))
        except:
            pass

        try:
            # Reads the patch into a numpy uint8 array    
            pred_patch = plt.imread(os.path.join(in_dir_path, file))
        # Makes sure that the fie is readable by matplotlib, which uses PIL
        except UnidentifiedImageError:
            print('Please use an image that can be opened by PIL.Image.open')
            raise

        # Linearizes the array for R, G, and B separately and normalizes
        # The result is an N X 3 array where N=height*width of the patch in pixels
        pred_patch_n = np.float32(pred_patch.reshape((-1, 3))/255.)

        # Condition when user has inputted a pre-fitted (e.g. on a superpatch) model
        if new_model == None:

            try:
                # Predicting the index/labels of the clusters on the fitted model from 'model' function
                # The result is an N X 3 array where N=height*width of the patch in pixels
                # Each value shows the label of the cluster that pixel belongs to
                labels = model.predict(pred_patch_n)
            except:
                # Catches an exception in case the predict attribute of the model does not work on the patch
                raise ValueError('Please input a valid sklearn.cluster._kmeans.KMeans model. You can fit a model to a superpatch using tilseg.seg.KMeans_superpatch_fit function.')

        # Condition when user has inputted an unfitted model
        else:
            
            # fitting a model and clustering on the same patch
            # Predicting the index/labels of the clusters on the fitted model from 'model' function
            # The result is an N X 3 array where N=height*width of the patch in pixels
            # Each value shows the label of the cluster that pixel belongs to
            labels = new_model.fit_predict(pred_patch_n)

        # Makes sure that the model is training for less than 9 clusters
        if len(np.unique(labels)) <= 8:
            pass
        else:
            raise ValueError("Looks like the model is being trained for more than 8 clusters. This is unlikely to happen with H&E images")

        # function imported from tilseg.cluster_processing module which produces the images and CSV files and counts the TILs
        TIL_count = image_postprocessing(clusters=labels.reshape(pred_patch.shape[0], pred_patch.shape[1]),
                         ori_img=pred_patch,
                         filepath= os.path.join(out_dir_path, file[:-4]), 
                         gen_all_clusters=save_all_clusters_img,
                         gen_overlays=save_cluster_overlays, 
                         gen_tils=save_TILs_overlay,
                         gen_masks=save_cluster_masks, 
                         gen_csv=save_csv)

    # appends the TILs count from the current patch to the dictionary as a value to a key that is the patch's file name without the extension
    TIL_count_dict[file[:-4]] = TIL_count

    # returns the dictionary containing patch filenames without the extension as the key and TIL counts as the values
    return TIL_count_dict