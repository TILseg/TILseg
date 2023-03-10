"""
Contains functions for fitting clustering models, predicting, and scoring based on a chosen clustering algorithm.
This is best done following hyperparameter optimization using tilseg.model_selection.
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

from tilseg.cluster_processing import image_postprocessing


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

    # Checking that the path to a patch is a string
    if type(patch_path) != str:
        raise TypeError('patch_path must be a string')
    else:
        pass

    # Checking that the patch_path actually exists
    path = pathlib.Path(patch_path)
    if not path.is_file():
        raise ValueError('Please input a path to a file that exists')
    else:
        pass

    # Creates a variable which references our preferred parameters for KMeans clustering
    key_list = list(hyperparameter_dict.keys())
    expected_key_list = ['n_clusters']
    # Checks that the expected keys are present in hyperparameters_dict
    if set(key_list) != set(expected_key_list):
        raise KeyError('Please enter the appropriate keys in hyperparameter_dict')
    else:
        pass

    # Checks that there are n_clusters is an integer and less than 9
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

    # Fits the model to our linearized and normalized patch data 
    model.fit(fit_patch_n)

    # Outputs KMeans model of the patch we want to cluster and will be used as input to 
    # clustering_score and segment_TILs functions
    return model


def clustering_score_KMeans_superpatch(patch_path: str,
                                model: sklearn.cluster._kmeans.KMeans=None,
                                gen_s_score: bool=False,
                                gen_ch_score: bool=True,
                                gen_db_score: bool=True):

    """
    Scores KMeans clustering model that has been fitted on a superpatch using various metrics

    Parameters
    -----
    model: sklearn.cluster._kmeans.KMeans
        the fitted KMeans model
    patch_path: str
        the directory path to the patch that will be clustered and scored on
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

    if type(patch_path) != str:
        raise TypeError('patch_path must be a string')
    else:
        pass

    path = pathlib.Path(patch_path)
    if not path.is_file():
        raise ValueError('Please input a path to a file that exists')
    else:
        pass

        # Reads the patch into a numpy uint8 array
    try:    
        pred_patch = plt.imread(patch_path)
    except UnidentifiedImageError:
        print('Please use an image that can be opened by PIL.Image.open')
        raise
    
    # Linearizes the array for R, G, and B separately and normalizes
    # The result is an N X 3 array where N=height*width of the patch in pixels
    pred_patch_n = np.float32(pred_patch.reshape((-1, 3))/255.)

    try:
        sklearn.utils.validation.check_is_fitted(model)
    except TypeError as te:
        raise TypeError('model is not a sklearn estimator')
    except NotFittedError:
        print('Please fit the first using tilseg.cluster.cluster_model_fitter')
        raise
    
    try:
        # Predicting the index/labels of the clusters on the fitted model from 'model' function
        # The result is an N X 3 array where N=height*width of the patch in pixels
        # Each value shows the label of the cluster that pixel belongs to
        labels = model.predict(pred_patch_n)
    except:
        raise ValueError('Please input a valid sklearn.cluster._kmeans.KMeans model. This can be produced using tilseg.seg.KMeans_superpatch_fit')

    # Scores the clustering based on various metrics
    # Note that calcualting silhoutte scores can take long times
    if gen_s_score:
        s_score = sklearn.metrics.silhouette_score(pred_patch.reshape((-1,3)), labels)
    else:
        s_score = None

    if gen_ch_score:
        ch_score = sklearn.metrics.calinski_harabasz_score(pred_patch.reshape((-1,3)), labels)
    else:
        ch_score = None

    if gen_db_score:
        db_score = sklearn.metrics.davies_bouldin_score(pred_patch.reshape((-1,3)), labels)
    else:
        db_score = None

    return s_score, ch_score, db_score


def clustering_score(patch_path: str,
                     hyperparameter_dict: dict,
                     algorithm: str='KMeans',
                     model: sklearn.cluster._kmeans.KMeans=None, 
                     gen_s_score: bool=False,
                     gen_ch_score: bool=True,
                     gen_db_score: bool=True):

    """
    Scores clustering models that have been fitted and predicted on a patch

    Parameters
    -----
    patch_path: str
        the directory path to the patch that will be fitted and/or clustered on to produce cluster labels that will be used for the scoring
    hyperparameter_dict: dict
        dicitonary of hyperparameters for the chosen algorithm
        this dictionary can be ready by the JSON file outputted by the model_selection module
        for KMeans: dictionary should have 'n_clusters' key
        for DBSCAN: dictionary should have 'eps' key
        for OPTICS: dictionary should have 'min_samples' and 'max_eps' keys
        for BIRCH: dictionary should have 'threshold', 'branching_factor', and 'n_clusters' keys
    algorithm: str
        the clustering algorithm to be used: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
    model: sklearn.cluster._kmeans.KMeans
        sklearn KMeans model fitted on a superpatch
        Only enter an input for model if the chosen algorith is KMeans the goal to score a model that has been fitted on a superpatch
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


    if type(patch_path) != str:
        raise TypeError('patch_path must be a string')
    else:
        pass

    path = pathlib.Path(patch_path)
    if not path.is_file():
        raise ValueError('Please input a path to a file that exists')
    else:
        pass

    if algorithm not in ['KMeans', 'DBSCAN', 'OPTICS', 'BIRCH']:
        raise ValueError('Please enter a valid clustering algorithm')
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

        if algorithm != 'KMeans':
            raise ValueError('Can only receive argument for model if algorithm is KMeans')
        else:
            pass

        if type(model) != sklearn.cluster._kmeans.KMeans:
            raise TypeError('Please input a valid sklearn.cluster._kmeans.KMeans model. You can fit a model to a superpatch using KMeans_superpatch_fit function.')
        else:
            pass

        try:
            sklearn.utils.validation.check_is_fitted(model)
        except TypeError:
            print('model is not a sklearn estimator')
            raise
        except NotFittedError:
            print('Please fit the first using tilseg.cluster.cluster_model_fitter')
            raise

        try:
            # Predicting the index/labels of the clusters on the fitted model from 'model' function
            # The result is an N X 3 array where N=height*width of the patch in pixels
            # Each value shows the label of the cluster that pixel belongs to
            labels = model.predict(pred_patch_n)
        except:
            raise ValueError('Please input a valid sklearn.cluster._kmeans.KMeans model. You can fit a model to a superpatch using KMeans_superpatch_fit function.')

    else:

        if algorithm == 'KMeans':
            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['n_clusters']
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using KMeans')
            else:
                pass

            if type(hyperparameter_dict['n_clusters']) != int or hyperparameter_dict['n_clusters'] > 8:
                raise ValueError('Please enter an integer less than 9 for n_clusters')
            else:
                pass

            model = sklearn.cluster.KMeans(**hyperparameter_dict, max_iter=20,
                        n_init=3, tol=1e-3)
            labels = model.fit_predict(pred_patch_n)

        elif algorithm == 'DBSCAN':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['eps']
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using DBSCAN')
            else:
                pass

            if type(hyperparameter_dict['eps']) != int or type(hyperparameter_dict['eps']) != float:
                raise ValueError('Please enter an integer or float for eps')
            else:
                pass

            model = sklearn.cluster.DBSCAN(**hyperparameter_dict)

        elif algorithm == 'OPTICS':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['min_samples', 'max_eps']
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using OPTICS')
            else:
                pass

            if type(hyperparameter_dict['min_samples']) != int or type(hyperparameter_dict['min_samples']) != float:
                raise ValueError('Please enter an integer or float for min_samples')
            else:
                pass

            if type(hyperparameter_dict['max_eps']) != int or type(hyperparameter_dict['max_eps']) != float or hyperparameter_dict['max_eps'] != np.inf:
                raise ValueError('Please enter an integer, float, or numpy.inf for max_eps')
            else:
                pass

            model = sklearn.cluster.OPTICS(**hyperparameter_dict)

        elif algorithm == 'BIRCH':

            key_list = list(hyperparameter_dict.keys())
            expected_key_list = ['threshold', 'branching_factor', 'n_clusters']
            if set(key_list) != set(expected_key_list):
                raise KeyError('Please enter the appropriate keys in hyperparameter_dict if using BIRCH')
            else:
                pass

            if type(hyperparameter_dict['threshold']) != int or type(hyperparameter_dict['threshold']) != float:
                raise ValueError('Please enter an integer or float for threshold')
            else:
                pass

            if type(hyperparameter_dict['branching_factor']) != int:
                raise ValueError('Please enter an integer for branch_factor')
            else:
                pass

            if type(hyperparameter_dict['n_clusters']) != int or hyperparameter_dict['n_clusters'] != None:
                raise ValueError('Please enter an integer or None for n_clusters')
            else:
                pass

            model = sklearn.cluster.Birch(**hyperparameter_dict)

        else:
            raise ValueError('Please enter a valid clustering algorithm')
        
    labels = model.fit_predict(pred_patch_n)
        
    # Scores the clustering based on various metrics
    # Note that calcualting silhoutte scores can take long times
    if gen_s_score:
        s_score = sklearn.metrics.silhouette_score(pred_patch.reshape((-1,3)), labels)
    else:
        s_score = None

    if gen_ch_score:
        ch_score = sklearn.metrics.calinski_harabasz_score(pred_patch.reshape((-1,3)), labels)
    else:
        ch_score = None

    if gen_db_score:
        db_score = sklearn.metrics.davies_bouldin_score(pred_patch.reshape((-1,3)), labels)
    else:
        db_score = None

    return s_score, ch_score, db_score


def segment_TILs(model: sklearn.base.ClusterMixin, 
                     in_dir_path: str, 
                     out_dir_path: str=None,
                     algorithm: str='KMeans',
                     save_TILs_overlay: bool=False,
                     save_cluster_masks: bool=False,
                     save_cluster_overlays: bool=False,
                     save_all_clusters_img: bool=False,
                     save_csv: bool=False):

    """
    Applies a fitted clustering model to patches and generates multiple images: binary segmentation masks of each cluster, 
    segmentation masks overlaid on the original patch, and all clusters overlaid on the original patch

    Parameters
    -----
    model: sklearn.cluster.model
        the fitted model
    algorithm: str
        the clustering algorithm to be used: 'KMeans', '', ''
    in_dir: str
        the directory path to the patches that will be predicted and clustered
    out_dir: str
        the directory path where output images will be saved
    """

    if not os.path.isdir(in_dir_path):
        raise ValueError('Please enter a valid input directory')
    else:
        pass

    if out_dir_path != None:
        if not os.path.isdir(out_dir_path):
            raise ValueError('Please enter a valid output directory')
        else:
            pass
    else:
        pass

    # Initializing dicitonary with the count of the TILs in each path in the input directory
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
            sklearn.utils.validation.check_is_fitted(model)
        except TypeError:
            print('model is not an estimator')
            raise
        except NotFittedError:
            print('Please fit the first using tilseg.cluster.cluster_model_fitter')
            raise
    
        # Reads the current patch into a numpy uint8 array 
        pred_patch = plt.imread(os.path.join(in_dir_path, file))
        # Linearizes the array for R, G, and B separately and normalizes
        # The result is an N X 3 array where N=height*width of the patch in pixels
        pred_patch_n = np.float32(pred_patch.reshape((-1, 3))/255.)

        try:
            # Predicting the index/labels of the clusters on the fitted model from 'model' function
            # The result is an N X 3 array where N=height*width of the patch in pixels
            # Each value shows the label of the cluster that pixel belongs to
            labels = model.predict(pred_patch_n)
        except:
            print('Please input a valid sklearn.cluster.model for model. This can be produced using tilseg.cluster.cluster_model_fitter')
            raise

        # Makes sure that the model is training for 8 clusters
        if len(np.unique(labels)) <= 8:
            pass
        else:
            print("Looks like the model is being trained for more than 8 clusters. Please consider training it on less number of clusters.")
            raise

        TIL_count = image_postprocessing(clusters=labels.reshape(pred_patch.shape[0], pred_patch.shape[1]),
                         ori_img=pred_patch,
                         filepath= os.path.join(out_dir_path, file[:-4]), 
                         gen_all_clusters=save_all_clusters_img,
                         gen_overlays=save_cluster_overlays, 
                         gen_tils=save_TILs_overlay,
                         gen_masks=save_cluster_masks, 
                         gen_csv=save_csv)

    TIL_count_dict[file[:-4]] = TIL_count

    return TIL_count_dict