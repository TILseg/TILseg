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


def cluster_model_fit(patch_path: str, 
                         algorithm: str='KMeans',
                         n_clusters: int=None): 

    """
    Fits a model using a chosen clustering algorithm

    Parameters
    -----
    patch_path: str
        the directory path to the patch that the model will be fitted to obtain cluster decision boundaries
    algorithm: str
        the clustering algorithm to be used: 'KMeans', '', ''
    n_clusters: int
        number of clusters in KMeans clustering

    Returns
    -----
    model: sklearn.cluster.model
        the fitted model
    """

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

    if algorithm not in ['KMeans']:
        raise ValueError('Please enter a valid clustering algorithm')
    else:
        pass

    if algorithm == 'KMeans' and n_clusters == None:
        raise ValueError('Please enter a number of clusters for KMeans clustering')
    else:
        pass

    if algorithm != 'KMeans' and n_clusters != None:
        raise ValueError('Can only specify number of clusters for KMeans clustering')
    else:
        pass

    if type(n_clusters) != int or n_clusters > 8:
        raise ValueError('Please enter an integer less than 9 for n_clusters')
    else:
        pass

    # Creates a variable which references our preferred parameters for KMeans clustering
    if algorithm == 'KMeans':

        model = sklearn.cluster.KMeans(n_clusters, max_iter=20,
                    n_init=3, tol=1e-3)
        
        # Reads the patch into a numpy uint8 array
        try:    
            fit_patch = plt.imread(patch_path)
        except UnidentifiedImageError:
            # raise UnidentifiedImageError('Please use an image that can be opened by PIL.Image.open')
            print('Please use an image that can be opened by PIL.Image.open')
            raise
        
        # Linearizes the array for R, G, and B separately and normalizes
        # The result is an N X 3 array where N=height*width of the patch in pixels
        fit_patch_n = np.float32(fit_patch.reshape((-1, 3))/255.)

        # Fits the model to our linearized and normalized patch data 
        model.fit(fit_patch_n)

    else:

        model = None

    # Outputs our specific model of the patch we want to cluster and will be used as input to pred_and_cluster function below
    return model


def clustering_score(model: sklearn.base.ClusterMixin, 
                     patch_path: str,
                     gen_s_score: bool=False,
                     gen_ch_score: bool=True,
                     gen_db_score: bool=True):

    """
    Scores the clustering using various metrics

    Parameters
    -----
    model: sklearn.base.ClusterMixin
        the fitted model
    patch_path: str
        the directory path to the patch that will be predicted and clustered

    Returns
    -----
    ch_score: float
        Calinski-Harabasz Index: Higher value of ch_score means the clusters are dense and well separated- there is no absolute cut-off value
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
        raise ValueError('Please input a valid sklearn.cluster.model for model. This can be produced using tilseg.cluster.cluster_model_fitter')

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

    return TIL_count