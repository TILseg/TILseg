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
import cv2
import PIL
import os
from skimage import io
import pathlib
from PIL import UnidentifiedImageError
from sklearn.exceptions import NotFittedError


def gen_base_arrays(ori_image: np.ndarray, num_clusts: int):
    """
    Generates set of two arrays which will be overlaid with the cluster data.
    The first array contains the number of clusters+1 images which will recieve
    masks based on the associated cluster. The second is a binary array for use
    in contour generation.
    """
    dims = ori_image.shape
    four_dim_array = np.expand_dims(ori_image, 0)
    binary_array = np.zeros((num_clusts, dims[0], dims[1]))
    all_mask_array = np.zeros((3, dims[0], dims[1]))
    final_array = four_dim_array
    for _ in range(num_clusts - 1):
        final_array = np.vstack((final_array, four_dim_array))
    return final_array, binary_array, all_mask_array


def generate_image_series(image_array: np.ndarray, filepath: str, prefix: str):
    """
    This takes in an array of image values and generates a directory of
    jpg images in a specified file location.
    """
    dims = image_array.shape
    path = os.path.join(filepath, prefix)
    if not os.path.exists(filepath):
        os.mkdir(path)
    else:
        pass
    os.chdir(path)
    for count in range(dims[2]+1):
        cv2.imwrite(f"Image{count + 1}.jpg", image_array[count][:][:][:])


def image_overlay_generator(img_clust: np.ndarray, original_image: np.ndarray):
    """
    Generates series of images equal to the number of clusters plus the
    original and saves it to the specified filepath
    """
    # Colors that will become associated with each cluster on overlays
    black = np.array([0, 0, 0])

    colors_overlay = np.array(([0, 0, 0], [255, 0, 0],
                              [0, 255, 0], [0, 0, 255]))

    # Making a dictionary of the original images that will be overwriten
    dims = img_clust.shape
    num_clust = int(img_clust.max() + 1)

    final_arrays, binary_arrays, all_masks = gen_base_arrays(original_image,
                                                             num_clust)

    for j in range(dims[0]):
        for k in range(dims[1]):
            key = int(img_clust[j][k])
            final_arrays[key][j][k] = black
            binary_arrays[key][j][k] = 1
            for count in range(3):
                all_masks[count][j][k] = colors_overlay[key][count]

    return final_arrays, binary_arrays, all_masks


def image_postprocessing(clusters: np.ndarray, ori_img: np.ndarray,
                         filepath: str, gen_overlays: bool = True,
                         gen_masks: bool = False, gen_csv: bool = True):
    """
    This is a wrapper function that will be used to group all postprocessing
    together.

    Inputs:
    ori_img - 3D array with dimensions X, Y, and color with three color
    channels as RGB
    clusters - 2D array with dimensions X, Y and values as the cluster
    identified via the model
    gen_overlays - boolean to determine if overlay images will be generated
    gen_csv - boolean to determine if CSV of contours will be generated
    """

    masked_images, masks, all_masks = image_overlay_generator(clusters,
                                                              ori_img)

    cv2.imwrite("Original.jpg", ori_img)
    cv2.imwrite("AllClusters.jpg", all_masks)

    if gen_overlays:
        generate_image_series(masked_images, filepath, "Overlaid Images")
    else:
        pass

    if gen_masks:
        generate_image_series(masks, filepath, "Masks")
    else:
        pass

    # if gen_csv:
    #     immune_cluster_generator(masks, filepath)
    # else:
    #     pass
    return masked_images, masks


def cluster_model_fitter(patch_path: str, 
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
                     patch_path: str):

    """
    Scores the clustering using various metrics

    Parameters
    -----
    model: sklearn.cluster.model
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

    # scores the clustering based on various metrics
    # Silhoutte score currently takes too long
    # s_score = sklearn.metrics.silhouette_score(pred_patch.reshape((-1,3)), labels)
    ch_score = sklearn.metrics.calinski_harabasz_score(pred_patch.reshape((-1,3)), labels)
    db_score = sklearn.metrics.davies_bouldin_score(pred_patch.reshape((-1,3)), labels)

    return ch_score, db_score


def pred_and_cluster(model: sklearn.base.ClusterMixin, 
                     in_dir_path: str, 
                     out_dir_path: str=None,
                     algorithm: str='KMeans',
                     save_cluster_masks: bool=False,
                     save_composite_cluster_mask: bool=False,
                     save_cluster_overlays: bool=False,
                     save_csv: bool=True):

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
        
        image_postprocessing(clusters=labels.reshape(pred_patch.shape[0], pred_patch.shape[1]), 
                             ori_img=pred_patch, 
                             filepath=out_dir_path, 
                             gen_overlays=save_cluster_overlays, 
                             gen_masks=save_cluster_masks, 
                             gen_csv=save_csv)

    return None

# def pred_and_cluster_old(model: sklearn.base.ClusterMixin, 
#                      in_dir_path: str, 
#                      out_dir_path: str=None,
#                      algorithm: str='KMeans',
#                      save_cluster_masks: bool=False,
#                      save_composite_cluster_mask: bool=False,
#                      save_all_cluster_overlay: bool=False):

#     """
#     Applies a fitted clustering model to patches and generates multiple images: binary segmentation masks of each cluster, 
#     segmentation masks overlaid on the original patch, and all clusters overlaid on the original patch

#     Parameters
#     -----
#     model: sklearn.cluster.model
#         the fitted model
#     algorithm: str
#         the clustering algorithm to be used: 'KMeans', '', ''
#     in_dir: str
#         the directory path to the patches that will be predicted and clustered
#     out_dir: str
#         the directory path where output images will be saved
#     """


#     if not os.path.isdir(in_dir_path):
#         raise ValueError('Please enter a valid input directory')
#     else:
#         pass

#     if out_dir_path != None:
#         if not os.path.isdir(out_dir_path):
#             raise ValueError('Please enter a valid output directory')
#         else:
#             pass
#     else:
#         pass

#     # Iterating over every patch in the directory
#     for file in os.listdir(in_dir_path):
        
#         # Creating a directory with the same file name (without extenstion)
#         # Passing if such a directory already exists
#         try:
#             os.mkdir(os.path.join(out_dir_path, file[:-4]))
#         except:
#             pass

#         try:
#             sklearn.utils.validation.check_is_fitted(model)
#         except TypeError:
#             print('model is not an estimator')
#             raise
#         except NotFittedError:
#             print('Please fit the first using tilseg.cluster.cluster_model_fitter')
#             raise
    
#         # Reads the current patch into a numpy uint8 array 
#         pred_patch = plt.imread(os.path.join(in_dir_path, file))
#         # Linearizes the array for R, G, and B separately and normalizes
#         # The result is an N X 3 array where N=height*width of the patch in pixels
#         pred_patch_n = np.float32(pred_patch.reshape((-1, 3))/255.)

#         try:
#             # Predicting the index/labels of the clusters on the fitted model from 'model' function
#             # The result is an N X 3 array where N=height*width of the patch in pixels
#             # Each value shows the label of the cluster that pixel belongs to
#             labels = model.predict(pred_patch_n)
#         except:
#             print('Please input a valid sklearn.cluster.model for model. This can be produced using tilseg.cluster.cluster_model_fitter')
#             raise

#         # Makes sure that the model is training for 8 clusters
#         if len(np.unique(labels)) <= 8:
#             pass
#         else:
#             print("Looks like the model is being trained for more than 8 clusters. Please consider training it on less number of clusters.")
#             raise
        
#         # creates a copy of the coordinates of the cluster centers in the RGB space
#         # The results is 8X3 numpy array
#         overlay_center = np.zeros((len(np.unique(labels)), 3))

#         # created a numpy uint8 array of the background image- this is just the H&E patch without any normalization
#         back_img = np.uint8(np.copy(pred_patch))

#         # Reassigning the cluster centers of the RGB space to custom colors for visual effects
#         # Essentially creating new RGB coordinates for each cluster center
#         cluster_center_RGB_list = [
#             np.array([0, 255, 255])/255, #Cyan
#             np.array([255, 102, 102])/255., #Light Red
#             np.array([153, 255, 51])/255, #Light Green
#             np.array([178, 102, 255])/255, #Light Purple
#             np.array([0, 128, 255])/255, #Light Blue
#             np.array([95, 95, 95])/255, #Grey
#             np.array([102, 0, 0])/255, #Maroon
#             np.array([255, 0, 127])/255 #Bright Pink
#                                    ] 
#         for i, cluster_center_RGB in enumerate(cluster_center_RGB_list[:len(model.cluster_centers_)]):
#             overlay_center[i] = cluster_center_RGB

#         # Iterating over each cluster centroid
#         for i in range(len(overlay_center)):

#             # Creating a copy of the linearized and normalized RGB array
#             seg_img = np.copy(pred_patch_n)

#             # The left-hand side is a mask that accesses all pixels that belong to cluster 'i'
#             # The ride hand side replaces the RGB values of each pixel with the RGB value of the corresponding custom-chosen RGB values for each cluster

#             seg_img[labels.flatten() == i] = overlay_center[i] 

#             # The left-hand side is a mask that accesses all pixels that DO NOT belong to cluster 'i'
#             # The ride hand side replaces the RGB values of each pixel with white color
#             # Therefor every pixel except for those in cluster 'i' will be white
#             seg_img[labels.flatten() != i] = np.array([255, 255, 255])/255.

#             # Reshapes the image with cluster 'i' identified to the original picture shape
#             seg_img = seg_img.reshape(pred_patch.shape)

#             # Saves the image as filename_segmented_#.jpg with 1,000 dots per inch printing resolution
#             # Thus there will be 8 images identifying each cluster from each patch
#             if save_cluster_masks:
#                 plt.imsave(os.path.join(out_dir_path, file[:-4], '_segmented_'+str(i)+'.jpg'), seg_img, dpi=1000)
#             else:
#                 pass

#             # Reversing the normalization of the RGB values of the image with the cluster
#             seg_img = np.uint8(seg_img*255.)

#             # cv2.addWeighted is a function that allows us to overlay one image on top of another and adjust their 
#             # alpha (transparency) so that the two can blended/overlayed and both still be clearly visible
#             # overlay_img is the image where the segmented image consisting of isolated cluster is overlayed over the 
#             # original H&E image
#             overlay_img = cv2.addWeighted(back_img, 0.4, seg_img, 0.6, 0)/255.

#             # Saves the overlayed image as filename_overlay_#.jpg with 1,000 dots per inch printing resolution
#             # Thus there will be 8 overlayed images identifying each cluster from each patch
#             if save_cluster_overlays:
#                 plt.imsave(os.path.join(out_dir_path, file[:-4], '_overlay_'+str(i)+'.jpg'), overlay_img, dpi=1000)
#             else:
#                 pass

#         # Make an image containing all the clusters in one
#         # Also reshapes the image with all clusters identified to the original picture shape
#         # Don't quite understand how this line would work without indexing error but I get what it is trying to do
#         all_cluster = overlay_center[labels.flatten()].reshape(pred_patch.shape)

#         # Saves the image as filename_all_cluster.jpg with 1,000 dots per inch printing resolution
#         # Thus there will be 1 image identifying all clusters on the same image from each patch
#         if save_composite_cluster_mask:
#             plt.imsave(os.path.join(out_dir_path, file[:-4], '_all_cluster.jpg'), all_cluster, dpi=1000)
#         else:
#             pass

#         # Overlaying the complete cluster:
#         # Reversing the normalization of the RGB values of the image with the all the clusters
#         seg_img = np.uint8(np.copy(all_cluster)*255.)

#         # overlay_img is the image where the segmented image consisting of all the isolated clusters 
#         # is overlayed over the original H&E image
#         overlay_img = cv2.addWeighted(back_img, 0.6, seg_img, 0.4, 0)

#         # Saves the overlayed image as filename_full_overlay.jpg with 1,000 dots per inch printing resolution
#         # Thus there will be 1 fully overlayed image identifying all the clusters from each patch
#         if save_all_cluster_overlay:
#             plt.imsave(os.path.join(out_dir_path, file[:-4], '_full_overlay.jpg'), overlay_img, dpi=1000)
#         else:
#             pass

#     return None