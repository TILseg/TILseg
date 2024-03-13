""" Unittests for test_refine_kmeans module"""

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
import unittest

# Internal imports
import tilseg
import tilseg.refine_kmeans

class TestRefineKMeans(unittest.TestCase):
    
    def test_mask_to_features(self):
        """
        Unittests for mask_to_features function
        """
        #Checks that an array of all 0's will return no features
        binary_mask_empty = np.zeros((10, 10))
        features_empty = mask_to_features(binary_mask_empty)
        self.assertEqual(len(features_empty), 0)

        #Checks that an array with 0's and 1's will recognize features
        binary_mask_with_features = np.array([[1, 0, 1],
                                          [0, 1, 0],
                                          [1, 1, 1]])
        features_not_empty = mask_to_features(binary_mask_with_features)
        expected_features = np.array([[0, 0],
                                      [0, 2],
                                      [1, 1],
                                      [2, 0],
                                      [2, 1],
                                      [2, 2]])
        self.assertTrue(np.array_equal(features_not_empty, expected_features))

    def test_kb_dbscan_wrapper(self):
        """
        Unittests for test_dbscan_wrapper function
        """
        # Creating example binary mask and defining parameters
        binary_mask = np.array([[0, 1, 0],
                                [1, 0, 1]
                                [0, 1, 0]])
        hyperparameter_dict = {'eps': 0.5, 'min_samples': 5}  # What hyperparameters are we choosing?
        save_filepath = 'path to the saved file'  # Not quite sure how to set this up for reproducability?
        
        #Calling km_dbscan_wrapper function
        all_labels, dbscan = km_dbscan_wrapper(binary_mask, hyperparameter_dict, save_filepath)

        # Testing the output shape
        self.assertEqual(all_labels.shape, mask.shape)

        #Testing the model type
        self.assertIsInstance(dbscan, sklearn.cluster.DBSCAN)
        self.assertIsNotNone(dbscan.components_)

        # Testing al_labels output
        self.assertTrue(np.all(all_labels >= -1))  # Making sure labels are greater than -1
        self.assertTrue(all_labels.dtype == int) #Checking label types are integers

class TestKMeanDBSCANImplementation(unittest.TestCase):
    
    def test_kmean_to_spatial_model_superpatch_wrapper(self):
        """
        Unittests for kmean_to_spatial_model_superpatch_wrapper function
        """
        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict = tilseg.seg.kmean_to_spatial_model_superpatch_wrapper(
            super_patch = SUPERPATCH_PATH, in_dir_path = TEST_IN_DIR_PATH, spatial_hyperparameters={'eps': 15, 'min_samples': 200}, n_clusters=[4], out_dir_path=None)
        
        #checks if each output type is correct
        self.assertIsInstance(IM_labels, np.ndarray)
        self.assertTrue(isinstance(dbscan_fit, sklearn.cluster.DBSCAN))
        self.assertIsInstance(cluster_mask_dict, dict)
        # checks that the model outputted above is fitted
        self.assertTrue(sklearn.utils.validation.check_is_fitted(dbscan_fit) is None)

    def test_kmean_dbscan_patch_wrapper(self):
        """
        Unittests for kmean_dbscan_patch_wrapper function
        """
        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict = tilseg.seg.kmean_dbscan_patch_wrapper(TEST_PATCH_PATH, spatial_hyperparameters={'eps': 15, 'min_samples': 200}, n_clusters[4], out_dir_path=None)

        #checks if each output type is correct
        self.assertIsInstance(IM_labels, np.ndarray)
        self.assertIsInstance(dbscan_fit, sklearn.cluster.DBSCAN)
        self.assertIsInstance(cluster_mask_dict, dict) 

        #Include error tests
