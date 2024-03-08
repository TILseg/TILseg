""" Unittests for test_refine_kmeans module"""

#core library imports
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

#Internal imports (others?)
import tilseg
import tilseg.refine_kmeans


#KMeans_superpatch_fit is tested in preprocessing?

class TestRefineKMeans(unittest.TestCase)
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
        #others?

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

        #Possibly include a unit test for filepath?
