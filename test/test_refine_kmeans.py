""" Unittests for test_refine_kmeans module"""

# Core library imports
import os
import shutil
import unittest
# import pathlib

# External library imports
import numpy as np
import pytest
import sklearn
import sys
import unittest

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tilseg import seg, refine_kmeans


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Move up one directory
parent_dir = os.path.dirname(current_dir)

# Creating dummy inputs
TEST_PATCH_PATH = os.path.join(parent_dir, 'test',
                               'test_patches', 'patches',
                               'test_small_patch.tif')
FAIL_TEST_PATCH_PATH = os.path.join(parent_dir, 'test',
                                    'test_patches', 'test_img.txt')
TEST_IN_DIR_PATH = os.path.join(parent_dir, 'test',
                                'test_patches', 'patches')
TEST_OUT_DIR_PATH = os.path.join(parent_dir, 'test',
                                'test_patches', 'test_results')
SUPERPATCH_PATH = os.path.join(parent_dir, 'test',
                                'test_patches', 'test_superpatch.tif')
TEST_SPATIAL_HYPERPARAMETERS = {
    'eps': 10,
    'min_samples': 100,
}


class TestRefineKMeans(unittest.TestCase):
    
    
    def test_mask_to_features(self):
        """
        Unittests for mask_to_features function
        """
        #Checks that an array of all 0's will return no features
        binary_mask_empty = np.zeros((10, 10))
        features_empty = refine_kmeans.mask_to_features(binary_mask_empty)
        self.assertEqual(len(features_empty), 0)

        #Checks that an array with 0's and 1's will recognize features
        binary_mask_with_features = np.array([[1, 0, 1],
                                          [0, 1, 0],
                                          [1, 1, 1]])
        features_not_empty = refine_kmeans.mask_to_features(binary_mask_with_features)
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
                                [1, 0, 1],
                                [0, 1, 0]])
        hyperparameter_dict = TEST_SPATIAL_HYPERPARAMETERS
        save_filepath = TEST_OUT_DIR_PATH
        
        # Calling km_dbscan_wrapper function
        all_labels, dbscan = refine_kmeans.km_dbscan_wrapper(binary_mask, hyperparameter_dict, 
                                                             save_filepath)

        # Testing the output shape
        self.assertEqual(all_labels.shape, binary_mask.shape)

        #Testing the model type
        self.assertIsInstance(dbscan, sklearn.cluster.DBSCAN)
        self.assertIsNotNone(dbscan.components_)

        # Testing al_labels output
        self.assertTrue(np.all(all_labels >= -1))  # Making sure labels are greater than -1
        self.assertTrue(all_labels.dtype == int) #Checking label types are integers

        directory_path = os.path.join(TEST_OUT_DIR_PATH, 'ClusteringResults')
        os.remove(os.path.join(directory_path, 'dbscan_result_colorbar.jpg'))
        os.remove(os.path.join(directory_path, 'dbscan_result.jpg'))

    @pytest.mark.skip
    def test_kmean_to_spatial_model_superpatch_wrapper(self):
        """
        Unittests for kmean_to_spatial_model_superpatch_wrapper function
        """
        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict = refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH)
        
        #checks if each output type is correct
        self.assertIsInstance(IM_labels, np.ndarray)
        self.assertTrue(isinstance(dbscan_fit, sklearn.cluster.DBSCAN))
        self.assertIsInstance(cluster_mask_dict, dict)
        # checks that the model outputted above is fitted
        self.assertTrue(sklearn.utils.validation.check_is_fitted(dbscan_fit) is None)

        shutil.rmtree(os.path.join(TEST_OUT_DIR_PATH, 'test_small_patch'))
        shutil.rmtree(os.path.join(TEST_OUT_DIR_PATH, 'test_small_patch_2'))              

    @pytest.mark.skip
    def test_kmean_to_spatial_model_patch_wrapper(self):
        """
        Unittests for kmean_dbscan_patch_wrapper function
        """
        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict = refine_kmeans.kmean_to_spatial_model_patch_wrapper(TEST_PATCH_PATH,
                        TEST_SPATIAL_HYPERPARAMETERS,
                        out_dir_path = TEST_OUT_DIR_PATH,
                        save_TILs_overlay = False,
                        save_cluster_masks = False,
                        save_cluster_overlays = False,
                        save_all_clusters_img = False,
                        save_csv = False,
                        random_state = None)

        #checks if each output type is correct
        self.assertIsInstance(IM_labels, np.ndarray)
        self.assertIsInstance(dbscan_fit, sklearn.cluster.DBSCAN)
        self.assertIsInstance(cluster_mask_dict, dict) 

        #Include error tests
