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
from PIL import UnidentifiedImageError

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
    
    def test_KMeans_superpatch_fit(self):
        """
        Unittests for KMeans_superpatch_fit function
        """

        # one-shot test with correct inputs
        model = refine_kmeans.KMeans_superpatch_fit(
            patch_path=TEST_PATCH_PATH,
            hyperparameter_dict={'n_clusters': 4})

        # checks that the model outputted above is of the correct type
        self.assertTrue(isinstance(model, sklearn.cluster._kmeans.KMeans))

        # checks that the model outputted above is fitted
        self.assertTrue(
            sklearn.utils.validation.check_is_fitted(model) is None)

        # tests that non-string input for patch_path is dealt with
        with self.assertRaises(TypeError):
            model = seg.KMeans_superpatch_fit(
                patch_path=2,
                hyperparameter_dict={'n_clusters': 4})

        # tests when input file does not exist
        with self.assertRaises(FileNotFoundError):
            model = seg.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH+'blahblah',
                hyperparameter_dict={'n_clusters': 4})

        # tests when input file is not an image
        with self.assertRaises(UnidentifiedImageError):
            model = seg.KMeans_superpatch_fit(
                patch_path=FAIL_TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4})

        # tests when hyperparameter_dict is not a dictionary
        with self.assertRaises(TypeError):
            model = seg.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict=4)

        # tests when hyperparameter_dict does not have the expected keys
        with self.assertRaises(KeyError):
            model = seg.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4, 'tol': 0.001})
        with self.assertRaises(KeyError):
            model = seg.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_flusters': 4})

        # tests when n_clusters is not an integer less than 9
        with self.assertRaises(ValueError):
            model = seg.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 'four'})
        with self.assertRaises(ValueError):
            model = seg.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 9})
    
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


    def test_kmean_to_spatial_model_superpatch_wrapper(self):
        """
        Unittests for kmean_to_spatial_model_superpatch_wrapper function
        """
        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict,cluster_index = refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH,
                                            save_TILs_overlay=True)
        
        #checks if each output type is correct
        self.assertIsInstance(IM_labels, dict)
        self.assertTrue(isinstance(dbscan_fit, dict))
        self.assertIsInstance(cluster_mask_dict, dict)
        # checks that the model outputted above is fitted
        self.assertTrue(sklearn.utils.validation.check_is_fitted(dbscan_fit[next(iter(dbscan_fit))]) is None)

        shutil.rmtree(os.path.join(TEST_OUT_DIR_PATH, 'test_small_patch'))
        shutil.rmtree(os.path.join(TEST_OUT_DIR_PATH, 'test_small_patch_2'))              


    def test_kmean_to_spatial_model_patch_wrapper(self):
        """
        Unittests for kmean_dbscan_patch_wrapper function
        """
        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict, cluster_index = refine_kmeans.kmean_to_spatial_model_patch_wrapper(TEST_PATCH_PATH,
                        TEST_SPATIAL_HYPERPARAMETERS,
                        out_dir_path = TEST_OUT_DIR_PATH,
                        save_TILs_overlay = True,
                        random_state = None)

        #checks if each output type is correct
        self.assertIsInstance(IM_labels, np.ndarray)
        self.assertIsInstance(dbscan_fit, sklearn.cluster.DBSCAN)
        self.assertIsInstance(cluster_mask_dict, dict) 

        #Include error tests
