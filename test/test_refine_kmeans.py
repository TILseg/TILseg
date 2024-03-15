""" 
Unittests for test_refine_kmeans module
"""

# Core library imports
import os
import shutil
import unittest

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
                                    'test_patches', 'image.tif')
TEST_IN_DIR_PATH = os.path.join(parent_dir, 'test',
                                'test_patches', 'patches')
FAIL_IN_PATH = os.path.join(parent_dir, 'test',
                                'test_patches', 'patchez')
TEST_OUT_DIR_PATH = os.path.join(parent_dir, 'test',
                                'test_patches', 'test_results')
FAIL_OUT_PATH = os.path.join(parent_dir, 'test',
                                    'test_patches', 'result')
SUPERPATCH_PATH = os.path.join(parent_dir, 'test',
                                'test_patches', 'test_superpatch.tif')
TEST_SPATIAL_HYPERPARAMETERS = {
    'eps': 10,
    'min_samples': 100,
}

# Skip mark
# @pytest.mark.skip

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
            model = refine_kmeans.KMeans_superpatch_fit(
                patch_path=2,
                hyperparameter_dict={'n_clusters': 4})

        # tests when input file does not exist
        with self.assertRaises(FileNotFoundError):
            model = refine_kmeans.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH+'blahblah',
                hyperparameter_dict={'n_clusters': 4})

        # tests when hyperparameter_dict is not a dictionary
        with self.assertRaises(TypeError):
            model = refine_kmeans.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict=4)

        # tests when hyperparameter_dict does not have the expected keys
        with self.assertRaises(KeyError):
            model = refine_kmeans.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4, 'tol': 0.001})
        with self.assertRaises(KeyError):
            model = refine_kmeans.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_flusters': 4})

        # tests when n_clusters is not an integer less than 9
        with self.assertRaises(ValueError):
            model = refine_kmeans.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 'four'})
        with self.assertRaises(ValueError):
            model = refine_kmeans.KMeans_superpatch_fit(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 9})
    
    
    def test_mask_to_features(self):
        """
        Unittests for mask_to_features function
        """
        # Test with an invalid input that is not a numpy array
        with self.assertRaises(ValueError):
            refine_kmeans.mask_to_features([1, 0, 1])

        # Test with an invalid input that is not a 2D numpy array
        with self.assertRaises(ValueError):
            refine_kmeans.mask_to_features(np.array([1, 0, 1]))

        # Checks that an array with 0's and 1's will recognize features
        binary_mask_test = np.array([[1, 0, 1],
                                    [0, 1, 0],
                                    [1, 1, 1]])
        features_not_empty = refine_kmeans.mask_to_features(binary_mask_test)
        # expected features based on the x,y coordinates of 1s in binary mask
        expected_features = np.array([[0, 0],
                                      [0, 2],
                                      [1, 1],
                                      [2, 0],
                                      [2, 1],
                                      [2, 2]])
        self.assertTrue(np.array_equal(features_not_empty, expected_features))

        # Checks that the output matrix has the expected shape 
        # based on the number of 1s in the binary mask
        expected_num_features = np.count_nonzero(binary_mask_test)
        self.assertEqual(features_not_empty.shape[0], expected_num_features)

        # Raises error for a binary mask that contains invalid values
        binary_mask_invalid = np.array([[0, 2, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]])
        with self.assertRaises(ValueError):
            refine_kmeans.mask_to_features(binary_mask_invalid)

        # Raises error for a binary mask that contains only 0s
        # Test with a mask containing only zeros
        binary_mask_zeros = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            refine_kmeans.mask_to_features(binary_mask_zeros)

        # Checks that the output is a 2D numpy array
        self.assertIsInstance(features_not_empty, np.ndarray)
        self.assertTrue(features_not_empty.ndim == 2)

    
    def test_kb_dbscan_wrapper(self):
        """
        Unittests for test_dbscan_wrapper function
        """
        # Creating example binary mask and defining parameters
        binary_mask = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]])
        
        # Calling km_dbscan_wrapper function
        all_labels, dbscan = refine_kmeans.km_dbscan_wrapper(binary_mask, 
                                                             TEST_SPATIAL_HYPERPARAMETERS, 
                                                             TEST_OUT_DIR_PATH)

        # Verifying the output shape
        self.assertEqual(all_labels.shape, binary_mask.shape)

        # Verifying the model type
        self.assertIsInstance(dbscan, sklearn.cluster.DBSCAN)
        self.assertIsNotNone(dbscan.components_)

        # Testing all_labels output
        self.assertTrue(np.all(all_labels >= -1))  # Making sure labels are greater than -1
        self.assertTrue(all_labels.dtype == int) #Checking label types are integers

        # Check if directory and files were created
        self.assertTrue(os.path.exists(os.path.join(TEST_OUT_DIR_PATH, 'ClusteringResults', 'dbscan_result_colorbar.jpg')))
        self.assertTrue(os.path.exists(os.path.join(TEST_OUT_DIR_PATH, 'ClusteringResults', 'dbscan_result.jpg')))

        # Raises error when the input mask is not a 2D numpy array
        mask_1D = np.array([0, 1, 0])
        with self.assertRaises(ValueError):
            refine_kmeans.km_dbscan_wrapper(mask_1D, 
                                           TEST_SPATIAL_HYPERPARAMETERS, 
                                           TEST_OUT_DIR_PATH)
        mask_notnp = [[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]]
        with self.assertRaises(ValueError):
            refine_kmeans.km_dbscan_wrapper(mask_notnp, 
                                           TEST_SPATIAL_HYPERPARAMETERS, 
                                           TEST_OUT_DIR_PATH)
            
        # Raises error when the save directory doesn't exist
        with self.assertRaises(FileNotFoundError):
            refine_kmeans.km_dbscan_wrapper(binary_mask, 
                                           TEST_SPATIAL_HYPERPARAMETERS, 
                                           FAIL_OUT_PATH)

        # Raises error when the save directory is unwritable
        TEST_OUT_DIR_PATH2 = TEST_OUT_DIR_PATH
        os.chmod(TEST_OUT_DIR_PATH2, 0o444)  # This sets read-only permissions
        with self.assertRaises(PermissionError):
            refine_kmeans.km_dbscan_wrapper(binary_mask, 
                                           TEST_SPATIAL_HYPERPARAMETERS, 
                                           TEST_OUT_DIR_PATH2)

        # clean-up
        os.chmod(TEST_OUT_DIR_PATH, 0o777)
        os.remove(os.path.join(TEST_OUT_DIR_PATH,'ClusteringResults', 'dbscan_result_colorbar.jpg'))
        os.remove(os.path.join(TEST_OUT_DIR_PATH,'ClusteringResults', 'dbscan_result.jpg'))

    
    def test_kmean_to_spatial_model_superpatch_wrapper(self):
        """
        Unittests for kmean_to_spatial_model_superpatch_wrapper function
        """
        # Restore writing permissions
        os.chmod(TEST_OUT_DIR_PATH, 0o777)

        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict, cluster_index = refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH,
                                            save_TILs_overlay=True)
        
        # checks if each output type is correct
        self.assertIsInstance(IM_labels, dict)
        self.assertTrue(isinstance(dbscan_fit, dict))
        self.assertIsInstance(cluster_mask_dict, dict)
        self.assertIsInstance(cluster_index, dict)
        
        # checks that the model outputted above is fitted
        self.assertTrue(sklearn.utils.validation.check_is_fitted(dbscan_fit[next(iter(dbscan_fit))]) is None)

        # Checks that IM_labels has the expected keys
        tif_files = [file for file in os.listdir(TEST_IN_DIR_PATH) if file.endswith('.tif')]
        expected_keys = [os.path.splitext(file)[0] for file in tif_files]
        missing_keys = [key for key in expected_keys if key not in IM_labels]
        self.assertEqual(missing_keys, [], msg=f"Keys {missing_keys} are missing from IM_labels")

        # Checks that dbscan_fit has the expected keys
        missing_keys = [key for key in expected_keys if key not in dbscan_fit]
        self.assertEqual(missing_keys, [], msg=f"Keys {missing_keys} are missing from dbscan_fit")
        
         # Checks that cluster_mask_dict has the expected keys
        missing_keys = [key for key in expected_keys if key not in cluster_mask_dict]
        self.assertEqual(missing_keys, [], msg=f"Keys {missing_keys} are missing from cluster_mask_dict")

         # Checks that cluster_index has the expected keys
        missing_keys = [key for key in expected_keys if key not in cluster_index]
        self.assertEqual(missing_keys, [], msg=f"Keys {missing_keys} are missing from cluster_index")

        # Raises error when the superpatch path doesn't exist
        with self.assertRaises(FileNotFoundError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = FAIL_TEST_PATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH,
                                            save_TILs_overlay=True)


        # Raises error when the in directory doesn't exist
        with self.assertRaises(FileNotFoundError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = FAIL_IN_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH,
                                            save_TILs_overlay=True)

        # Raises error when the out directory doesn't exist
        with self.assertRaises(FileNotFoundError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = FAIL_OUT_PATH,
                                            save_TILs_overlay=True)

        # Raises error when the out directory is unwritable
        TEST_OUT_DIR_PATH2 = TEST_OUT_DIR_PATH
        os.chmod(TEST_OUT_DIR_PATH2, 0o444)  # This sets read-only permissions
        with self.assertRaises(PermissionError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH2,
                                            save_TILs_overlay=True)

        # clean-up
        os.chmod(TEST_OUT_DIR_PATH, 0o777)
        shutil.rmtree(os.path.join(TEST_OUT_DIR_PATH, 'test_small_patch'))
        shutil.rmtree(os.path.join(TEST_OUT_DIR_PATH, 'test_small_patch_2'))              

    
    def test_kmean_to_spatial_model_patch_wrapper(self):
        """
        Unittests for kmean_dbscan_patch_wrapper function
        """
        # Restore writing permissions
        os.chmod(TEST_OUT_DIR_PATH, 0o777)

        # one-shot test with correct inputs
        IM_labels, dbscan_fit, cluster_mask_dict, cluster_index = refine_kmeans.kmean_to_spatial_model_patch_wrapper(TEST_PATCH_PATH,
                        TEST_SPATIAL_HYPERPARAMETERS,
                        out_dir_path = TEST_OUT_DIR_PATH,
                        save_TILs_overlay = True,
                        random_state = None)

        # checks if each output type is correct
        self.assertIsInstance(IM_labels, np.ndarray)
        self.assertIsInstance(dbscan_fit, sklearn.cluster.DBSCAN)
        self.assertIsInstance(cluster_mask_dict, dict) 
        self.assertIsInstance(cluster_index, int)

        # Checks that cluster_mask_dict has the expected keys
        expected_key = os.path.splitext(os.path.basename(TEST_PATCH_PATH))[0]
        self.assertIn(expected_key, cluster_mask_dict, msg=f"Key is missing from cluster_mask_dict")

        # Raises error when the superpatch path doesn't exist
        with self.assertRaises(FileNotFoundError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = FAIL_TEST_PATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH,
                                            save_TILs_overlay=True)


        # Raises error when the in directory doesn't exist
        with self.assertRaises(FileNotFoundError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = FAIL_IN_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH,
                                            save_TILs_overlay=True)

        # Raises error when the out directory doesn't exist
        with self.assertRaises(FileNotFoundError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = FAIL_OUT_PATH,
                                            save_TILs_overlay=True)

        # Raises error when the out directory is unwritable
        TEST_OUT_DIR_PATH2 = TEST_OUT_DIR_PATH
        os.chmod(TEST_OUT_DIR_PATH2, 0o444)  # This sets read-only permissions
        with self.assertRaises(PermissionError):
            refine_kmeans.kmean_to_spatial_model_superpatch_wrapper(superpatch_path = SUPERPATCH_PATH,
                                            in_dir_path = TEST_IN_DIR_PATH,
                                            spatial_hyperparameters = TEST_SPATIAL_HYPERPARAMETERS,
                                            out_dir_path = TEST_OUT_DIR_PATH2,
                                            save_TILs_overlay=True)
        # note: no further testing needed for IM_labels since it is an output of
        # km_dbscan_wrapper function
            
        # clean-up
        os.chmod(TEST_OUT_DIR_PATH, 0o777)
        parent_dir = os.path.join(TEST_OUT_DIR_PATH, expected_key)
        os.remove(os.path.join(parent_dir,'ClusteringResults', 'ContourMask.jpg'))
        os.remove(os.path.join(parent_dir,'ClusteringResults', 'ContourOverlay.jpg'))
        os.remove(os.path.join(parent_dir,'ClusteringResults', 'dbscan_result_colorbar.jpg'))
        os.remove(os.path.join(parent_dir,'ClusteringResults', 'dbscan_result.jpg'))
        os.remove(os.path.join(parent_dir,'ClusteringResults', 'Original.jpg'))
    
