"""
Unittests for cluster processing module
"""

# Standard Library Imports
import os
import shutil
import unittest

# External Library Imports
import cv2 as cv
import matplotlib.pyplot
import numpy as np
import pandas as pd
import skimage.data

# Local imports
import tilseg.cluster_processing


class TestClusterProcessing(unittest.TestCase):
    """
    Test case for functions within cluster_processing.py
    """


    def test_image_series_exceptions(self):
        """
        Test for the image series exceptions function
        """
        dim_array_1 = np.ones(10)
        dim_array_3 = -1 * np.ones((10, 10, 3))
        dim_array_4 = np.ones((10, 10, 10, 4))

        with self.assertRaises(ValueError):
            tilseg.cluster_processing.image_series_exceptions(dim_array_1, True)
        with self.assertRaises(ValueError):
            tilseg.cluster_processing.image_series_exceptions(dim_array_1, False)
        with self.assertRaises(ValueError):
            tilseg.cluster_processing.image_series_exceptions(dim_array_4, True)
        with self.assertRaises(ValueError):
            tilseg.cluster_processing.image_series_exceptions(dim_array_3, False)


    def test_generate_image_series(self):
        """
        Test for generate image series function
        """
        dim_array_3 = np.ones((10, 10, 3))
        dim_array_4 = np.ones((10, 10, 10, 3))

        color_dir_path = os.path.join(".", "color")
        color_image_path = os.path.join(color_dir_path, "Image1.jpg")
        gray_dir_path = os.path.join(".", "gray")
        gray_image_path = os.path.join(color_dir_path, "Image1.jpg")

        # Generate image series
        tilseg.cluster_processing.generate_image_series(dim_array_4, ".", "color", True)
        tilseg.cluster_processing.generate_image_series(dim_array_3, ".", "gray", False)

        # Check if file was succesfully created
        self.assertTrue(os.path.isdir(color_dir_path))
        self.assertTrue(os.path.isfile(color_image_path))
        self.assertTrue(os.path.isdir(gray_dir_path))
        self.assertTrue(os.path.isfile(gray_image_path))

        # Delete file
        shutil.rmtree(color_dir_path)
        shutil.rmtree(gray_dir_path)


    def test_gen_base_arrays(self):
        dim_array_3 = 3 * np.ones((10, 10, 3))
        dims = dim_array_3.shape
        num_clust = 4
        final_array, binary_array, all_mask_array=tilseg.cluster_processing.gen_base_arrays(dim_array_3, num_clust)

        self.assertIsInstance(final_array, np.ndarray)
        self.assertTrue(([3]) == np.unique(final_array))
        self.assertTrue(num_clust == final_array.shape[0])
        self.assertTrue(dims[0] == final_array.shape[1])
        self.assertTrue(dims[1] == final_array.shape[2])

        self.assertIsInstance(binary_array, np.ndarray)
        self.assertTrue(([0]) == np.unique(binary_array))
        self.assertTrue(num_clust == binary_array.shape[0])
        self.assertTrue(dims[0] == binary_array.shape[1])
        self.assertTrue(dims[1] == binary_array.shape[2])

        self.assertIsInstance(all_mask_array, np.ndarray)
        self.assertTrue(([0]) == np.unique(all_mask_array))
        self.assertTrue(dims[0] == all_mask_array.shape[0])
        self.assertTrue(dims[1] == all_mask_array.shape[1])


    def test_result_image_generator(self):
        original_image = 120 * np.ones((10, 10, 3))
        img_clust = np.random.randint(1, 4, size=(10, 10))

        final_arrays, binary_arrays, all_masks = tilseg.cluster_processing.result_image_generator(img_clust, original_image)

        with self.assertRaises(ValueError):
            _, _, _ = tilseg.cluster_processing.result_image_generator(original_image, original_image)
        with self.assertRaises(ValueError):
            _, _, _ = tilseg.cluster_processing.result_image_generator(img_clust, img_clust)

        self.assertTrue(4 == final_arrays.shape[0])
        self.assertTrue(4 == binary_arrays.shape[0])

        self.assertIsInstance(final_arrays, np.ndarray)
        self.assertIsInstance(binary_arrays, np.ndarray)
        self.assertIsInstance(all_masks, np.ndarray)

        self.assertTrue(all(([0, 120]) == np.unique(final_arrays)))
        self.assertTrue(all(([0, 1]) == np.unique(binary_arrays)))
        self.assertTrue(all(([0, 255]) == np.unique(all_masks)))


    def test_mask_only_generator(self):
        original_image = 120 * np.ones((10, 10, 3))
        img_clust = np.random.randint(1, 4, size=(10, 10))

        binary_arrays = tilseg.cluster_processing.mask_only_generator(img_clust)

        with self.assertRaises(ValueError):
            _, _, _ = tilseg.cluster_processing.mask_only_generator(original_image)

        self.assertTrue(4 == binary_arrays.shape[0])

        self.assertTrue(all(([0, 1]) == np.unique(binary_arrays)))

    
    def test_filter_boolean(self):
        pass_contour = np.array(([[0, 0]], [[40, 0]], [[40, 40]], [[0, 40]]))
        area_fail_contour = np.array(([[[0, 0]],[[0, 1]],[[1, 1]],[[1, 0]]]))
        roundness_fail_contour = np.array(([[0, 0]], [[100, 0]], [[100, 3]], [[0, 3]]))

        meets_crit = tilseg.cluster_processing.filter_boolean(pass_contour)
        self.assertIsInstance(meets_crit, bool)
        
        self.assertTrue(tilseg.cluster_processing.filter_boolean(pass_contour))
        self.assertFalse(tilseg.cluster_processing.filter_boolean(area_fail_contour))
        self.assertFalse(tilseg.cluster_processing.filter_boolean(roundness_fail_contour))
    
    def test_contour_generator(self):
        single_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.5, 1)
        contours_mod, contours_count = tilseg.cluster_processing.contour_generator(single_blob)
        self.assertIsInstance(contours_mod, list)
        self.assertIsInstance(contours_count, int)
        self.assertEquals(contours_count, 2)
        with self.assertRaises(ValueError):
            contours_mod, contours_count = tilseg.cluster_processing.contour_generator(2*single_blob)


