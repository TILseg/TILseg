"""
Unittests for cluster processing module
"""

# Standard Library Imports
import os
import shutil
import unittest

# External Library Imports
import numpy as np
import pandas as pd
import skimage.data
import sys

# Local imports: add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tilseg import cluster_processing

class TestClusterProcessing(unittest.TestCase):
    """
    Test case for functions within cluster_processing.py
    """

    def test_image_series_exceptions(self):
        """
        Test for the image series exceptions function
        """
        # generate multidimensional test arrays
        dim_array_1 = np.ones(10)
        dim_array_3 = -1 * np.ones((10, 10, 3))
        dim_array_4 = np.ones((10, 10, 10, 4))

        # asserts error when image is wrong dimension for RGB and grayscale
        with self.assertRaises(ValueError):
            cluster_processing.image_series_exceptions(dim_array_1,
                                                              True)
        with self.assertRaises(ValueError):
            cluster_processing.image_series_exceptions(dim_array_1,
                                                              False)
        # asserts error when image contains more than RGB channels
        with self.assertRaises(ValueError):
            cluster_processing.image_series_exceptions(dim_array_4,
                                                              True)
        # asserts error when grayscale values are out of range
        with self.assertRaises(ValueError):
            cluster_processing.image_series_exceptions(dim_array_3,
                                                              False)

    def test_generate_image_series(self):
        """
        Test for generate image series function
        """
        # generate multidimensional test arrays
        dim_array_3 = np.ones((10, 10, 3))
        dim_array_4 = np.ones((10, 10, 10, 3))

        # generate filepaths to check later for correct files
        home = os.getcwd()
        color_dir_path = os.path.join(home, "color")
        color_image_path = os.path.join(color_dir_path, "Image1.jpg")
        gray_dir_path = os.path.join(home, "gray")
        gray_image_path = os.path.join(color_dir_path, "Image1.jpg")

        # Generate image series
        cluster_processing.generate_image_series(dim_array_4, home,
                                                        "color", True)
        cluster_processing.generate_image_series(dim_array_3, home,
                                                        "gray", False)

        # Check if file and directory was succesfully created
        self.assertTrue(os.path.isdir(color_dir_path))
        self.assertTrue(os.path.isfile(color_image_path))
        self.assertTrue(os.path.isdir(gray_dir_path))
        self.assertTrue(os.path.isfile(gray_image_path))

        # Delete files and directories
        shutil.rmtree(color_dir_path)
        shutil.rmtree(gray_dir_path)

    def test_gen_base_arrays(self):
        """
        Test for generate base arrays function
        """
        # make test image array and input number of clusters
        dim_array_3 = 3 * np.ones((10, 10, 3))
        dims = dim_array_3.shape
        num_clust = 4

        # run gen_base_arrays and store outputs
        (final_array,
         binary_array,
         all_mask_array) = cluster_processing.gen_base_arrays(
            dim_array_3, num_clust)

        # ensure it outputs an array where every value is 3
        self.assertIsInstance(final_array, np.ndarray)
        self.assertTrue(([3]) == np.unique(final_array))
        # ensure the dimensions of the array match those expected
        self.assertTrue(num_clust == final_array.shape[0])
        self.assertTrue(dims[0] == final_array.shape[1])
        self.assertTrue(dims[1] == final_array.shape[2])

        # ensure it outputs an empty array for binary array
        self.assertIsInstance(binary_array, np.ndarray)
        self.assertTrue(([0]) == np.unique(binary_array))
        # ensure the dimensions of the array match those expected
        self.assertTrue(num_clust == binary_array.shape[0])
        self.assertTrue(dims[0] == binary_array.shape[1])
        self.assertTrue(dims[1] == binary_array.shape[2])

        # ensure it outputs an empty array for binary array
        self.assertIsInstance(all_mask_array, np.ndarray)
        self.assertTrue(([0]) == np.unique(all_mask_array))
        # ensure the dimensions of the array match those expected
        self.assertTrue(dims[0] == all_mask_array.shape[0])
        self.assertTrue(dims[1] == all_mask_array.shape[1])

    def test_result_image_generator(self):
        """
        Test for result image generator function
        """
        # generate test image and set of clusters
        original_image = 120 * np.ones((10, 10, 3))
        img_clust = np.random.randint(1, 4, size=(10, 10))

        # generate results and store the arrays
        (final_arrays,
         binary_arrays,
         all_masks) = cluster_processing.result_image_generator(
            img_clust, original_image)

        # ensure it raises an error if the image or cluster arrays are the
        # wrong dimensions
        with self.assertRaises(ValueError):
            _, _, _ = cluster_processing.result_image_generator(
                original_image, original_image)
        with self.assertRaises(ValueError):
            _, _, _ = cluster_processing.result_image_generator(
                img_clust, img_clust)

        # checks that all of the outputs are arrays
        self.assertIsInstance(final_arrays, np.ndarray)
        self.assertIsInstance(binary_arrays, np.ndarray)
        self.assertIsInstance(all_masks, np.ndarray)

        # check that the arrays have the right number of clusters in them
        self.assertTrue(4 == final_arrays.shape[0])
        self.assertTrue(4 == binary_arrays.shape[0])

        # checks that the values are as expected based on the original image
        # a binary array and a multicolored array
        self.assertTrue(all(([0, 120]) == np.unique(final_arrays)))
        self.assertTrue(all(([0, 1]) == np.unique(binary_arrays)))
        self.assertTrue(all(([0, 255]) == np.unique(all_masks)))

    def test_mask_only_generator(self):
        """
        Test for mask only generator function
        """
        # generate image and clusters for testing
        original_image = 120 * np.ones((10, 10, 3))
        img_clust = np.random.randint(1, 4, size=(10, 10))

        # run function and generate results
        binary_arrays = cluster_processing.mask_only_generator(
            img_clust)

        # check that a value error is raised when wrong dimensional array
        # is passed in
        with self.assertRaises(ValueError):
            _, _, _ = cluster_processing.mask_only_generator(
                original_image)

        # ensure mask shape and values are as expected
        self.assertTrue(4 == binary_arrays.shape[0])
        self.assertTrue(all(([0, 1]) == np.unique(binary_arrays)))

    def test_filter_boolean(self):
        """
        Test for filter boolean function
        """
        # generate a set of mock contours for testing
        pass_contour = np.array(([[0, 0]], [[40, 0]], [[40, 40]], [[0, 40]]))
        area_fail_contour = np.array(([[[0, 0]], [[0, 1]],
                                      [[1, 1]], [[1, 0]]]))
        roundness_fail_contour = np.array(([[0, 0]], [[100, 0]],
                                           [[100, 3]], [[0, 3]]))

        # generate results and check to ensure the output is a boolean
        meets_crit = cluster_processing.filter_boolean(pass_contour)
        self.assertIsInstance(meets_crit, bool)
        # check for each contour to ensure it passes or fails the filters as
        # expected based on area and roundness
        self.assertTrue(cluster_processing.filter_boolean(pass_contour))
        self.assertFalse(
            cluster_processing.filter_boolean(area_fail_contour))
        self.assertFalse(cluster_processing.filter_boolean(
            roundness_fail_contour))

    def test_contour_generator(self):
        """
        Test for contour generator function
        """
        # generate masks with one or two regions that should be identified
        # as contours
        single_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.5, 1)
        double_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.25, 1)
        # generate list of contours from the masks and store
        contours_mod, contours_count = \
            cluster_processing.contour_generator(single_blob)
        _, contours_count2 = \
            cluster_processing.contour_generator(double_blob)

        # check output types
        self.assertIsInstance(contours_mod, list)
        self.assertIsInstance(contours_count, int)
        # check that number of contours detected is as expected
        # note that it counts two contours for each region as it
        # captures the inside and outside of the region
        self.assertEqual(contours_count, 2)
        self.assertEqual(contours_count2, 4)

        # ensure an error is thrown if the input is not binary
        with self.assertRaises(ValueError):
            contours_mod, contours_count = \
                cluster_processing.contour_generator(2*single_blob)

    def test_csv_results_compiler(self):
        """
        Test for csv results compiler function
        """
        # generate mask with contours
        double_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.25, 1)
        # generate contours and save them to a csv
        contours_mod, _ = \
            cluster_processing.contour_generator(double_blob)
        cluster_processing.csv_results_compiler(contours_mod, ".")
        # check that a csv was generated in the correct location
        path = os.path.join(".", "Compiled_Data.csv")
        self.assertTrue(os.path.isfile(path))
        # read in csv and check that it has the expected shape
        # shape represents correct number of contours and data fields
        csv = pd.read_csv(path)
        self.assertTrue(csv.shape == (4, 4))
        # delete the csv
        os.remove(path)

    def test_immune_cluster_analyzer(self):
        """
        Test for immune cluster analyzer function
        """
        # generate masks with contours
        double_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.25, 1)
        single_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.5, 1)

        # generate list of filtered contours
        til_contour, contour_count, cluster_mask, count_index = \
            cluster_processing.immune_cluster_analyzer([double_blob,
                                                               single_blob])
        # check output types are as expected
        self.assertIsInstance(til_contour, list)
        self.assertIsInstance(contour_count, int)
        self.assertIsInstance(cluster_mask, np.ndarray)
        self.assertIsInstance(count_index, int)
        # ensure the correct number of contours were counted
        self.assertTrue(contour_count == 4)

    def test_draw_til_images(self):
        """
        Test for draw til images function
        """
        # create an array to act as the original image
        dim_array_3 = np.ones((100, 100, 3), np.uint8)
        # create a mask and generate contours from the image
        double_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.25, 1)
        contours_mod, _ = \
            cluster_processing.contour_generator(double_blob)

        # generate expected filepaths for final output files
        overlay = os.path.join(".", "ContourOverlay.jpg")
        mask = os.path.join(".", "ContourMask.jpg")
        # save files based on contour list and original image
        cluster_processing.draw_til_images(dim_array_3,
                                                  contours_mod, ".")
        # check that the files were successfully saved
        self.assertTrue(os.path.isfile(overlay))
        self.assertTrue(os.path.isfile(mask))
        # remove the generated files
        os.remove(overlay)
        os.remove(mask)

    def file_exists_function(self):
        """
        Checks and returns if expected files exist for the image
        postprocessing function

        Returns
        -----
        exists_list: list
            booleans that represent if one of the expected files or
            directories was generated
        """
        # initialize a list to hold boolean outcomes
        exists_list = [0] * 7
        # check that the parent directory is created
        dir_path = os.path.join(".", "ClusteringResults")
        if os.path.isdir(dir_path):
            exists_list[0] = True
        # check and store if original image is saved
        original_path = os.path.join(dir_path, "Original.jpg")
        if os.path.isfile(original_path):
            exists_list[1] = True
        # chack and store if all cluster image is generated
        all_cluster_path = os.path.join(dir_path, "AllClusters.jpg")
        if os.path.isfile(all_cluster_path):
            exists_list[2] = True
        # check and store if overlaid image and directory were generated
        overlay_dir_path = os.path.join(dir_path, "OverlaidImages")
        overlay_img_path = os.path.join(overlay_dir_path, "Image1.jpg")
        if (os.path.isdir(overlay_dir_path) and
                os.path.isfile(overlay_img_path)):
            exists_list[3] = True
        # check and store if til image outputs were generated
        til_overlay_path = os.path.join(dir_path, "ContourOverlay.jpg")
        til_mask_path = os.path.join(dir_path, "ContourMask.jpg")
        if os.path.isfile(til_overlay_path) and os.path.isfile(til_mask_path):
            exists_list[4] = True
        # check and store if mask image and directory were generated
        mask_dir_path = os.path.join(dir_path, "Masks")
        mask_img_path = os.path.join(mask_dir_path, "Image1.jpg")
        if os.path.isdir(mask_dir_path) and os.path.isfile(mask_img_path):
            exists_list[5] = True
        # check and store if csv file was generated
        csv_path = os.path.join(dir_path, "Compiled_Data.csv")
        if os.path.isfile(csv_path):
            exists_list[6] = True
        # if the parent directory exists, delete it and all files it contains
        if exists_list[0]:
            shutil.rmtree(dir_path)

        return exists_list

    def test_image_postprocessing(self):
        """
        Test for image postprocessing wrapper function
        """
        # get current directory
        home = os.getcwd()
        # generate mask of contour data for input to function
        double_blob = skimage.data.binary_blobs(100, 0.5, 2, 0.25, 1)
        # make various arrays to act as image and incorrect image inputs
        dim_array_3 = np.ones((100, 100, 3), np.uint8)
        wr_chan_array = np.ones((100, 100, 4), np.uint8)
        wr_shape_array = np.ones((100, 50, 3), np.uint8)
        # ensure raise error when clusters has wrong dimension
        with self.assertRaises(ValueError):
            _ = cluster_processing.image_postprocessing(dim_array_3,
                                                               dim_array_3,
                                                               home)
        # ensure raises error when image has wrong dimension
        with self.assertRaises(ValueError):
            _ = cluster_processing.image_postprocessing(double_blob,
                                                               double_blob,
                                                               home)
        # ensure raises error when image does not have RGB channels
        with self.assertRaises(ValueError):
            _ = cluster_processing.image_postprocessing(double_blob,
                                                               wr_chan_array,
                                                               home)
        # ensure raises error when image and clusters have different X, Y
        # dimensions
        with self.assertRaises(ValueError):
            _ = cluster_processing.image_postprocessing(double_blob,
                                                               wr_shape_array,
                                                               home)
        # run function without any file outputs
        til_count, mask, index = cluster_processing.image_postprocessing(double_blob,
                                                                   dim_array_3,
                                                                   home)
        # ensure correct number of tils was generated
        self.assertTrue(til_count == 4)
        self.assertIsInstance(mask, np.ndarray)
        self.assertIsInstance(index, int)
        # check aligned file outputs
        file_exists = self.file_exists_function()
        self.assertTrue(all(([False, False, False, False, False, False, False],
                        file_exists)))
        # run through various boolean arguments and ensure the expected output
        # files are generated and unexpected files do not exist
        til_count = \
            cluster_processing. \
            image_postprocessing(double_blob, dim_array_3,
                                 home, gen_all_clusters=True)
        file_exists = self.file_exists_function()
        self.assertTrue(all(([True, True, True, False, False, False, False],
                        file_exists)))

        til_count = \
            cluster_processing.image_postprocessing(double_blob,
                                                           dim_array_3,
                                                           home,
                                                           gen_overlays=True)
        file_exists = self.file_exists_function()
        self.assertTrue(all(([True, True, False, True, False, False, False],
                        file_exists)))

        til_count = \
            cluster_processing.image_postprocessing(double_blob,
                                                           dim_array_3,
                                                           home,
                                                           gen_tils=True)
        file_exists = self.file_exists_function()
        self.assertTrue(all(([True, True, False, False, True, False, False],
                        file_exists)))

        til_count = \
            cluster_processing.image_postprocessing(double_blob,
                                                           dim_array_3,
                                                           home,
                                                           gen_masks=True)
        file_exists = self.file_exists_function()
        self.assertTrue(all(([True, True, False, False, False, True, False],
                        file_exists)))

        til_count = \
            cluster_processing.image_postprocessing(double_blob,
                                                           dim_array_3,
                                                           home,
                                                           gen_csv=True)
        file_exists = self.file_exists_function()
        self.assertTrue(all(([True, False, False, False, False, False, True],
                        file_exists)))

        til_count = \
            cluster_processing.image_postprocessing(double_blob,
                                                           dim_array_3,
                                                           home,
                                                           gen_tils=True,
                                                           gen_csv=True)
        file_exists = self.file_exists_function()
        self.assertTrue(all(([True, True, False, False, True, False, True],
                        file_exists)))

        til_count = \
            cluster_processing.image_postprocessing(
                double_blob, dim_array_3, home, gen_all_clusters=True,
                gen_overlays=True, gen_tils=True, gen_masks=True, gen_csv=True)
        file_exists = self.file_exists_function()
        self.assertTrue(all(([True, True, True, True, True, True, True],
                        file_exists)))
