"""Unittests for preprocessing module."""

# Core library imports
import collections
import os
import unittest

# External libraries
import numpy as np
import openslide
import pandas as pd
import pytest
import sys

# Local imports: add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tilseg import preprocessing

# pylint: disable=useless-return

# Modify options for pandas to stop chained assignment warning
pd.options.mode.chained_assignment = None
collections.Callable = collections.abc.Callable

# Change this if svs full slide image available
# Since a full slide image can't be uploaded to github due to file size
#   the tests marked with pytest.mark.skipif's can't be run as part of
#   the github workflow. The tests were run locally with a full slide image
#   available, and passed, but can't run on github
#   Since svs is proprietary, and generated from instruments, it has
#   proven challenging to create a small svs file for testing purposes.
#   For now, the tests which require it are being skipped.
FULL_SLIDE_AVAILABLE = False


class TestPreProcessing(unittest.TestCase):
    """Test case for the functions within preprocessing.py"""

    # def test_open_slide(self):
    #     """
    #     open_slide testings
    #     ensure open_slide function runs
    #     """
    #     # edge test: ensures path is a string
    #     with self.assertRaises(TypeError):
    #         path1 = 3.0
    #         preprocessing.open_slide(path1)
    #
    #     # edge test: ensure path exits
    #     with self.assertRaises(ValueError):
    #         preprocessing.open_slide('Users/cyrushaas/Documents/WI23_\
    #                                   MolecularDataScience/project/whole_\
    #                                   slide.svs')
    #
    # # smoke test: checks the function can open svs file
    # !! i'm not sure if this file is oaky
    # path2 = '/home/braden/Project/TCGA-3C-AALI-01Z-00-DX1.svs'
    # preprocessing.open_slide(path2)

    def test_get_tile_size(self):
        """Test get_tile_size function."""
        # smoke test: make sure get_tile_size executes
        preprocessing.get_tile_size(4000, 12000, cutoff=4)

        # edge test: make sure invalid max is caught
        with self.assertRaises(TypeError):
            preprocessing.get_tile_size('a', 12000)

        # edge test: make sure invalid size is caught
        with self.assertRaises(TypeError):
            preprocessing.get_tile_size(4000, '12000')

        # edge test: make sure invalid cutoff is caught
        with self.assertRaises(TypeError):
            preprocessing.get_tile_size(4000, 12000, 'b')

        # one shot (pattern?) test: get expected patch size,
        # number of patches, and loss
        patch_size, num_patches, loss = preprocessing.get_tile_size(4000,
                                                                    12000)
        assert np.isclose(patch_size, 4000)
        assert np.isclose(num_patches, 3)
        assert np.isclose(loss, 0)

        return

    def test_percent_of_pixels_lost(self):
        """Test percent_of_pixels_lost function."""
        # edge test: make sure invalid lost_x is caught
        with self.assertRaises(TypeError):
            preprocessing.percent_of_pixels_lost('10', 3000, 10, 3000,
                                                 4000, 4000)

        # edge test: make sure invalid patch_x is caught
        with self.assertRaises(TypeError):
            preprocessing.percent_of_pixels_lost(10, '3000', 10, 3000,
                                                 4000, 4000)

        # edge test: make sure invalid lost_y is caught
        with self.assertRaises(TypeError):
            preprocessing.percent_of_pixels_lost(10, 3000, '10', 3000,
                                                 4000, 4000)

        # edge test: make sure invalid patch_y is caught
        with self.assertRaises(TypeError):
            preprocessing.percent_of_pixels_lost(10, 3000, 10, '3000',
                                                 4000, 4000)

        # edge test: make sure invalid x_size is caught
        with self.assertRaises(TypeError):
            preprocessing.percent_of_pixels_lost(10, 3000, 10, 3000,
                                                 '4000', 4000)

        # edge test: make sure invalid y_size is caught
        with self.assertRaises(TypeError):
            preprocessing.percent_of_pixels_lost(10, 3000, 10, 3000,
                                                 4000, '4000')

        # smoke test: ensuring percent_of_pixels_lost runs
        preprocessing.percent_of_pixels_lost(10, 3000, 10, 3000, 4000, 4000)

        # pattern test: answer loss should be zero as written
        percentt = preprocessing.percent_of_pixels_lost(0, 300, 0, 300,
                                                        4000, 4000)
        assert np.isclose(percentt, 0)

        # one shot test: answer should be zero as written
        percentt = preprocessing.percent_of_pixels_lost(10, 4, 10, 4, 40, 40)
        assert np.isclose(percentt, 10)

        return

    def test_save_image(self):
        """Test save_image function."""
        dummy_img = np.array(np.random.rand(100, 100, 3) * 255,dtype=np.uint8)
        dummy1_img = np.array(np.random.rand(100, 100) * 255, dtype = np.uint8)
        dummy2_img = np.array(np.random.rand(100, 100, 2) * 255, dtype = np.uint8)

        # edge test: ensure error is raised if path is not a string
        with self.assertRaises(TypeError):
            preprocessing.save_image(4, 'test', dummy_img)

        # edge test: ensure the image array has some number of channels
        with self.assertRaises(IndexError):
            preprocessing.save_image('path', 'name', dummy1_img)

        # edge test: ensure image array has 3 channels (RGB)
        with self.assertRaises(IndexError):
            preprocessing.save_image('path', 'name', dummy2_img)

        # edge test: path sure the path exists
        with self.assertRaises(ValueError):
            preprocessing.save_image('./whole_slide.svs',
                                     'image1', dummy_img)

        # edge test: ensure name has file extenstion
        with self.assertRaises(ValueError):
            preprocessing.save_image('path', 'test', dummy_img)

        # smoke test: make sure the function runs
        preprocessing.save_image(os.getcwd(), 'test_image.jpeg', dummy_img)

        # one shot test: make sure the image is saved
        preprocessing.save_image(os.getcwd(), 'test_image.jpeg', dummy_img)
        newimg_path = os.path.join(os.getcwd(), 'test_image.jpeg')

        if os.path.exists(newimg_path):
            pass
        else:
            raise TypeError('test failed!!!!!!')
        # Delete image when test complete
        os.remove(os.path.join(os.getcwd(), 'test_image.jpeg'))
        return

    @pytest.mark.skipif(not FULL_SLIDE_AVAILABLE,
                        reason="No fullslide image available")
    def test_create_patches(self):
        """Test create_patches function."""
        slide = openslide.OpenSlide

        # edge test: ensure ypatch must be an int
        with self.assertRaises(TypeError):
            preprocessing.create_patches(slide, '5', 5, 200, 200)

        # edge test: ensure xpatch must be an int
        with self.assertRaises(TypeError):
            preprocessing.create_patches(slide, 5, '5', 200, 200)

        # edge test: ensure xdim must be an int
        with self.assertRaises(TypeError):
            preprocessing.create_patches(slide, 5, 5, '200', 200)

        # edge test: ensure ydim must be an int
        with self.assertRaises(TypeError):
            preprocessing.create_patches(slide, 5, 5, 200, '200')

        # edge test: make sure too big xslides are caught
        with self.assertRaises(IndexError):
            preprocessing.create_patches(slide, 5, 2, 100000, 2000)

        # edge test: make sure too big yslides are caught
        with self.assertRaises(IndexError):
            preprocessing.create_patches(slide, 3, 5, 200, 70000)

        # pattern (?) test: checking the correct number of patches are created
        # memory error
        # np_patches, patch_position = preprocessing.create_patches(slide, 4,
        #                                                           4, 25000,
        #                                                           17500)
        #
        # smoke test: making sure the function runs (takes 45 mins hehe)
        # preprocessing.create_patches(slide, 20, 3000, 20, 3000)

        return

    def test_get_average_color(self):
        """Test get_average_color function."""
        dummy_img = np.random.rand(100, 100, 3) * 255
        dummy_img1 = np.random.rand(100, 100, 2) * 255
        dummy_img2 = np.zeros((100, 100, 3), dtype=int)

        # edge test: ensure error caught if array not passed
        with self.assertRaises(TypeError):
            preprocessing.get_average_color('tester')

        # edge test: ensure error caught if not 3 channel img
        with self.assertRaises(IndexError):
            preprocessing.get_average_color(dummy_img1)

        # smoke test: make sure function runs
        preprocessing.get_average_color(dummy_img)

        # pattern test: make sure the average function works
        avg = preprocessing.get_average_color(dummy_img2)
        assert (avg == [0, 0, 0]).all()

        return

    def test_get_grey(self):
        """Test get_grey function."""
        # edge test: make sure that a string is not accepted
        with self.assertRaises(TypeError):
            preprocessing.get_grey('tester')

        # edge test: make sure list is long enough
        with self.assertRaises(IndexError):
            preprocessing.get_grey([1, 1])

        # edge test: make sure list is not too long
        with self.assertRaises(IndexError):
            preprocessing.get_grey([1, 1, 1, 1])

        # smoke test: make sure function runs
        preprocessing.get_grey([1, 1, 1])

        # pattern test: know the average
        assert np.isclose(150, preprocessing.get_grey([125, 150, 175]))

        return

    def test_save_all_images(self):
        """Test save_all_images function."""
        d = {'patch_xy': [1, 2, 3, 4, 5]}

        # error when no correct path is given
        with self.assertRaises(FileNotFoundError):
            preprocessing.save_all_images(pd.DataFrame(), '',
                                          'test_file.svs')

        with self.assertRaises(TypeError):
            # error when file does not have extension
            preprocessing.save_all_images(pd.DataFrame(), '.',
                                          'test_filesvs')
            # error when dataframe not given
            preprocessing.save_all_images(pd.Series(), '.',
                                          'test_file.svs')
            # error when patch_xy has no tuples
            preprocessing.save_all_images(pd.DataFrame(data=d), '.',
                                          'test_file.svs')

        with self.assertRaises(ValueError):
            # error when patch_xy is not a column in the dataframe
            preprocessing.save_all_images(pd.DataFrame(
                                          columns=['A', 'B', 'C']),
                                          '.', 'f.svs')

        return

    def test_is_it_background(self):
        """Test is_it_background function."""
        # test if it works as expected
        self.assertTrue(preprocessing.is_it_background(200, 240))
        self.assertFalse(preprocessing.is_it_background(221, 198))

        return

    def test_sort_patches(self):
        """Test sort_patches function."""
        d = {'greys': [1, 2, 3, 4, 'a']}
        d2 = {'e': [1, 2, 3, 4, 5]}

        with self.assertRaises(TypeError):
            # check that no dataframe will raise an error
            preprocessing.sort_patches('a')
            # check that a dataframe without numeric values for
            # type column will raise errors
            preprocessing.sort_patches(pd.DataFrame(data=d))

        with self.assertRaises(KeyError):
            # check that error is raised with no greys column
            preprocessing.sort_patches(pd.DataFrame(data=d2))

        return

    @pytest.mark.skipif(not FULL_SLIDE_AVAILABLE,
                        reason="No fullslide image available")
    def test_count_images(self):
        """Test count_images function."""
        # make svs dummy file path
        svs_path = os.path.join(os.path.dirname(__file__), 'test',
                                'test_patches', 'dummy.svs')

        # smoke test: make sure the function runs
        preprocessing.count_images()

        # pattern test: there is no .svs file in this
        # test writer's cwd
        assert np.isclose(0, preprocessing.count_images())

        # BKC: MAYBE DELETE!!!!!!!!!!!1
        # pattern test: this writer knows there one file in this dir
        assert np.isclose(1, preprocessing.count_images(svs_path))

        return

    @pytest.mark.skipif(not FULL_SLIDE_AVAILABLE,
                        reason="No fullslide image available")
    def test_patches_per_img(self):
        """Test patches_per_img function"""
        # make svs dummy file path
        svs_path = os.path.join(os.path.dirname(__file__), 'test',
                                'test_patches', 'dummy.svs')

        # smoke test: making sure the function runs
        preprocessing.patches_per_img(6)

        # pattern test: test writer knows there are no images in cwd
        assert np.isclose(0, preprocessing.patches_per_img(6))

        # # pattern test: test writer knows there one image in this path
        assert np.isclose(6, preprocessing.patches_per_img(6, svs_path))

        # edge test: ensure string value not accepted for num_patches
        with self.assertRaises(TypeError):
            preprocessing.patches_per_img('6')

        # edge test: ensure a path that does not exist is not accessed
        with self.assertRaises(ValueError):
            preprocessing.patches_per_img(6, 'filepath')

    @pytest.mark.skipif(not FULL_SLIDE_AVAILABLE,
                        reason="No fullslide image available")
    def test_get_superpatch_patches(self):
        """Test get_superpatch_patches"""
        # make svs dummy file path
        svs_path = os.path.join(os.path.dirname(__file__), 'test',
                                'test_patches', 'dummy.svs')

        # make csv dummy file path
        csv_path = os.path.join(os.path.dirname(__file__), 'test',
                                'test_patches', 'dummy.csv')

        # edge test: make sure nondataframe is not accepted
        dummy_img = np.random.rand(100, 100)
        with self.assertRaises(TypeError):
            preprocessing.get_superpatch_patches(dummy_img)

        # edge test: make sure nonexistent path is caught
        test_df = pd.read_csv(csv_path)
        with self.assertRaises(ValueError):
            preprocessing.get_superpatch_patches(test_df,
                                                 path='.')

        # edge test: make sure non int patches is not accepted
        with self.assertRaises(TypeError):
            df1 = pd.DataFrame(data=dummy_img)
            preprocessing.get_superpatch_patches(df1, '6')

        # # smoke test: make sure function runs
        preprocessing.get_superpatch_patches(test_df,
                                             path=svs_path)

        # edge test: make sure dataframe contains patches column
        test_df1 = test_df.drop(['patches'], axis=1)
        with self.assertRaises(KeyError):
            preprocessing.get_superpatch_patches(test_df1,
                                                 path=svs_path)

        # edge test: make sure dataframe contains id columns
        test_df2 = test_df.drop(['id'], axis=1)
        with self.assertRaises(KeyError):
            preprocessing.get_superpatch_patches(test_df2,
                                                 path=svs_path)

        # pattern test: know six patches should be output
        expected = len(preprocessing.get_superpatch_patches(test_df,
                                                            patches=5,
                                                            path=svs_path))
        assert np.isclose(5, expected)

        return

    @pytest.mark.skipif(not FULL_SLIDE_AVAILABLE,
                        reason="No fullslide image available")
    def test_superpatcher(self):
        """Test superpatcher function"""
        # make svs dummy file path
        svs_path = os.path.join(os.path.dirname(__file__), 'test',
                                'test_patches', 'dummy.svs')

        # make csv dummy file path
        csv_path = os.path.join(os.path.dirname(__file__), 'test',
                                'test_patches', 'dummy.csv')

        test_df = pd.read_csv(csv_path)
        expected = preprocessing.get_superpatch_patches(test_df, patches=5,
                                                        path=svs_path)

        # edge test: ensure nonint sp_width not accepted
        with self.assertRaises(TypeError):
            preprocessing.superpatcher(expected, '3')

        # edge test: patches column does not actually contain arrays
        with self.assertRaises(TypeError):
            preprocessing.superpatcher(expected)

        return


if __name__ == '__main__':
    unittest.main()
