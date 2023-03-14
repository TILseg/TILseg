"""Unittests for preprocessing module."""

import collections
import os
import unittest
import openslide
import tilseg.preprocessing as preprocessing

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

collections.Callable = collections.abc.Callable

# pylint: disable=useless-return


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
        dummy_img = np.random.rand(100, 100, 3) * 255
        dummy1_img = np.random.rand(100, 100) * 255
        dummy2_img = np.random.rand(100, 100, 2) * 255

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
            preprocessing.save_image('Users/cyrushaas/Documents/WI23_Mole\
                                     cularDataScience/project/whole_slide.svs',
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

        return

    def test_create_patches(self):
        """Test create_patches function."""
        slide = openslide.OpenSlide('/home/braden/TILseg/TCGA-3C\
                                    -AALI-01Z-00-DX1.svs')

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
            preprocessing.save_all_images(pd.DataFrame(), '/Users/',
                                          'test_filesvs')
            # error when dataframe not given
            preprocessing.save_all_images(pd.Series(), '/Users/',
                                          'test_file.svs')
            # error when patch_xy has no tuples
            preprocessing.save_all_images(pd.DataFrame(data=d), '/Users/',
                                          'test_file.svs')

        with self.assertRaises(ValueError):
            # error when patch_xy is not a column in the dataframe
            preprocessing.save_all_images(pd.DataFrame(
                                          columns=['A', 'B', 'C']),
                                          '/Users/', 'f.svs')

        return

    def test_find_max(self):
        """Test find_max function."""
        with self.assertRaises(TypeError):
            # error if a nonboolean is passed through for the greater
            # than argument
            preprocessing.find_max([1, 2, 3], 1, 'a')
            # error if 'None' is passed through for the greater than argument
            preprocessing.find_max([1, 2, 3], 1, None)
            # error if not a list/array
            preprocessing.find_max('s', 1, True)
            # error if the cuttoff value is not numericc
            preprocessing.find_max([1, 4, 2], 'a', False)

        with self.assertRaises(ValueError):
            # error if array contains negative numbers
            preprocessing.find_max([1, 2, -3], 4.3, True)

        # check that max is working as expected
        self.assertEqual(preprocessing.find_max([3, 6, 100], 1, True), 1)
        self.assertEqual(preprocessing.find_max([3, 10, 20, 33, 103, 6, 100],
                                                4, True), 4)
        self.assertEqual(preprocessing.find_max([3, 10, 20, 33, 103, 6, 100],
                                                2, False), 4)

        return

    def test_find_min(self):
        """Test find_min function."""
        with self.assertRaises(TypeError):
            # error if a non numeric is passed through for the range_min
            preprocessing.find_min([1, 2, 3], '1', 3)
            # error if a non numeric is passed through for the range_max
            preprocessing.find_min([1, 2, 3], 1.3, '0')
            # error if not a list/array
            preprocessing.find_min('s', 1, 23)

        with self.assertRaises(ValueError):
            # error if array contains negative numbers
            preprocessing.find_min([1, 2, -3], 4.3, True)

        with self.assertRaises(AssertionError):
            # error if range is equal to each other
            preprocessing.find_min([1, 2, 3, 4, 5, 6], 11, 11)
            # error if range is not right
            preprocessing.find_min([1, 2, 3, 4, 5, 6], 10, 3)

        # check that max is working as expected
        self.assertEqual(preprocessing.find_min([3, 6, 100, 9], 0, 3), 1)
        self.assertEqual(preprocessing.find_min([3, 10, 20, 33, 103, 6, 100],
                                                4, 6), 5)
        self.assertEqual(preprocessing.find_min([3, 10, 20, 33, 103, 6, 100],
                                                2, 6), 5)

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

    def test_count_images(self):
        """Test count_images function."""
        # smoke test: make sure the function runs
        preprocessing.count_images()

        # pattern test: there is no .svs file in this
        # test writer's cwd
        assert np.isclose(0, preprocessing.count_images())

        # BKC: MAYBE DELETE!!!!!!!!!!!1
        # pattern test: this writer knows there one file in this dir
        assert np.isclose(1, preprocessing.count_images('/home/bra\
                                                        den/TILseg'))

        return

    def test_patches_per_img(self):
        """Test patches_per_img function"""
        # smoke test: making sure the function runs
        preprocessing.patches_per_img(6)

        # pattern test: test writer knows there are no images in cwd
        assert np.isclose(0, preprocessing.patches_per_img(6))

        # pattern test: test writer knows there one image in this path
        assert np.isclose(6, preprocessing.patches_per_img(6,
                                                           '/home/bra\
                                                            den/TILseg'))

        # edge test: ensure string value not accepted for num_patches
        with self.assertRaises(TypeError):
            preprocessing.patches_per_img('6')

        # edge test: ensure a path that does not exist is not accessed
        with self.assertRaises(ValueError):
            preprocessing.patches_per_img(6, 'filepath')

    def test_get_superpatch_patches(self):
        """Test get_superpatch_patches"""
        # edge test: make sure nondataframe is not accepted
        dummy_img = np.random.rand(100, 100)
        with self.assertRaises(TypeError):
            preprocessing.get_superpatch_patches(dummy_img)

        # edge test: make sure nonexistent path is caught
        test_df = pd.read_csv('/home/braden/Project/dummy.csv')
        with self.assertRaises(ValueError):
            preprocessing.get_superpatch_patches(test_df,
                                                 path='/home/braden/poo')

        # edge test: make sure non int patches is not accepted
        with self.assertRaises(TypeError):
            df1 = pd.DataFrame(data=dummy_img)
            preprocessing.get_superpatch_patches(df1, '6')

        # smoke test: make sure function runs
        preprocessing.get_superpatch_patches(test_df,
                                             path='/home/braden/TILseg')

        # edge test: make sure dataframe contains patches column
        test_df1 = test_df.drop(['patches'], axis=1)
        with self.assertRaises(KeyError):
            preprocessing.get_superpatch_patches(test_df1,
                                                 path='/home/braden/TILseg')

        # edge test: make sure dataframe contains id columns
        test_df2 = test_df.drop(['id'], axis=1)
        with self.assertRaises(KeyError):
            preprocessing.get_superpatch_patches(test_df2,
                                                 path='/home/braden/TILseg')

        # pattern test: know six patches should be output
        expected = len(preprocessing.get_superpatch_patches(test_df,
                                                            patches=5,
                                                            path='/home/br\
                                                                aden/TILseg'))
        assert np.isclose(5, expected)

        return

    def test_superpatcher(self):
        """Test superpatcher function"""
        test_df = pd.read_csv('/home/braden/Project/dummy.csv')
        expected = preprocessing.get_superpatch_patches(test_df, patches=5,
                                                        path='/home/br\
                                                        aden/TILseg')

        # edge test: ensure nonint sp_width not accepted
        with self.assertRaises(TypeError):
            preprocessing.superpatcher(expected, '3')

        # edge test: patches column does not actually contain arrays
        with self.assertRaises(TypeError):
            preprocessing.superpatcher(expected)

        return


if __name__ == '__main__':
    unittest.main()
