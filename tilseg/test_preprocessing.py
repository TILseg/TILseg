"""Unittests for preprocessing module."""

import collections
import openslide
import unittest
import preprocessing

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

collections.Callable = collections.abc.Callable

class TestPreProcessing(unittest.TestCase):
    """Test case for the functions within preprocessing.py"""
    
    def test_save_all_images(self):
        """Test save_all_images function."""

        d = {'patch_xy': [1,2,3,4,5]}

        # error when no correct path is given
        with self.assertRaises(FileNotFoundError):
            preprocessing.save_all_images(pd.DataFrame(), '', 'test_file.svs')

        with self.assertRaises(TypeError):
            # error when file does not have extension
            preprocessing.save_all_images(pd.DataFrame(), '/Users/', 'test_filesvs')
            # error when dataframe not given
            preprocessing.save_all_images(pd.Series(), '/Users/', 'test_file.svs')
            # error when patch_xy has no tuples
            preprocessing.save_all_images(pd.DataFrame(data=d), '/Users/', 'test_file.svs')

        with self.assertRaises(ValueError):
            # error when patch_xy is not a column in the dataframe
            preprocessing.save_all_images(pd.DataFrame(columns=['A','B','C']), '/Users/', 'f.svs')

        return
    
    def test_find_max(self):
        """Test find_max function."""

        with self.assertRaises(TypeError):
            # error if a nonboolean is passed through for the greater than argument
            preprocessing.find_max([1,2,3], 1, 'a')
            # error if 'None' is passed through for the greater than argument
            preprocessing.find_max([1,2,3], 1, None)
            # error if not a list/array
            preprocessing.find_max('s', 1, True)
            # error if the cuttoff value is not numericc
            preprocessing.find_max([1,4,2], 'a', False)
            
        with self.assertRaises(ValueError):
            # error if array contains negative numbers
            preprocessing.find_max([1,2,-3], 4.3, True)

        # check that max is working as expected
        self.assertEqual(preprocessing.find_max([3,6,100], 1, True), 1)
        self.assertEqual(preprocessing.find_max([3,10,20,33,103,6,100], 4, True), 4)
        self.assertEqual(preprocessing.find_max([3,10,20,33,103,6,100], 2, False), 4)

        return
    
    def test_find_min(self):
        """Test find_min function."""

        with self.assertRaises(TypeError):
            # error if a non numeric is passed through for the range_min
            preprocessing.find_min([1,2,3], '1', 3)
            # error if a non numeric is passed through for the range_max
            preprocessing.find_min([1,2,3], 1.3, '0')
            # error if not a list/array
            preprocessing.find_min('s', 1, 23)
            
        with self.assertRaises(ValueError):
            # error if array contains negative numbers
            preprocessing.find_min([1,2,-3], 4.3, True)

        with self.assertRaises(AssertionError):
            # error if range is equal to each other
            preprocessing.find_min([1,2,3,4,5,6], 11, 11)
            # error if range is not right
            preprocessing.find_min([1,2,3,4,5,6], 10, 3)

        # check that max is working as expected
        self.assertEqual(preprocessing.find_min([3,6,100, 9], 0, 3), 1)
        self.assertEqual(preprocessing.find_min([3,10,20,33,103,6,100], 4, 6), 5)
        self.assertEqual(preprocessing.find_min([3,10,20,33,103,6,100], 2, 6), 5)

        return
    
    def test_is_it_background(self):
        """Test is_it_background function."""

        # test if it works as expected
        self.assertTrue(preprocessing.is_it_background(200, 240))
        self.assertFalse(preprocessing.is_it_background(221, 198))

        return
    
    def test_sort_patches(self):
        """Test sort_patches function."""

        d = {'greys': [1,2,3,4,'a']}
        d2 = {'e': [1,2,3,4,5]}

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
    

if __name__ == '__main__':
    unittest.main()