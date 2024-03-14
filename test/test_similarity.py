import unittest
import numpy as np
from numpy.random import rand
from sklearn.metrics import mean_squared_error
from tilseg.similarity import image_similarity
from tilseg.similarity import superpatch_similarity

class TestImageSimilarity(unittest.TestCase):
    """
    Test case for the image_similiarity function within similarity.py
    """

    def test_input_type(self):
        """
        test the inputs of the function is indeed np.array
        """
        with self.assertRaises(TypeError):
            image_similarity('t',np.array([1,2,3]))

        with self.assertRaises(TypeError):
            image_similarity(np.array([1,2,3]),'tt')

    def test_input_shape(self):
        """
        test if the two input arrays have the same shape
        """
        with self.assertRaises(ValueError):
            image_similarity(rand(4),rand(5))
    
    def test_mean_squared_error(self):
        """
        test the mse output
        """
        # Test Case 1
        image1 = rand(2,2)
        image2 = rand(2,2)
        mse, _ = image_similarity(image1, image2)
        expected_mse = mean_squared_error(image1.flatten(), image2.flatten())
        self.assertAlmostEqual(mse, expected_mse, places=5)

        # Test case 2
        image1 = rand(5,5)
        image2 = rand(5,5)
        mse, _ = image_similarity(image1, image2)
        expected_mse = mean_squared_error(image1.flatten(), image2.flatten())
        self.assertAlmostEqual(mse, expected_mse, places=5)

    def test_difference_array(self):
        """
        test the diff output
        """

        # Test Case 1
        image1 = np.array([[0, 0], [0, 0]])
        image2 = np.array([[1, 1], [1, 1]])
        _, diff_array = image_similarity(image1, image2)
        expected_diff_array = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(diff_array, expected_diff_array)


class TestSuperpatchSimilarity(unittest.TestCase):
    """
    Test case for the superpatch_similiarity function within similarity.py"""
    def test_input_type(self):
        """
        test the inputs of the function is indeed np.array
        """
        real_path = "/Users/user/Downloads/"
        with self.assertRaises(TypeError):
            superpatch_similarity(3,real_path,real_path,rand(3))

        with self.assertRaises(TypeError):
            superpatch_similarity(real_path,3,real_path,rand(3))

        with self.assertRaises(TypeError):
            superpatch_similarity(real_path,real_path,float(334.3),rand(3))

        with self.assertRaises(TypeError):
            superpatch_similarity(real_path,real_path,real_path,'not an array')

    def test_superpatchpath_contains_file(self):
        """
        test if the input superpatch path has any files in it
        """
        with self.assertRaises(ValueError):
            real_path = "/Users/user/Downloads/"
            directory_path = '/path/to/your/directory'
            superpatch_similarity(directory_path,real_path,real_path,rand(50))

if __name__ == '__main__':
    unittest.main()
