"""
Unittests for cluster module
"""

import unittest
import tilseg.cluster
import sklearn.linear_model
import numpy as np

test_patch_path = '/Users/abishek/Desktop/DataScienceClasses/TILSeg/abi_patches/test_patch.tif'
fail_test_patch_path = '/Users/abishek/Desktop/DataScienceClasses/TILSeg/abi_patches/test_img.txt'


class TestClusterModelFitter(unittest.TestCase):
    """
    Unittests for tilseg.cluster.cluster_model_fitter
    """

    def test_smoke(self):
        """
        Tests if code runs with correct inputs
        """
        try:
            model = tilseg.cluster.cluster_model_fitter(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters=4)
        except:
            self.assertTrue(False)

    def test_patch_path_type(self):
        """
        Tests that non-string input for patch_path is dealt with
        """
        try:
            model = tilseg.cluster.cluster_model_fitter(
                patch_path=2,
                algorithm='KMeans',
                n_clusters=4)
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)

    def test_patch_file(self):
        """
        Testing when input file does not exist
        """
        try:
            model = tilseg.cluster.cluster_model_fitter(
                patch_path=test_patch_path+'blahblah',
                algorithm='KMeans',
                n_clusters=4)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
 
    # def test_patch_img(self):
    #     """
    #     Testing when input file is not an image
    #     """
    #     class UnidentifiedImageError(Exception):
    #         pass
    #     try:
    #         model = tilseg.cluster.cluster_model_fitter(
    #             patch_path=fail_test_patch_path,
    #             algorithm='KMeans',
    #             n_clusters=4)
    #         self.assertTrue(False)
    #     except UnidentifiedImageError:
    #         self.assertTrue(True)

    def test_algorithm_in(self):
        """
        Testing when non-string input for algorithm
        """
        try:
            model = tilseg.cluster.cluster_model_fitter(
                patch_path=test_patch_path,
                algorithm=5,
                n_clusters=4)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_n_clusters_KMeans(self):
        """
        Testing that n_clusters is numerical when algorithm is KMeans
        """
        try:
            model = tilseg.cluster.cluster_model_fitter(
                patch_path=test_patch_path,
                algorithm='KMeans'
                )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_n_clusters_in1(self):
        """
        Tests when n_cluster input is not an integer
        """
        try:
            model = tilseg.cluster.cluster_model_fitter(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters='4'
                )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_n_clusters_in2(self):
        """
        Tests when n_cluster input is more than 8
        """
        try:
            model = tilseg.cluster.cluster_model_fitter(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters=9
                )
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


model_KMeans = tilseg.cluster.cluster_model_fitter(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters=4)

model_linear = sklearn.linear_model.LinearRegression()
model_linear.fit(np.random.rand(5).reshape(-1,1), np.random.rand(5))


class TestClusteringScore(unittest.TestCase):
    """
    Unittests for tilseg.cluster.clustering_score
    """

    def test_smoke(self):
        """
        Testing if the outputs are floats and the code runs with good inputs
        """
        [ch_score, db_score] = tilseg.cluster.clustering_score(model=model_KMeans, patch_path=test_patch_path)
        self.assertTrue((isinstance(ch_score, float)))
        self.assertTrue((isinstance(db_score, float)))

    def test_patch_path_type(self):
        """
        Tests that non-string input for patch_path is dealt with
        """
        try:
            [ch_score, db_score] = tilseg.cluster.clustering_score(model=model_KMeans, patch_path=2)
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)

    def test_patch_file(self):
        """
        Testing when input file does not exist
        """
        try:
            [ch_score, db_score] = tilseg.cluster.clustering_score(model=model_KMeans, patch_path=test_patch_path+'blahblah')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
 
    # def test_patch_img(self):
    #     """
    #     Testing when input file is not an image
    #     """
    #     class UnidentifiedImageError(Exception):
    #         pass
    #     try:
    #         [ch_score, db_score] = tilseg.cluster.clustering_score(model=model_KMeans, patch_path=test_patch_path)
    #         self.assertTrue(False)
    #     except UnidentifiedImageError:
    #         self.assertTrue(True)

    # def test_model_in1(self):
    #     """
    #     Testing when input model is not an estimator
    #     """
    #     try:
    #         [ch_score, db_score] = tilseg.cluster.clustering_score(model=5, patch_path=test_patch_path)
    #         self.assertTrue(False)
    #     except TypeError:
    #         self.assertTrue(True)

    # def test_model_in2(self):
    #     """
    #     Testing when input model is not fitted
    #     """
    #     try:
    #         [ch_score, db_score] = tilseg.cluster.clustering_score(model=model_KMeans, patch_path=test_patch_path)
    #         self.assertTrue(False)
    #     except:
    #         self.assertTrue(True)

    def test_cluster_model(self):
        """
        Testing when input model is not sklearn.cluster.model
        """
        try:
            [ch_score, db_score] = tilseg.cluster.clustering_score(model=model_linear, patch_path=test_patch_path)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
