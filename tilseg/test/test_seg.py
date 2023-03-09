"""
Unittests for cluster module
"""

import os
import re

import unittest
import tilseg.seg
import sklearn.linear_model
import numpy as np
import sklearn.cluster

from PIL import UnidentifiedImageError
from sklearn.exceptions import NotFittedError

print(os.getcwd())

current_dir = re.findall(f"{os.sep}[a-zA-Z0-9]+$",os.getcwd())[0][1:]
if current_dir == "tilseg":
    test_patch_path = os.path.join("..","abi_patches","test", "test_patch.tif")
    fail_test_patch_path = os.path.join("..","abi_patches","test", "test_img.txt")
elif current_dir == "TILseg":
    test_patch_path = os.path.join("abi_patches","test", "test_patch.tif")
    fail_test_patch_path = os.path.join("abi_patches","test", "test_img.txt")
elif current_dir == "test":
    test_patch_path = os.path.join("..","..","abi_patches","test", "test_patch.tif")
    fail_test_patch_path = os.path.join("..","..","abi_patches","test", "test_img.txt")

class TestClusterModelFit(unittest.TestCase):
    """
    Unittests for tilseg.cluster.cluster_model_fit
    """

    def test_smoke(self):
        """
        Tests if code runs with correct inputs
        """
        try:
            model = tilseg.seg.cluster_model_fit(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters=4)
        except:
            self.assertTrue(False)

    def test_patch_path_type(self):
        """
        Tests that non-string input for patch_path is dealt with
        """
        with self.assertRaises(TypeError):
            model = tilseg.seg.cluster_model_fit(
                patch_path=2,
                algorithm='KMeans',
                n_clusters=4)

    def test_patch_file(self):
        """
        Testing when input file does not exist
        """
        with self.assertRaises(ValueError):
            model = tilseg.seg.cluster_model_fit(
                patch_path=test_patch_path+'blahblah',
                algorithm='KMeans',
                n_clusters=4)
 
    def test_patch_img(self):
        """
        Testing when input file is not an image
        """
        with self.assertRaises(UnidentifiedImageError):
            model = tilseg.seg.cluster_model_fit(
                patch_path=fail_test_patch_path,
                algorithm='KMeans',
                n_clusters=4)

    def test_algorithm_in(self):
        """
        Testing when non-string input for algorithm
        """
        with self.assertRaises(ValueError):
            model = tilseg.seg.cluster_model_fit(
                patch_path=test_patch_path,
                algorithm=5,
                n_clusters=4)

    def test_n_clusters_KMeans(self):
        """
        Testing that n_clusters is numerical when algorithm is KMeans
        """
        with self.assertRaises(ValueError):
            model = tilseg.seg.cluster_model_fit(
                patch_path=test_patch_path,
                algorithm='KMeans'
                )

    def test_n_clusters_in1(self):
        """
        Tests when n_cluster input is not an integer
        """
        with self.assertRaises(ValueError):
            model = tilseg.seg.cluster_model_fit(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters='4'
                )

    def test_n_clusters_in2(self):
        """
        Tests when n_cluster input is more than 8
        """
        with self.assertRaises(ValueError):
            model = tilseg.seg.cluster_model_fit(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters=9
                )

model_KMeans = tilseg.seg.cluster_model_fit(
                patch_path=test_patch_path,
                algorithm='KMeans',
                n_clusters=4)

model_unfitted = model = sklearn.cluster.KMeans(n_clusters=4, max_iter=20, n_init=3, tol=1e-3)

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
        [ch_score, db_score] = tilseg.seg.clustering_score(model=model_KMeans, patch_path=test_patch_path)
        self.assertTrue((isinstance(ch_score, float)))
        self.assertTrue((isinstance(db_score, float)))

    def test_patch_path_type(self):
        """
        Tests that non-string input for patch_path is dealt with
        """
        with self.assertRaises(TypeError):
            [ch_score, db_score] = tilseg.seg.clustering_score(model=model_KMeans, patch_path=2)

    def test_patch_file(self):
        """
        Testing when input file does not exist
        """
        with self.assertRaises(ValueError):
            [ch_score, db_score] = tilseg.seg.clustering_score(model=model_KMeans, patch_path=test_patch_path+'blahblah')
 
    def test_patch_img(self):
        """
        Testing when input file is not an image
        """
        with self.assertRaises(UnidentifiedImageError):
            [ch_score, db_score] = tilseg.seg.clustering_score(model=model_KMeans, patch_path=fail_test_patch_path)

    def test_model_in1(self):
        """
        Testing when input model is not an estimator
        """
        with self.assertRaises(TypeError):
            [ch_score, db_score] = tilseg.seg.clustering_score(model=5, patch_path=test_patch_path)

    def test_model_in2(self):
        """
        Testing when input model is not fitted
        """
        with self.assertRaises(NotFittedError):
            [ch_score, db_score] = tilseg.seg.clustering_score(model=model_unfitted, patch_path=test_patch_path)

    def test_cluster_model(self):
        """
        Testing when input model is not sklearn.cluster.model
        """
        with self.assertRaises(ValueError):
            [ch_score, db_score] = tilseg.seg.clustering_score(model=model_linear, patch_path=test_patch_path)