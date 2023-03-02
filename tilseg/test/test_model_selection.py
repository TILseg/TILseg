"""Unittests for model selection module
"""
# Standard Library Imports
import os
import unittest

# External Library Imports
import numpy as np
import sklearn.datasets
import sklearn.cluster

# Local imports
import tilseg.model_selection

class TestModelSelection(unittest.TestCase):
    """
    Test case for the functions within model_selection.py
    """
    cluster_data = None
    elbow_data = np.array([[1,217.64705882352948],
                [2,68.42857142857143],
                [3,16.228571428571424],
                [4,12.695238095238096],
                [5,9.6],
                [6,7.166666666666666],
                [7,5.5],
                [8,4.083333333333334],
                [9,2.9999999999999996]])
    @classmethod
    def setUpClass(cls):
        """
        Method to create testing data
        """
        cls.cluster_data = sklearn.datasets.make_blobs(n_samples = 100,
                                                       n_features = 3,
                                                       centers=None,
                                                       random_state=3141)[0]
    def test_find_elbow(self):
        """
        Test the find elbow function
        """
        n_clusters = tilseg.model_selection.find_elbow(self.elbow_data)
        self.assertIsInstance(n_clusters, int)
        self.assertEqual(n_clusters, 3)
    def test_eval_km_elbow(self):
        """
        Test the eval knn elbow function
        """
        n_clusters = tilseg.model_selection.eval_km_elbow(self.cluster_data,
                                                           list(range(1,10)),
                                                           r2_cutoff=0.9)
        self.assertAlmostEqual(n_clusters, 3)
        self.assertIsInstance(n_clusters, int)
    def test_eval_model_hyperparameters(self):
        """
        Test the eval_model_hyperparameters function
        """
        # Create hyperparameter dictionary list
        hyper = [
            {"n_clusters":2},
            {"n_clusters":3},
            {"n_clusters":4},
            {"n_clusters":5},
            {"n_clusters":6},
            {"n_clusters":7},
            {"n_clusters":8},
            {"n_clusters":9},
        ]
        model = sklearn.cluster.KMeans
        n_clusters = tilseg.model_selection.eval_model_hyperparameters(
            self.cluster_data,
            model,
            hyper)["n_clusters"]
        self.assertAlmostEqual(n_clusters,3)

