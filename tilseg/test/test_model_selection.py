"""
Unittests for model selection module
"""
# Standard Library Imports
import numbers
import os
import unittest

# External Library Imports
import matplotlib.figure
import numpy as np
import sklearn.base
import sklearn.cluster
import sklearn.datasets
import sklearn.metrics
from sklearn.utils._param_validation import InvalidParameterError

# Local imports
import tilseg.model_selection


class TestModelSelection(unittest.TestCase):
    """
    Test case for the functions within model_selection.py
    """
    cluster_data = None
    elbow_data = np.array([[1, 217.64705882352948],
                           [2, 68.42857142857143],
                           [3, 16.228571428571424],
                           [4, 12.695238095238096],
                           [5, 9.6],
                           [6, 7.166666666666666],
                           [7, 5.5],
                           [8, 4.083333333333334],
                           [9, 2.9999999999999996]])

    @classmethod
    def setUpClass(cls):
        """
        Method to create testing data
        """
        cls.cluster_data = sklearn.datasets.make_blobs(n_samples=100,
                                                       n_features=3,
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
                                                          list(range(1, 10)),
                                                          r2_cutoff=0.9,
                                                          n_init="auto")
        self.assertAlmostEqual(n_clusters, 3)
        self.assertIsInstance(n_clusters, int)

    def test_eval_model_hyperparameters(self):
        """
        Test the eval_model_hyperparameters function
        """
        # Create hyperparameter dictionary list
        hyper = [
            {"n_clusters": 2, "n_init": "auto"},
            {"n_clusters": 3, "n_init": "auto"},
            {"n_clusters": 4, "n_init": "auto"},
            {"n_clusters": 5, "n_init": "auto"},
            {"n_clusters": 6, "n_init": "auto"},
            {"n_clusters": 7, "n_init": "auto"},
            {"n_clusters": 8, "n_init": "auto"},
            {"n_clusters": 9, "n_init": "auto"},
        ]
        model = sklearn.cluster.KMeans
        # Catch fire test
        n_clusters = tilseg.model_selection.eval_model_hyperparameters(
            self.cluster_data,
            model,
            hyper)["n_clusters"]
        # Expected output based on data
        self.assertAlmostEqual(n_clusters, 3)
        # Expected output type
        self.assertIsInstance(n_clusters, numbers.Number)

    def test_eval_models(self):
        """
        Test eval_models function
        """
        models = [sklearn.cluster.KMeans,
                  sklearn.cluster.AgglomerativeClustering,
                  sklearn.cluster.AgglomerativeClustering]
        hyperparameters = [
            {"n_clusters": 3, "n_init": "auto"},
            {"n_clusters": 3, "linkage": "complete"},
            {"n_clusters": 3, "linkage": "ward"},
        ]
        metric = sklearn.metrics.silhouette_score
        metric_direction = "max"
        full_return = False
        # Catch fire test
        model = tilseg.model_selection.eval_models(
            self.cluster_data,
            models,
            hyperparameters,
            metric,
            metric_direction,
            full_return
        )
        # Make sure it is a sklearn model
        self.assertIsInstance(model, sklearn.base.ClusterMixin)

    def test_eval_models_dict(self):
        """
        Test eval_models_dict wrapper function
        """
        model_dict = {
            sklearn.cluster.KMeans: {"n_clusters": 3, "n_init": "auto"},
            sklearn.cluster.AgglomerativeClustering: {
                "n_clusters": 3, "linkage": "complete"}
        }
        # Catch fire test
        model = tilseg.model_selection.eval_models_dict(
            self.cluster_data, model_dict)
        # Check if it correctly returns a model
        self.assertIsInstance(model, sklearn.base.ClusterMixin)

    def test_eval_models_silhouette_score(self):
        """
        Test eval_models_silhouette_score function
        """
        models = [sklearn.cluster.KMeans,
                  sklearn.cluster.AgglomerativeClustering,
                  sklearn.cluster.AgglomerativeClustering]
        hyperparameters = [
            {"n_clusters": 3, "n_init": "auto"},
            {"n_clusters": 3, "linkage": "complete"},
            {"n_clusters": 3, "linkage": "ward"},
        ]
        full_return = False
        # Catch fire test
        model = tilseg.model_selection.eval_models_silhouette_score(
            self.cluster_data,
            models,
            hyperparameters,
            full_return
        )
        # Make sure it is a sklearn model
        self.assertIsInstance(model, sklearn.base.ClusterMixin)

    def test_eval_models_calinski_harabasz(self):
        """
        Test eval_models_calinski_harabasz function
        """
        models = [sklearn.cluster.KMeans,
                  sklearn.cluster.AgglomerativeClustering,
                  sklearn.cluster.AgglomerativeClustering]
        hyperparameters = [
            {"n_clusters": 3, "n_init": "auto"},
            {"n_clusters": 3, "linkage": "complete"},
            {"n_clusters": 3, "linkage": "ward"},
        ]
        full_return = False
        # Catch fire test
        model = tilseg.model_selection.eval_models_calinski_harabasz(
            self.cluster_data,
            models,
            hyperparameters,
            full_return
        )
        # Make sure it is a sklearn model
        self.assertIsInstance(model, sklearn.base.ClusterMixin)

    def test_eval_models_davies_bouldin(self):
        """
        Test eval_models_davies_bouldin function
        """
        models = [sklearn.cluster.KMeans,
                  sklearn.cluster.AgglomerativeClustering,
                  sklearn.cluster.AgglomerativeClustering]
        hyperparameters = [
            {"n_clusters": 3, "n_init": "auto"},
            {"n_clusters": 3, "linkage": "complete"},
            {"n_clusters": 3, "linkage": "ward"},
        ]
        full_return = False
        # Catch fire test
        model = tilseg.model_selection.eval_models_davies_bouldin(
            self.cluster_data,
            models,
            hyperparameters,
            full_return
        )
        # Make sure it is a sklearn model
        self.assertIsInstance(model, sklearn.base.ClusterMixin)

    def test_plot_inertia(self):
        """
        Test plot_inertia function
        """
        path = os.path.join(".", "test_plot.png")
        plot = tilseg.model_selection.plot_inertia(
            self.cluster_data,
            list(range(1, 10)),
            path,
            True,
            0.9,
            n_init='auto')
        # Check if file was succesfully crated
        self.assertTrue(os.path.isfile(path))
        # Delete file
        os.remove(path)
        # check type of plot
        self.assertIsInstance(plot, matplotlib.figure.Figure)

    def test_opt_kmeans(self):
        """
        Test opt_kmeans function
        """
        opt_kmeans_result = tilseg.model_selection.opt_kmeans(
            self.cluster_data, list(range(1, 10)), n_init="auto")
        self.assertAlmostEqual(opt_kmeans_result, 3.0)
        with self.assertRaises(InvalidParameterError):
            _ = tilseg.model_selection.opt_kmeans(
                self.cluster_data, list(range(10)), n_init="auto")

    def test_opt_dbscan(self):
        """
        test opt_dbscan function
        """
        # Catch fire test
        result = tilseg.model_selection.opt_dbscan(
            self.cluster_data, eps_list=[0.1, 0.7, 1.4, 2.1, 3.0])
        # Known value
        self.assertAlmostEqual(result['eps'], 2.1)
        with self.assertRaises(ValueError):
            _ = tilseg.model_selection.opt_dbscan(
                self.cluster_data, eps_list=[0.001, 0.02, 0.3])

    def test_opt_birch(self):
        """
        test opt_birch function
        """
        # Catch fire test
        result = tilseg.model_selection.opt_birch(
            self.cluster_data,
            threshold_list=[0.25, 0.5, 1.0],
            branching_factor_list=[10, 25, 30],
            n_clusters_list=[None, None, None],
        )
        # Known output shape
        self.assertEqual(len(result), 3)
        # Known values tests
        self.assertAlmostEqual(result["threshold"], 1.0)
        self.assertAlmostEqual(result["branching_factor"], 30)
        self.assertIsNone(result["n_clusters"])

    def test_opt_optics(self):
        """
        test opt_optics function
        """
        # Catch fire test
        result = tilseg.model_selection.opt_optics(
            self.cluster_data,
            min_samples_list=[2, 5, 10],
            max_eps_list=[3, 5, np.inf]
        )
        # Known shape of output
        self.assertEqual(len(result), 2)
        # Known output
        self.assertDictEqual(result, {"min_samples": 10, "max_eps": np.inf})


if __name__ == "__main__":
    unittest.main()
