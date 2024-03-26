"""
Unittests for seg module
"""

# KMeans and TILs do not conform to snake-case naming:
# pylint: disable=invalid-name


# Need to check types of inputs and outputs which have protected class type
# e.g. sklearn models:
# pylint: disable=protected-access

import os
import unittest
import numpy as np
import sklearn.linear_model
import sklearn.cluster
import sklearn.utils.validation
from sklearn.exceptions import NotFittedError
from PIL import UnidentifiedImageError
from PIL import Image
import sys

# Local imports: add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tilseg import seg, refine_kmeans

TEST_PATCH_PATH = os.path.join(os.path.dirname(__file__),
                               'test_patches', 'patches',
                               'test_small_patch.tif')
FAIL_TEST_PATCH_PATH = os.path.join(os.path.dirname(__file__),
                                    'test_patches', 'test_img.txt')
TEST_IN_DIR_PATH = os.path.join(os.path.dirname(__file__),
                                'test_patches', 'patches')
SUPERPATCH_PATH = os.path.join(os.path.dirname(__file__),
                               'test_patches', 'test_superpatch.tif')

files = []
for file in os.listdir(TEST_IN_DIR_PATH):
    files.append(file[:-4])

# Will use this fitted model in a unittest
model_super = refine_kmeans.KMeans_superpatch_fit(
    patch_path=SUPERPATCH_PATH,
    hyperparameter_dict={'n_clusters': 4})
# unfitted model
model_unfitted = sklearn.cluster.KMeans(n_clusters=4, max_iter=20, n_init=3,
                                        tol=1e-3)
# not a sklearn.cluster._kmeans.KMeans model
model_linear = sklearn.linear_model.LinearRegression()
model_linear.fit(np.random.rand(5).reshape(-1, 1), np.random.rand(5))


class TestSeg(unittest.TestCase):
    """
    Unittests for functions within seg.py
    """

    # def test_KMeans_superpatch_fit(self):
    #     """
    #     Unittests for KMeans_superpatch_fit function
    #     """

    #     # one-shot test with correct inputs
    #     model = refine_kmeans.KMeans_superpatch_fit(
    #         patch_path=TEST_PATCH_PATH,
    #         hyperparameter_dict={'n_clusters': 4})

    #     # checks that the model outputted above is of the correct type
    #     self.assertTrue(isinstance(model, sklearn.cluster._kmeans.KMeans))

    #     # checks that the model outputted above is fitted
    #     self.assertTrue(
    #         sklearn.utils.validation.check_is_fitted(model) is None)

    #     # tests that non-string input for patch_path is dealt with
    #     with self.assertRaises(TypeError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=2,
    #             hyperparameter_dict={'n_clusters': 4})

    #     # tests when input file does not exist
    #     with self.assertRaises(FileNotFoundError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=TEST_PATCH_PATH+'blahblah',
    #             hyperparameter_dict={'n_clusters': 4})

    #     # tests when input file is not an image
    #     with self.assertRaises(UnidentifiedImageError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=FAIL_TEST_PATCH_PATH,
    #             hyperparameter_dict={'n_clusters': 4})

    #     # tests when hyperparameter_dict is not a dictionary
    #     with self.assertRaises(TypeError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=TEST_PATCH_PATH,
    #             hyperparameter_dict=4)

    #     # tests when hyperparameter_dict does not have the expected keys
    #     with self.assertRaises(KeyError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=TEST_PATCH_PATH,
    #             hyperparameter_dict={'n_clusters': 4, 'tol': 0.001})
    #     with self.assertRaises(KeyError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=TEST_PATCH_PATH,
    #             hyperparameter_dict={'n_flusters': 4})

    #     # tests when n_clusters is not an integer less than 9
    #     with self.assertRaises(ValueError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=TEST_PATCH_PATH,
    #             hyperparameter_dict={'n_clusters': 'four'})
    #     with self.assertRaises(ValueError):
    #         model = seg.KMeans_superpatch_fit(
    #             patch_path=TEST_PATCH_PATH,
    #             hyperparameter_dict={'n_clusters': 9})

    def test_clustering_score(self):
        """
        Unittests for clustering_score function
        """

        # smoke test when algorithm is KMeans and fitting on the same patch as
        # predicting
        _ = seg.clustering_score(
            patch_path=TEST_PATCH_PATH,
            hyperparameter_dict={'n_clusters': 4},
            algorithm='KMeans')

        # Smoke test when algorithm is DBSCAN and fitting on the same patch as
        # predicting
        _ = seg.clustering_score(
            patch_path=TEST_PATCH_PATH,
            hyperparameter_dict={'eps': 0.03578947, 'min_samples': 5},
            algorithm='DBSCAN')

        # Smoke test when algorithm is OPTICS and fitting on the same patch as
        # predicting
        _ = seg.clustering_score(
            patch_path=TEST_PATCH_PATH,
            hyperparameter_dict={"min_samples": 5, "max_eps": np.inf},
            algorithm='OPTICS')

        # Smoke test when algorithm is BIRCH and fitting on the same patch as
        # predicting
        _ = seg.clustering_score(
            patch_path=TEST_PATCH_PATH,
            hyperparameter_dict={"threshold": 0.1,
                                 "branching_factor": 10,
                                 "n_clusters": 3},
            algorithm='BIRCH')

        # Smoke test when algorithm is KMeans and using a prefitted model
        s_true, ch_true, db_true = seg.clustering_score(
            patch_path=TEST_PATCH_PATH,
            hyperparameter_dict=None,
            algorithm='KMeans',
            model=model_super,
            gen_s_score=True,
            gen_ch_score=True,
            gen_db_score=True)

        # Smoke test when algorithm is KMeans and using a prefitted model but
        # all boolean inputs are false
        s_false, ch_false, db_false = seg.clustering_score(
            patch_path=TEST_PATCH_PATH,
            hyperparameter_dict=None,
            algorithm='KMeans',
            model=model_super,
            gen_s_score=False,
            gen_ch_score=False,
            gen_db_score=False)

        # checking the output type when boolean inputs are true and false
        self.assertTrue(isinstance(s_true, float) and
                        isinstance(ch_true, float) and
                        isinstance(db_true, float))
        self.assertTrue(s_false is None and
                        ch_false is None and
                        db_false is None)

        # tests that non-string input for patch_path is dealt with
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=2,
                hyperparameter_dict={'n_clusters': 4})

        # tests when input file does not exist
        with self.assertRaises(FileNotFoundError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH+'blahblah',
                hyperparameter_dict={'n_clusters': 4})

        # tests when input file is not an image
        with self.assertRaises(UnidentifiedImageError):
            _ = seg.clustering_score(
                patch_path=FAIL_TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4})

        # tests when hyperparameter_dict is not a dictionary
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict=4)

        # tests that hyperparameters_dict is not None if no model is input
        with self.assertRaises(ValueError):
            _ = seg.clustering_score(patch_path=TEST_PATCH_PATH)

        # tests when algorithm is not a string
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4},
                algorithm=5)

        # tests when algorithm is not one of the accepted strings
        with self.assertRaises(ValueError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4},
                algorithm='kmeans')

        # tests when model is not an sklearn model
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                model=5)

        # tests when model is not fitted
        with self.assertRaises(NotFittedError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                model=model_unfitted)

        # tests when model is not a sklearn.cluster._kmeans.KMeans model
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                model=model_linear)

        # tests when gen_s_score is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4},
                gen_s_score=5)

        # tests when gen_ch_score is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4},
                gen_ch_score='True')

        # tests when gen_db_score is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4},
                gen_db_score=6)

        # checks that the right hyperparameters are entered for KMeans
        with self.assertRaises(KeyError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"eps": 0.07333333},
                algorithm='KMeans')

        # checks that the right hyperparameters are entered for DBSCAN
        with self.assertRaises(KeyError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 4},
                algorithm='DBSCAN')

        # checks that the right hyperparameters are entered for BIRCH
        with self.assertRaises(KeyError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"eps": 0.07333333},
                algorithm='BIRCH')

        # checks that the right hyperparameters are entered for OPTICS
        with self.assertRaises(KeyError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"eps": 0.07333333},
                algorithm='OPTICS')

        # tests if n_clusters greater than 9 for KMeans
        with self.assertRaises(ValueError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 9},
                algorithm='KMeans')

        # tests if n_clusters is not an integer
        with self.assertRaises(ValueError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': '4'},
                algorithm='KMeans')

        # tests if eps is not an integer or float for DBSCAN
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'eps': '0.03578947',
                                     'min_samples': 5},
                algorithm='DBSCAN')

        # tests if min_samples is not an integer or float for OPTICS
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"min_samples": '5',
                                     "max_eps": np.inf},
                algorithm='OPTICS')

        # tests if max_eps is not an integer, float or np.inf for OPTICS
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"min_samples": 5, "max_eps": 'inf'},
                algorithm='OPTICS')

        # tests if threshold is not an integer or float for BIRCH
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"threshold": '0.1',
                                     "branching_factor": 10,
                                     "n_clusters": 3},
                algorithm='BIRCH')

        # tests if branching_factor is not an integer for BIRCH
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"threshold": '0.1',
                                     "branching_factor": 10.0,
                                     "n_clusters": 3},
                algorithm='BIRCH')

        # tests if n_clusters is not an integer or None for BIRCH
        with self.assertRaises(TypeError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={"threshold": '0.1',
                                     "branching_factor": 10,
                                     "n_clusters": '3'},
                algorithm='BIRCH')

        # tests if both hyperparameters and model is specified
        # If a model is input then hyperparameters should not be input
        # since the fitting has been performed prior
        with self.assertRaises(ValueError):
            _ = seg.clustering_score(
                patch_path=TEST_PATCH_PATH,
                hyperparameter_dict={'n_clusters': 3},
                algorithm='KMeans',
                model=model_super)

    def test_segment_TILs(self):
        """
        Unittests for segment_TILs function
        """

        # tests that non-string input for in_dir_path is dealt with
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=5,
                hyperparameter_dict={'n_clusters': 4})

        # tests if save_TILs_overlay is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4},
                save_TILs_overlay='True')

        # tests if save_cluster_masks is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4},
                save_cluster_masks=5)

        # tests if save_cluster_overlays is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4},
                save_cluster_overlays=6.00)

        # tests if save_csv is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4},
                save_csv='False')

        # tests if save_all_clusters_img is not a boolean
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4},
                save_all_clusters_img='True')

        # tests if out_dir_path is necessary if one of the boolean inputs is
        # true
        with self.assertRaises(ValueError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4},
                save_all_clusters_img=True)

        # tests if out_dir_path is not a string
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                out_dir_path=5,
                hyperparameter_dict={'n_clusters': 4},
                save_all_clusters_img=True)

        # tests if out_dir_path actually exists
        with self.assertRaises(NotADirectoryError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                out_dir_path='gaga',
                hyperparameter_dict={'n_clusters': 4},
                save_all_clusters_img=True)

        # tests when algorithm is not KMeans when a fitted model is input
        # this model should have been fit using KMeans clustering with
        # KMeans_superpatch_fit function
        with self.assertRaises(ValueError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'eps': 0.03578947, 'min_samples': 5},
                algorithm='DBSCAN',
                model=model_super)

        # tests when model is not an sklearn model
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                        model=5)

        # tests when model is not fitted
        with self.assertRaises(NotFittedError):
            _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                        model=model_unfitted)

        # tests when model is not a sklearn.cluster._kmeans.KMeans model
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                        model=model_linear)

        # tests if both hyperparameters and model is specified
        # If a model is input then hyperparameters should not be input since
        # the fitting has been performed prior
        with self.assertRaises(ValueError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"n_clusters": 3},
                model=model_super)

        # tests that hyperparameters_dict is not None if no model is input
        with self.assertRaises(ValueError):
            _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH)

        # tests when hyperparameter_dict is not a dictionary
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                        hyperparameter_dict=4)

        # tests when algorithm is not a string
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"n_clusters": 3},
                algorithm=5)

        # tests when algorithm is not one of the given choices
        with self.assertRaises(ValueError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"n_clusters": 3},
                algorithm='kmeans')

        # checks that the right hyperparameters are entered for KMeans
        with self.assertRaises(KeyError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"eps": 0.07333333},
                algorithm='KMeans')

        # checks that the right hyperparameters are entered for DBSCAN
        with self.assertRaises(KeyError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4},
                algorithm='DBSCAN')

        # checks that the right hyperparameters are entered for BIRCH
        with self.assertRaises(KeyError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"eps": 0.07333333},
                algorithm='BIRCH')

        # checks that the right hyperparameters are entered for OPTICS
        with self.assertRaises(KeyError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"eps": 0.07333333},
                algorithm='OPTICS')

        # checks that the right hyperparameters are entered for OPTICS
        with self.assertRaises(KeyError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"eps": 0.07333333},
                algorithm='OPTICS')

        # tests when n_clusters is not an integer for KMeans
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 4.0},
                algorithm='KMeans')

        # tests when n_clusters is greated than 8 for KMeans
        with self.assertRaises(ValueError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'n_clusters': 9},
                algorithm='KMeans')

        # tests if eps is not an integer or float for DBSCAN
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={'eps': '0.03578947', 'min_samples': 5},
                algorithm='DBSCAN')

        # tests if min_samples is not an integer or float for OPTICS
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"min_samples": '5', "max_eps": np.inf},
                algorithm='OPTICS')

        # tests if max_eps is not an integer, float or np.inf for OPTICS
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"min_samples": 5, "max_eps": 'inf'},
                algorithm='OPTICS')

        # tests if threshold is not an integer or float for BIRCH
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"threshold": '0.1',
                                     "branching_factor": 10,
                                     "n_clusters": 3},
                algorithm='BIRCH')

        # tests if branching_factor is not an integer for BIRCH
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"threshold": '0.1',
                                     "branching_factor": 10.0,
                                     "n_clusters": 3},
                algorithm='BIRCH')

        # tests if n_clusters is not an integer or None for BIRCH
        with self.assertRaises(TypeError):
            _ = seg.segment_TILs(
                in_dir_path=TEST_IN_DIR_PATH,
                hyperparameter_dict={"threshold": '0.1',
                                     "branching_factor": 10,
                                     "n_clusters": '3'},
                algorithm='BIRCH')

        # smoke test when using a prefitted model all boolean inputs are false
        tcd,_,_,_ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                      model=model_super)
        # checks that the output is a dictionary
        self.assertTrue(isinstance(tcd, dict))
        # checks that keys are the filenames
        self.assertEqual(set(list(tcd.keys())), set(files))
        # checks that the first value is an integer
        self.assertTrue(isinstance(tcd[files[0]], int))
        # checks that there is one TIL count for each patch in the directory
        self.assertEqual(len(files), len(tcd.keys()))

        # smoke testing all four clustering algorithms by fitting on the same
        # patch with boolean inputs set to False
        _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                    hyperparameter_dict={'n_clusters': 4},
                                    algorithm='KMeans')
        _ = seg.segment_TILs(
            in_dir_path=TEST_IN_DIR_PATH,
            hyperparameter_dict={"threshold": 0.1,
                                 "branching_factor": 10,
                                 "n_clusters": 3},
            algorithm='BIRCH')
        _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                    hyperparameter_dict={'eps': 0.03578947,
                                                         'min_samples': 5},
                                    algorithm='DBSCAN')
        _ = seg.segment_TILs(in_dir_path=TEST_IN_DIR_PATH,
                                    hyperparameter_dict={'min_samples': 22,
                                                         'max_eps': np.inf},
                                    algorithm='OPTICS')
    