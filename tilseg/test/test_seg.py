"""
Unittests for seg module
"""

import os
import re

import unittest
import tilseg.seg
import tilseg.cluster_processing
import sklearn.linear_model
import numpy as np
import sklearn.cluster
import sklearn.utils.validation

from PIL import UnidentifiedImageError
from sklearn.exceptions import NotFittedError

print(os.getcwd())

# current_dir = re.findall(f"{os.sep}[a-zA-Z0-9]+$",os.getcwd())[0][1:]
# if current_dir == "tilseg":
#     test_patch_path = os.path.join("..","abi_patches","test", "test_patch.tif")
#     fail_test_patch_path = os.path.join("..","abi_patches","test", "test_img.txt")
#     small_test_patch_path = os.path.join("..","abi_patches","test", "test_small_patch.tif")
# elif current_dir == "TILseg":
#     test_patch_path = os.path.join("abi_patches","test", "test_patch.tif")
#     fail_test_patch_path = os.path.join("abi_patches","test", "test_img.txt")
#     small_test_patch_path = os.path.join("..","abi_patches","test", "test_small_patch.tif")
# elif current_dir == "test":
#     test_patch_path = os.path.join("..","..","abi_patches","test", "test_patch.tif")
#     fail_test_patch_path = os.path.join("..","..","abi_patches","test", "test_img.txt")
#     small_test_patch_path = os.path.join("..","abi_patches","test", "test_small_patch.tif")

test_patch_path = '/Users/abishek/Desktop/DataScienceClasses/TILseg/abi_patches/test/test_small_patch.tif'
fail_test_patch_path = '/Users/abishek/Desktop/DataScienceClasses/TILseg/abi_patches/test/test_img.txt'
test_in_dir_path = '/Users/abishek/Desktop/DataScienceClasses/TILseg/abi_patches/test/patches'

# Will use this fitted model in a unittest
model_super = tilseg.seg.KMeans_superpatch_fit(
    patch_path=test_patch_path,
    hyperparameter_dict={'n_clusters': 4})
# unfitted model
model_unfitted = sklearn.cluster.KMeans(n_clusters=4,max_iter=20, n_init=3, tol=1e-3)
# not a sklearn.cluster._kmeans.KMeans model
model_linear = sklearn.linear_model.LinearRegression()
model_linear.fit(np.random.rand(5).reshape(-1,1), np.random.rand(5))

class TestSeg(unittest.TestCase):
    """
    Unittests for functions within seg.py
    """


    def test_KMeans_superpatch_fit(self):
        """
        Unittests for KMeans_superpatch_fit function
        """
       
       # smoke test with correct inputs
        try:
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=test_patch_path,
                hyperparameter_dict={'n_clusters': 4})
        except:
            self.assertTrue(False)

        # checks that the model outputted above is of the correct type
        self.assertTrue(type(model) == sklearn.cluster._kmeans.KMeans)

        # checks that the model outputted above is fitted
        self.assertTrue(sklearn.utils.validation.check_is_fitted(model) == None)

        # tests that non-string input for patch_path is dealt with
        with self.assertRaises(TypeError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=2,
                hyperparameter_dict={'n_clusters': 4})
        
        # tests when input file does not exist
        with self.assertRaises(FileNotFoundError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=test_patch_path+'blahblah',
                hyperparameter_dict={'n_clusters': 4})
            
        # tests when input file is not an image
        with self.assertRaises(UnidentifiedImageError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=fail_test_patch_path,
                hyperparameter_dict={'n_clusters': 4})
            
        # tests when hyperparameter_dict is not a dictionary
        with self.assertRaises(TypeError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=test_patch_path,
                hyperparameter_dict=4)
            
        # tests when hyperparameter_dict does not have the expected keys
        with self.assertRaises(KeyError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=test_patch_path,
                hyperparameter_dict={'n_clusters': 4, 'tol': 0.001})
        with self.assertRaises(KeyError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=test_patch_path,
                hyperparameter_dict={'n_flusters': 4})
            
        # tests when n_clusters is not an integer less than 9
        with self.assertRaises(ValueError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=test_patch_path,
                hyperparameter_dict={'n_clusters': 'four'})
        with self.assertRaises(ValueError):
            model = tilseg.seg.KMeans_superpatch_fit(
                patch_path=test_patch_path,
                hyperparameter_dict={'n_clusters': 9})
            

    def test_clustering_score(self):
        """
        Unittests for clustering_score function
        """

        # Smoke test when algorithm is KMeans and fitting on the same patch as predicting
        try:
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 4},
                     algorithm='KMeans')
        except:
            self.assertTrue(False)

        # Smoke test when algorithm is DBSCAN and fitting on the same patch as predicting
        try:
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                            hyperparameter_dict={"eps": 0.07333333},
                            algorithm='DBSCAN')
        except:
            self.assertTrue(False)

        # Smoke test when algorithm is OPTICS and fitting on the same patch as predicting
        try:
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                            hyperparameter_dict={"min_samples": 5, "max_eps": np.inf},
                            algorithm='OPTICS')
        except:
            self.assertTrue(False)

        # Smoke test when algorithm is BIRCH and fitting on the same patch as predicting
        try:
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                            hyperparameter_dict={"threshold": 0.1, "branching_factor": 10, "n_clusters": 3},
                            algorithm='BIRCH')
        except:
            self.assertTrue(False)

        # Smoke test when algorithm is KMeans and using a prefitted model
        try:
            s_true, ch_true, db_true = tilseg.seg.clustering_score(patch_path=test_patch_path,
                            hyperparameter_dict=None,
                            algorithm='KMeans',
                            model=model_super, 
                            gen_s_score=True,
                            gen_ch_score=True,
                            gen_db_score=True)
        except:
            self.assertTrue(False)
        
        # Smoke test when algorithm is KMeans and using a prefitted model but all boolean inputs are false
        try:
            s_false, ch_false, db_false = tilseg.seg.clustering_score(patch_path=test_patch_path,
                            hyperparameter_dict=None,
                            algorithm='KMeans',
                            model=model_super, 
                            gen_s_score=False,
                            gen_ch_score=False,
                            gen_db_score=False)
        except:
            self.assertTrue(False)

        # checking the output type when boolean inputs are true and false
        self.assertTrue(isinstance(s_true, float) and isinstance(ch_true, float) and isinstance(db_true, float))
        self.assertTrue(s_false==None and ch_false==None and db_false==None)

        # tests that non-string input for patch_path is dealt with
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=2,
                     hyperparameter_dict={'n_clusters': 4})
        
        # tests when input file does not exist
        with self.assertRaises(FileNotFoundError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path+'blahblah',
                     hyperparameter_dict={'n_clusters': 4})
            
        # tests when input file is not an image
        with self.assertRaises(UnidentifiedImageError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=fail_test_patch_path,
                     hyperparameter_dict={'n_clusters': 4})

        # tests when hyperparameter_dict is not a dictionary
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict=4)
            
        # tests that hyperparameters_dict is not None if no model is input
        with self.assertRaises(ValueError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path)

        # tests when algorithm is not a string
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 4},
                     algorithm=5)
            
        # tests when algorithm is not one of the accepted strings
        with self.assertRaises(ValueError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 4},
                     algorithm='kmeans')
            
        # tests when model is not an sklearn model
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     model=5)

        # tests when model is not fitted
        with self.assertRaises(NotFittedError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     model=model_unfitted)
            
        # tests when model is not a sklearn.cluster._kmeans.KMeans model
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     model=model_linear)
            
        # tests when gen_s_score is not a boolean
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 4},
                     gen_s_score=5)
            
        # tests when gen_ch_score is not a boolean
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 4},
                     gen_ch_score='True')
            
        # tests when gen_db_score is not a boolean
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 4},
                     gen_db_score=6)
            
        # checks that the right hyperparameters are entered for KMeans
        with self.assertRaises(KeyError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"eps": 0.07333333},
                     algorithm='KMeans')
            
        # checks that the right hyperparameters are entered for DBSCAN
        with self.assertRaises(KeyError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 4},
                     algorithm='DBSCAN')
            
        # checks that the right hyperparameters are entered for BIRCH
        with self.assertRaises(KeyError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"eps": 0.07333333},
                     algorithm='BIRCH')
            
        # checks that the right hyperparameters are entered for OPTICS
        with self.assertRaises(KeyError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"eps": 0.07333333},
                     algorithm='OPTICS')
            
        # tests if n_clusters greater than 9 for KMeans
        with self.assertRaises(ValueError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': 9},
                     algorithm='KMeans')

        # tests if n_clusters is not an integer
        with self.assertRaises(ValueError):      
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'n_clusters': '4'},
                     algorithm='KMeans')
            
        # tests if eps is not an integer or float for DBSCAN
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={'eps': '0.75'},
                     algorithm='DBSCAN')
            
        # tests if min_samples is not an integer or float for OPTICS
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"min_samples": '5', "max_eps": np.inf},
                     algorithm='OPTICS')
            
        # tests if max_eps is not an integer, float or np.inf for OPTICS
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"min_samples": 5, "max_eps": 'inf'},
                     algorithm='OPTICS')
            
        # tests if threshold is not an integer or float for BIRCH
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"threshold": '0.1', "branching_factor": 10, "n_clusters": 3},
                     algorithm='BIRCH')
            
        # tests if branching_factor is not an integer for BIRCH
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"threshold": '0.1', "branching_factor": 10.0, "n_clusters": 3},
                     algorithm='BIRCH')
            
        # tests if n_clusters is not an integer or None for BIRCH
        with self.assertRaises(TypeError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                     hyperparameter_dict={"threshold": '0.1', "branching_factor": 10, "n_clusters": '3'},
                     algorithm='BIRCH')

        # tests if both hyperparameters and model is specified
        # If a model is input then hyperparameters should not be input since the fitting has been performed prior
        with self.assertRaises(ValueError):
            s, ch, db = tilseg.seg.clustering_score(patch_path=test_patch_path,
                            hyperparameter_dict={'n_clusters': 3},
                            algorithm='KMeans',
                            model=model_super)

            
    def test_segment_TILs(self):
        """
        Unittests for segment_TILs function
        """

        # tests that non-string input for in_dir_path is dealt with
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=5,
                                          hyperparameter_dict={'n_clusters': 4})

        # tests that in_dir_path actually exists
        with self.assertRaises(NotADirectoryError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path='gaga',
                                          hyperparameter_dict={'n_clusters': 4})

        # tests if save_TILs_overlay is not a boolean
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_TILs_overlay='True') 
            
        # tests if save_cluster_masks is not a boolean
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_cluster_masks=5) 
            
        # tests if save_cluster_overlays is not a boolean
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_cluster_overlays=6.00) 
            
        # tests if save_csv is not a boolean
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_csv='False') 
            
        # tests if save_all_clusters_img is not a boolean
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_all_clusters_img='True') 
            
        # tests if out_dir_path is necessary if one of the boolean inputs is true
        with self.assertRaises(ValueError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_all_clusters_img=True)

        # tests if out_dir_path is not a string
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          out_dir_path=5,
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_all_clusters_img=True)  
            
        # tests if out_dir_path actually exists
        with self.assertRaises(NotADirectoryError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          out_dir_path='gaga',
                                          hyperparameter_dict={'n_clusters': 4},
                                          save_all_clusters_img=True)  
            
        # tests when algorithm is not KMeans when a fitted model is input
        # this model should have been fit using KMeans clustering with KMeans_superpatch_fit function
        with self.assertRaises(ValueError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"eps": 0.07333333},
                                          algorithm='DBSCAN',
                                          model=model_super)  

        # tests when model is not an sklearn model
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          model=5)
            
        # tests when model is not fitted
        with self.assertRaises(NotFittedError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          model=model_unfitted)
            
        # tests when model is not a sklearn.cluster._kmeans.KMeans model
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          model=model_linear)
            
        # tests if both hyperparameters and model is specified
        # If a model is input then hyperparameters should not be input since the fitting has been performed prior
        with self.assertRaises(ValueError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"n_clusters": 3},
                                          model=model_super)
            
       # tests that hyperparameters_dict is not None if no model is input
        with self.assertRaises(ValueError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path)

        # tests when hyperparameter_dict is not a dictionary
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict=4)
            
        # tests when algorithm is not a string
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"n_clusters": 3},
                                          algorithm=5)
            
        # tests when algorithm is not one of the given choices
        with self.assertRaises(ValueError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"n_clusters": 3},
                                          algorithm='kmeans')

        # checks that the right hyperparameters are entered for KMeans
        with self.assertRaises(KeyError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"eps": 0.07333333},
                                          algorithm='KMeans')

        # checks that the right hyperparameters are entered for DBSCAN
        with self.assertRaises(KeyError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4},
                                          algorithm='DBSCAN')
            
        # checks that the right hyperparameters are entered for BIRCH
        with self.assertRaises(KeyError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"eps": 0.07333333},
                                          algorithm='BIRCH')
            
        # checks that the right hyperparameters are entered for OPTICS
        with self.assertRaises(KeyError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"eps": 0.07333333},
                                          algorithm='OPTICS')
            
        # checks that the right hyperparameters are entered for OPTICS
        with self.assertRaises(KeyError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"eps": 0.07333333},
                                          algorithm='OPTICS')
            
        # tests when n_clusters is not an integer for KMeans
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 4.0},
                                          algorithm='KMeans')
            
        # tests when n_clusters is greated than 8 for KMeans
        with self.assertRaises(ValueError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'n_clusters': 9},
                                          algorithm='KMeans')
            
        # tests if eps is not an integer or float for DBSCAN
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={'eps': '0.75'},
                                          algorithm='DBSCAN')
            
        # tests if min_samples is not an integer or float for OPTICS
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"min_samples": '5', "max_eps": np.inf},
                                          algorithm='OPTICS')
            
        # tests if max_eps is not an integer, float or np.inf for OPTICS
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"min_samples": 5, "max_eps": 'inf'},
                                          algorithm='OPTICS')
            
        # tests if threshold is not an integer or float for BIRCH
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"threshold": '0.1', "branching_factor": 10, "n_clusters": 3},
                                          algorithm='BIRCH')
            
        # tests if branching_factor is not an integer for BIRCH
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"threshold": '0.1', "branching_factor": 10.0, "n_clusters": 3},
                                          algorithm='BIRCH')
            
        # tests if n_clusters is not an integer or None for BIRCH
        with self.assertRaises(TypeError):
            Tcd = tilseg.seg.segment_TILs(in_dir_path=test_in_dir_path,
                                          hyperparameter_dict={"threshold": '0.1', "branching_factor": 10, "n_clusters": '3'},
                                          algorithm='BIRCH')