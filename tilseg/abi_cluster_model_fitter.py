import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cluster_model_fitter(patch_path, algorithm, n_clusters=None): 
    
    '''
    patch_path: path of the patch that model needs to be fitted.
    '''
    
    # Creates a variable which references our preferred parameters for KMeans clustering
    if algorithm == 'KMeans':
        model = KMeans(n_clusters, max_iter=20,
                    n_init=3, tol=1e-3)
        # Reads the patch into a numpy uint8 array    
        fit_patch = plt.imread(patch_path) 
        # Linearizes the array for R, G, and B separately and normalizes
        # The result is an N X 3 array where N=height*width of the patch in pixels
        fit_patch_n = np.float32(fit_patch.reshape((-1, 3))/255.)
        # Fits the model to our linearized and normalized patch data 
        model.fit(fit_patch_n)
    else:
        model = None

    # Outputs our specific model of the patch we want to cluster and will be used as input to pred_and_cluster function below
    return model