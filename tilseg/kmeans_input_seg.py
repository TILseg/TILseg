#Imports
import numpy as np
from tilseg.model_selection import opt_mean_shift
import sklearn.cluster
# Local imports

#Mean-Shift
def mean_shift_patch_fit(data):
    data = np.array(data)
    hyperparameter_dict = opt_mean_shift(data = data,
                   bandwidth = [0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9],
                   seeds=[0.1,0.2,0.4,0.5])
    model = sklearn.cluster.MeanShift(**hyperparameter_dict, max_iter=20,
                                   n_init=3, tol=1e-3)
    model.fit_predict(data)
    cluster_labels = model.labels_
    cluster_centers = model.cluster_centers_
    return model, cluster_labels, cluster_centers