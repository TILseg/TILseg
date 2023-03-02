"""
Contains functions for selecting clustering algorithms and their hyperparameters
"""
# System imports
from collections.abc import Sequence
from typing import Callable

# External imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.base
import sklearn.metrics
import scipy.stats


def find_elbow(data: np.array, r2_cutoff: float = 0.9) -> int:
    """
    Function to find the elbow of an inertia plot
    Parameters
    ----------
    data: Cluster and inertia data, first column is number of clusters,
        second column is the intertia (pr other metric)
    r2_cutoff: Cutoff for r2 score for the elbow, when the reamining data 
        fits a linear regression line with 
    Returns
    -------
    n_clusters: Ideal number of clusters based on elbow method
    """
    n_clusters = None
    for i in range(len(data)):
        # Create an array with data beyond the current elbow
        remaining = data[i:]
        # Find rvalue to asses how lienar this data is
        rvalue = scipy.stats.linregress(remaining[:, 0],
                                        remaining[:, 1]).rvalue
        if rvalue**2 > r2_cutoff:
            n_clusters = data[i, 0]
            break
    # This else is called only if the for loop fails to break
    else:
        raise ValueError("Unable to find elbow")
    return int(n_clusters)


def eval_km_elbow(data: np.array,
                   n_clusters_list: Sequence[int] = range(10),
                   r2_cutoff=0.9, 
                   **kwargs) -> int:
    """
    Function to find ideal number of clusters for knn using inertia plots and elbow method
    Parameters
    ----------
    data: Data to cluster
    n_clusters_list: List of the number of clusters to try
    r2_cutoff: passed to find_elbow function
    **kwargs: Keyword arguments passed to skleanr.cluster.KMeans
    Returns
    -------
    n_clusters: The number of clusters identified by the elbow method
    """
    inertia = np.zeros((len(n_clusters_list), 2))
    row = 0
    for n_clusters in n_clusters_list:
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, **kwargs)
        kmeans.fit(data)
        inertia[row, 0] = n_clusters
        inertia[row, 1] = kmeans.inertia_
        row += 1
    return find_elbow(inertia, r2_cutoff=r2_cutoff)


def eval_model_hyperparameters(data: np.array,
                               model: sklearn.base.ClusterMixin,
                               hyperparameters: Sequence[dict],
                               metric: Callable = sklearn.metrics.silhouette_score,
                               metric_direction: str = "max",
                               full_return: bool = False,
                               **kwargs):
    """
    Function to find hyperparameters based on silhouette coefficient
    Parameters
    ----------
    data: Data to cluster
    model: Cluster class (not object or instance) for which to evaluate the hyperparameters
    hyperparameters: Sequence (list) of dictionaries containing hyperparameters to test
    metric: Metric to use for scoring 
    metric_direction: Determines whether greater or smaller scores are better
    full_return: whether to return all the scores, or just the best parameters 
    **kwargs: Keyword arguments passed to sklearn metric function
    Returns
    -------
    hyperparameter:dict
     Dictionary with the hyperparameters position in the provided list
    or
    scores:dict
      Dictionary mapping the hyperparameters position to the silhouette coefficient
    """
    # Create a dictionary to hold the scores
    scores = {}
    # Iterate through the parameters
    for count,parameters in enumerate(hyperparameters):
        clusterer = model(**parameters)
        clusterer.fit(data)
        scores[count] = metric(data, clusterer.predict(data), **kwargs)
        if np.isnan(scores[count]):
            raise ValueError(f"Couldn't find Score for {parameters}")
    if full_return:
        return scores
    if metric_direction in ["max", "maximum", "greater", "->", ">", "right", "higher"]:
        max_val = np.NINF
        max_val_parameters = None
        for key, value in scores.items():
            if value > max_val:
                max_val = value
                max_val_parameters = key
        return hyperparameters[max_val_parameters]
    if metric_direction in ["min", "minimum", "less", "<-", "<", "left", "lower"]:
        min_val = np.Inf
        min_val_parameters = None
        for key, value in scores.items():
            if value < min_val:
                min_val = value
                min_val_parameters = key
        return hyperparameters[min_val_parameters]


def eval_models(data: np.array,
                models: Sequence[sklearn.base.ClusterMixin],
                hyperparameters: Sequence[dict],
                metric: Callable = sklearn.metrics.silhouette_score,
                metric_direction: str = "max",
                full_return: bool = False,
                **kwargs):
    """
    Function to evaluate how well different models 
    Parameters
    ----------
    data: np.array containing data for clustering
    models: Sequence (list) of models to evaluate (should be class, not object or instance)
    hyperparameters: Sequence of hyperparameters (dicts) to create the models to compare
    metric: Metric to evaluate the clustering methods with (function)
    metric_direction: Determines whether greater or smaller scores are better
    full_return: Whether to return dictionary of models: scores, or just best scoring model
    **kwargs: Keyword arguments passed to metric function
    Returns
    -------
    model: sklearn.base.ClusterMixin
        model which clusters the data beest according to the metric
    or
    model_dictionary: dict
        dictionary mapping models to scores
    """
    # Dicionary mapping model to score
    model_scores = {}
    if len(models) != len(hyperparameters):
        raise ValueError("Models and Hyperparameter list lengths must match")
    # Iterate through the models and find the scores
    for i, model in enumerate(models, start=0):
        # Create the clusterer object
        clusterer = model(**hyperparameters[i])
        # fit the clusterer to the data
        clusterer.fit(data)
        model_scores[clusterer] = metric(
            data, clusterer.predict(data), **kwargs)
    if full_return:
        return model_scores
    if metric_direction in ["max", "maximum", "greater", "->", ">", "right", "higher"]:
        max_val = np.NINF
        max_val_model = None
        for key, value in model_scores.items():
            if value > max_val:
                max_val = value
                max_val_model = key
        return max_val_model
    if metric_direction in ["min", "minimum", "less", "<-", "<", "left", "lower"]:
        min_val = np.Inf
        min_val_model = None
        for key, value in model_scores.items():
            if value < min_val:
                min_val = value
                min_val_model = key
        return min_val_model
    raise ValueError("Invalid Metric Direction Specification")


def eval_models_dict(data: np.array,
                     model_parameter_dict: dict,
                     metric: Callable = sklearn.metrics.silhouette_score,
                     metric_direction: str = "max",
                     full_return: bool = False,
                     **kwargs):
    """
    Convinience method to take a model:hyperparameter dictionary, and call eval_models
    Parameters
    ----------
    data: np.array containing data for clustering
    model_parameter_dict: dict of model:parameters
    metric: Metric to evaluate the clustering methods with (function)
    metric_direction: Determines whether greater or smaller scores are better
    full_return: Whether to return dictionary of models: scores, or just best scoring model
    **kwargs: Keyword arguments passed to metric function
    Returns
    -------
    model: sklearn.base.ClusterMixin
        model which clusters the data beest according to the metric
    or
    model_dictionary: dict
        dictionary mapping models to scores
    """
    models = []
    hyperparameters = []
    for key, value in model_parameter_dict.items():
        models += [key]
        hyperparameters += [value]
    return eval_models(data,
                       models=models,
                       hyperparameters=hyperparameters,
                       metric=metric,
                       metric_direction=metric_direction,
                       full_return=full_return,
                       **kwargs)

def eval_models_silhouette_score(data:np.array,
                                 models:Sequence[sklearn.base.ClusterMixin],
                                 hyperparameters:Sequence[dict],
                                 full_return:bool=False,
                                 **kwargs):
    """
    Wrapper function for eval_models with silhouette_score
    Parameters
    ----------
    data: np.array containing data for clustering
    models: Sequence (list) of models to evaluate (should be class, not object or instance)
    hyperparameters: Sequence of hyperparameters (dicts) to create the models to compare
    full_return: Whether to return dictionary of models: scores, or just best scoring model
    **kwargs: Keyword arguments passed to metric function
    Returns
    -------
    model: sklearn.base.ClusterMixin
        model which clusters the data beest according to Silhouette Score
    or
    model_dictionary: dict
        dictionary mapping models to Silhouette Scores
    """
    return eval_models(data,
                       models,
                       hyperparameters,
                       metric=sklearn.metrics.silhouette_score,
                       metric_direction="max",
                       full_return=full_return,
                       **kwargs)

def eval_models_calinski_harabasz(data:np.array,
                                 models:Sequence[sklearn.base.ClusterMixin],
                                 hyperparameters:Sequence[dict],
                                 full_return:bool=False,
                                 **kwargs):
    """
    Wrapper function for eval_models with Calinski Harabasz Index
    Parameters
    ----------
    data: np.array containing data for clustering
    models: Sequence (list) of models to evaluate (should be class, not object or instance)
    hyperparameters: Sequence of hyperparameters (dicts) to create the models to compare
    full_return: Whether to return dictionary of models: scores, or just best scoring model
    **kwargs: Keyword arguments passed to metric function
    Returns
    -------
    model: sklearn.base.ClusterMixin
        model which clusters the data beest according to Calinski Harabasz index
    or
    model_dictionary: dict
        dictionary mapping models to Calinski Harabasz index
    """
    return eval_models(data,
                       models,
                       hyperparameters,
                       metric=sklearn.metrics.calinski_harabasz_score,
                       metric_direction="max",
                       full_return=full_return,
                       **kwargs)

def eval_models_davies_bouldin(data:np.array,
                                 models:Sequence[sklearn.base.ClusterMixin],
                                 hyperparameters:Sequence[dict],
                                 full_return:bool=False,
                                 **kwargs):
    """
    Wrapper function for eval_models with Davies Bouldin Index
    Parameters
    ----------
    data: np.array containing data for clustering
    models: Sequence (list) of models to evaluate (should be class, not object or instance)
    hyperparameters: Sequence of hyperparameters (dicts) to create the models to compare
    full_return: Whether to return dictionary of models: scores, or just best scoring model
    **kwargs: Keyword arguments passed to metric function
    Returns
    -------
    model: sklearn.base.ClusterMixin
        model which clusters the data beest according to Davies Bouldin Index
    or
    model_dictionary: dict
        dictionary mapping models to Davies Bouldin Index
    """
    return eval_models(data,
                       models,
                       hyperparameters,
                       metric=sklearn.metrics.davies_bouldin_score,
                       metric_direction="min",
                       full_return=full_return,
                       **kwargs)

def plot_inertia(data:np.array,
                 n_clusters_list: Sequence[int],
                 file_path,
                 mark_elbow:bool=False,
                 r2_cutoff:float=0.9,
                 **kwargs
                 ):
    """
    Plots the inertia for each of the n_clusters in n_clusters_list
    Parameters
    ----------
    data: np.array containing data to cluster
    n_clusters_list: List of n_clusters to create the inertial plot for 
    file_path: path of where to save the image of the plot, either string or pathlike object
    Returns
    -------
    matplotlib plot object
    """
    inertia = np.zeros((len(n_clusters_list), 2))
    for row, n_clusters in enumerate(n_clusters_list):
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, **kwargs)
        kmeans.fit(data)
        inertia[row,0] = n_clusters
        inertia[row,1] = kmeans.inertia_
    fig = plt.figure()
    axes = plt.axes()
    axes.plot(inertia[:,0], inertia[:,1], "o-b", label="inertia")
    axes.set_xlabel("Number of Clusters")
    axes.set_ylabel("Inertia")
    axes.set_title("Inertia Plot")
    if not mark_elbow:
        plt.savefig(file_path)
        return fig
    elbow_n_cluster = find_elbow(inertia, r2_cutoff=r2_cutoff)
    elbow_inertia = inertia[np.where(inertia==elbow_n_cluster)[0][0],1]
    axes.scatter(elbow_n_cluster,elbow_inertia, "X-r", markersize = 12)
    plt.savefig(file_path)
    return fig