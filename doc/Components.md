#### PREPROCESSING
##### What it does: 
Takes the image file of the whole slide image (WSI) and separates it into smaller patches which are then sampled and used for creating the superpatch.
Inputs: WSIs, usually .svs or .ndpi


**Subcomponents**
##### Patch Creation
Divides the whole slide image(s) (WSI) into smaller sections called “patches”
1. open_slide: opens a slide and returns OpenSlide object and slide’s dimensions
      - Input:
      - Output:




#### MODEL SELECTION
##### What it does:
Utilizes various functions to score and select clustering algorithms and their hyperparameters based on three metrics: Silhouette scores, Calinski Harabasz scores, and Davies Bouldin scores. Also contains optimization functions for kmeans, dbscan, birch, and optics models.

**Subcomponents**
1. Elbow method:
      - find_elbow: Find the elbow of an inertia plot to determine the optimal number of clusters
           - Inputs:
                - Data: array, column 1 contains number of clusters, column 2 contains inertia
                - R2_cutoff: Point where the r2 is cutoff for the elbow, and where the training data fits a linear regression
            - Outputs: 
                 - n_clusters (integer): Ideal number of clusters based on the elbow method
   - Eval_km_elbow: Finds the ideal number of clusters for K-Nearest Neighbors using both inertia plots and the elbow method (check types in docstrings)
      - Inputs:
           - Data (np.array): Data being used for clustering
           - N_clusters (list): List of the various number of clusters to be tested
           - R2_cutoff (int): passed to find_elbow function
           - **kwargs: Keyword arguments passed to skleanr.cluster.KMeans
	   - Outputs: 
            - n_clusters (int): Number of clusters identified by the elbow method, returns the find_elbow function with inertia and r2_cutoff inputs

2. Evaluating Model & Model Parameters:
      - eval_model_hyperparameters: Find the elbow of an inertia plot to determine the optimal number of clusters
           - Inputs:
                - data (np.array): Contains data used for clustering
                - Model: Cluster class that will be used to evaluate the hyperparameters (sklearn.base.ClusterMixin)
                - Hyperparameters (dict): List of dictionaries that containing the hyperparameters to test
                - Metric: Metric used to evaluate the hyperparameters. This can be either Sillhouette score, Calinksi Harabasz score, or Davies Bouldin score using the sklearn package.
                - Metric_direction (str): Sets whether a small or large value is “good” in terms of score, and helps in comparing to other models
                - Full_return (bool): False = return best scoring parameters, True = return scores, dictionary mapping hyperparameters to their scores
                - Verbose (bool): Whether or not a verbose is desired
                - **kwargs: Keyword arguments passed to metric function
            - Outputs: 
                 - Depends on full_return input; If full_return = True: Dictionary that maps the hyperparameters to their score based on the metric provided, If full_return = False: Dictionary with the hyperparameters 
      - eval_models:
           - Inputs:
           - Outputs:
      - eval_models_dict:
           - Inputs:
           - Outputs:
      - eval_models_silhouette_score:
           - Inputs
           - Outputs:
      - eval_models_calinski_harabasz:
           - Inputs:
           - Outputs:
      - eval_models_davies_bouldin:
           - Inputs:
           - Outputs:
      - plot_inertia:
           - Inputs:
           - Outputs:

3. Optimizing Models:
   - opt_kmeans:
        - Inputs:
        - Outputs:
   - opt_dbscan:
        - Inputs:
        - Outputs:
   - opt_birch
        - Inputs:
        - Outputs: 
   - opt_optics
        - Inputs:
        - Outputs:
     
5. Miscellaneous:
   - sample_patch:
        - Inputs:
        - Outputs:
   - generate_hyperparameter_combinations:
        -Inputs:
        - Outputs:
   - read_json_hyperparameters:
        - Inputs:
        - Outputs:


#### REFINED KMEANS
##### What it does:
**Subcomponents**



#### IMAGE SEGMENTATION
##### What it does:
Includes functions that fit and score clustering models as well as segment TILs in H&E stained patches within an image, or images. This should be done after using the tilseg.model_selection module to select and optimize hyperparameters of a model.

**Subcomponents**



#### CLUSTER PROCESSING
##### What it does:
Uses a clustering model developed in modules above to generate a variety of information from the clusters generated which can be used for subsequent analysis. The output is a series of images which represent the original image and relevant overlays of the determined clusters. Additionally, based on the clusters, data from filtered cell clusters will be compiled into a CSV. Immune cell groups are identified using the contour functionality from OpenCV. The implemented filters are based on area and roundness of the derived contours.

**Subcomponents**
