## PREPROCESSING
##### What it does: 
Takes the image file of the whole slide image (WSI) and separates it into smaller patches which are then sampled and used for creating the superpatch.
Inputs: WSIs, usually .svs or .ndpi


**Subcomponents**
##### Patch Creation
Divides the whole slide image(s) (WSI) into smaller sections called “patches”
1. open_slide: opens a slide and returns OpenSlide object and slide’s dimensions
      - Input:
      - Output:




## MODEL SELECTION
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


## IMAGE SEGMENTATION
##### What it does:
This section include functions that either train a kmeans model on a superpatch or test a ground-truth prediction on a single patch. For the first case, segmentation will be run on a folder of patches, where a kmeans model is fit to a superpatch and then predicted on this folder of patches. For the second case, a kmeans model is fit and predicted on a single patch. This provides our self-defined "validation test". With information from both cases, we can compare the results of the segmentated images to see how "well" our kmeans model is behaving. This should be done after using the tilseg.model_selection module to select and optimize hyperparameters of a model.

**Subcomponents**
1. Superpatch Fit and Scoring
	- kmeans_superpatch_fit: Fits a KMeans clustering model to a patch that will be used to cluster other patches
		- Inputs:
			- patch_path (str) : the directory path to the patch that the model will be fitted to obtain cluster decision boundaries
			- hyperparameter_dict (dict): dicitonary of hyperparameters for KMeans containing 'n_clusters' as the only key
		- Outputs:
			- model (sklearn.base.ClusterMixin): the fitted model

	- clustering_score: Scores clustering models after they have been fit and predicted to an individual patch. The goal of this function is NOT to get high throughput scores from multiple patches in a whole slide image
		- Inputs:
			- patch_path (str) : the directory path to the patch that will be fitted and/or clustered on to produce cluster labels that will be used for the scoring
			- Hyperparameter_dict (dict) : Optimized hyperparameters for algorithm. This dictionary can be read by the JSON file outputted by tilseg.model_selection module for KMeans: dictionary should have 'n_clusters' key for DBSCAN: dictionary should have 'eps' and 'min_samples' keys for OPTICS: dictionary should have 'min_samples' and 'max_eps' keys for BIRCH: dictionary should have 'threshold', 'branching_factor', and 'n_clusters' keys
			- Algorithm (str): Chosen clustering algorithm: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
			- model: (sklearn.cluster._kmeans.KMeans) sklearn model fitted on a superpatch. Note: Only input a model if the algorithm is KMeans and the user wants to score a model that has been fit on a superpatch.
     			- gen_s_score (bool): generate Silhouette score
          		- Gen_ch_score (bool): generate Calinksi score
              		- Gen_db_score (bool): generate Davies-Bouldin score
		- Outputs:
			- S_score (float): Silhouette score; ranges from -1 to 1 with 0 meaning in different clusters, -1 meaning clustered assigned in a wrong fashion and 1 meaning far apart and well separated clusters
     		  	- Ch_score (float): Calinksi index; higher value of ch_score means the clusters are dense and well separated- there is no absolute cut-off value
			- Db_score (float): Davies-Bouldin score; lower values mean better clustering with zero being the minimum value

3. Apply Clustering Model
   	- segment_TILS:
   	  	- Inputs:
   	  	  	- in_dir_path (str): ground truth test (path to an individual patch) or predicting from superpatch fit (path to a directory of patches)
   	  	  	- out_dir_path (str): the directory path where output images and CSV files will be saved
   	  	  	- Hyperparameter_dict (dict): None if using kmeans_fit as the model, or input a dictionary of hyperparameters (one would also need to set the model input equal to the model you would like to use, instead of passing None)
   	  	  	- Algorithm(str): clustering algorithm to be used: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
   	  	  	- model (sklearn.cluster._kmeans.KMeans): Any input here will assume goal is to cluster all patches and the algorithm chosen is KMeans. If no model is inputted, the clustering algorithm will fit a model on the same patch that it is clustering
   	  	  	- save_TILs_overlay(bool):  generate image containing TILs overlayed on the original H&E patch
   	  	  	- save_cluster_masks(bool): generate image showing binary segmentation masks of each cluster
   	  	  	- save_cluster_overlays(bool): generate image containing individual clusters overlayed on the original patch
   	  	  	- Save_all_clusters_img (bool): generate image of all the clusters
   	  	  	- Save_csv (bool): generate CSV file containing contour information of each TIL segmented from the patch
   	  	  	- multiple_images(bool): True if the model will be fit to superpatch and predicted on sub-patches and False if model will be fit to a single patch and be predicted on this patch ("validation test")
   	  	- Outputs:
   	  	  	- TIL_count_dict(dict): contains patch filenames without the extension as the key and TIL counts in respective patches as the values
   	  	  	- kmeans_labels_dict(dict): contains patch filenames names without the extension as the key (e.g. 'position_7_8tissue') and the kmean cluster label array as the values
   	  	  	- cluster_mask_dict(dict): contains patch filenames without the extension as the key and the binary cluster mask for the cluster that had the highest contour count. This mask is a 2D array where dimensions correspond to the X and Y pixel dimensions in the original image. The mask will contain 1s in pixels associated with the cluster and 0s everywhere else.

5. Wrapper Functions
   	- Kmean_to_spatial_model_superpatch_wrapper: A wrapper used to optimize a KMeans model on a superpatch to generate binary cluster masks for each sub-patch of the slide. These masks are converted to dataframes (X pixel, Y pixel, binary mask value) and fed into a spatial algorithm (e.g Dbscan) to perform further segmentation on the highest contour count cluster returned by segment_TILS for each path.
   	  	- Inputs:
   	  	  	- superpatch_path(str): filepath to superpatch image from preprocessing step (.tif)
   	  	  	- in_dir_path(str): the directory path to the patches that will be clustered and have TILs segmented from superpatch model. This directory could be one that contains all the extracted patches containing significant amount of tissue using the tilseg.preprocessing module.
   	  	  	- spatial_hyperparameters: the spatial algorithm's optimized hyperparameters (use 'eps' = 15, 'min_samples' = 100)
   	  	  	- n_clust(list): a list of the number clusters to test in KMeans optimization
   	  	  	- out_dir(str): the directory path where output images and CSV files will be saved\
   	  	  	- save_TILS_overlay(bool): generate image containing TILs overlayed on the original H&E patch
   	  	  	- save_cluster_masks(bool): generate image showing binary segmentation masks of each cluster
   	  	  	- save_cluster_overlays(bool): generate image containing individual clusters overlayed on the original patch
   	  	  	- save_all_clusters_img(bool): generate image of all the clusters
   	  	  	- save_csv(bool): generate CSV file containing contour information of each TIL segmented from the patch

   	  	- Outputs:
   	  	  	- IM_labels (np.ndarray): labels from fitted sptail model
   	  	  	- dbscan_fit (sklearn.cluster.DBSCAN): fitted spatial model object
   	  	  	- cluster_mask_dict (dict): dictionary containg the filenames of the patches
    without the extensions as the keys and the binary cluster masks from 
    segment_TILS as the values
   	  	  	  
   	- Kmean_dbscan_patch_wrapper: A wrapper used to optimize a KMeans model on a patch to generate a binary cluster mask. This mask is converted to a dataframe (X pixel, Y pixel, binary mask value) and fed into a spatial algorithm (e.g Dbscan) to perform further segmentation on the highest contour count cluster returned by segment_TILS. This function is used to generate a ground truth image for scoring (fit KMeans model to patch and predict on same patch)
   	  	- Inputs:
   	  	  	- patch_path(str): file path to a single patch image from the preprocessing step (.tif)
   	  	  	- spatial_hyperparameters: the spatial algorithm's optimized hyperparameters (use 'eps' = 15, 'min_samples' = 100)
   	  	  	- n_clust(list): a list of the number clusters to test in KMeans optimization
   	  	  	- out_dir(str): the directory path where output images and CSV files will be saved
   	  	  	- save_TILS_overlay(bool): generate image containing TILs overlayed on the original H&E patch
   	  	  	- save_cluster_masks(bool): generate image showing binary segmentation masks of each cluster
   	  	  	- save_cluster_overlays(bool): generate image containing individual clusters overlayed on the original patch
   	  	  	- save_all_clusters_img(bool): generate image of all the clusters
   	  	  	- save_csv(bool): generate CSV file containing contour information of each TIL segmented from the patch
   	  	- Outputs:
   	  	  	- IM_labels (np.ndarray): labels from fitted sptail model
   	  	  	- dbscan_fit (sklearn.cluster.DBSCAN): fitted spatial model object
   	  	  	- cluster_mask_dict (dict): dictionary containg the filenames of the patches without the extensions as the keys and the binary cluster masks from segment_TILS as the values


### REFINED KMEANS
##### What it does:
This folder creates a superpatch from 3 class classified images, optimizes DBSCAN hyperparameters on these images, and then fits this model on a superpatch.

**Subcomponents**
1. Improved KMeans Integration
	- KMeans_superpatch_fit: documentation can be found above in Segmentation

2. Integration with DBSCAN
	- mask_to_features: Generates the spatial coordinates from a binary mask as features to cluster with DBSCAN
		- Inputs: binary_mask(np.narray): a binary mask with 1's corresponding to the pixels involved in the cluster with the most contours and 0's for pixels not
		- Outputs: features(np.array): an array where each row corresponds to a set of 
    	coordinates (x,y) of the pixels where the binary_mask had a value of 1
    
	- km_dbscan_wrapper: Generates a fitted dbscan model and labels when provided a binary mask 
    2D array for the KMeans cluster with the highest contour count. A plot of 
    the dbscan clustering results is printed to the window, with a colorbar and 
    non-color bar version saved to the "ClusteringResults" directory as 
    "dbscan_result.jpg"
		- Inputs:
			- binary_mask (np.ndarray): a binary mask with 1's corresponding to the pixels 
    involved in the cluser with the most contours and 0's for pixels not
     			- hyperparameter_dict(dict): Contains hyperparameters as keys, corresponding to optimized values	
		- Outputs:
			- all_labels (np.ndarray): labels of image after dbscan clustering for plotting
     			- dbscan (sklearn.cluster.DBSCAN): fitted dbscan model


## CLUSTER PROCESSING
##### What it does:
This python file is meant to generate human and computer readable data for
analysis based on the clusters generated via the clustering model previously
developed. The output is a series of images which represent the
original image and relevant overlays of the determined clusters. Additionally,
based on the clusters, data from filtered cell clusters will be compiled into
a CSV. Immune cell groups are identified using the contour functionality from
OpenCV. The implemented filters are based on area and roundness of the derived
contours.

##### Side Effects:
Incorrect identification of clusters

**Subcomponents**
1. Image Preparation & Generation
   	- image_series_exceptions: This function is used by generate_image_series in order to throw exceptions from receiving incorrect array types.
   	  	- Inputs:
   	  	  	- Image_array (np.ndarray): a 4 dimensional array where the dimensions are image number, X, Y, color from which RGB images are generated
   	  	  	- rgb_bool (boolean): is the image being passed in color or grayscale
   	  	- Outputs: Either a pass or a ValueError
   	  	  
   	- generate_image_series: This takes in an array of image values and generates a directory of .jpg images in the specified file location
   	  	- Inputs:
   	  	  	- image_array (np.ndarray): a 4 dimensional array where the dimensions are image number, X, Y, color from which RGB images are generated
   	  	  	- filepath (str): the filepath (relative or absolute) in which the directory of images is generated
   	  	  	- prefix (str): the name of the directory created to store the generated images
   	  	  	- rgb_bool (bool): is the image being passed in color or grayscale
   	  	- Outputs:  Directory of .jpg images in the file location specified
   	  	  
   	- gen_base_arrays: Generates three arrays as the basis of cluster assignment. The first array contains the original image and will be used to make overlaid images. The second is all zeros and will be used to generate boolean masks. The third also contains 0 but with different dimensions for use to generate an all cluster image.
   	  	- Inputs:
   	  	  	- ori_image (np.ndarray): the original image as a 3 dimensional array with dimensions of X, Y, Color
   	  	  	- num_clusts (int): number of clusters which defines length of added dimension in overlaid and masks arrays
   	  	- Outputs:
   	  	  	- final_array (np.ndarray): 4 dimensional array best thought of as a series of 3D arrays where each 3D array is the original image and the 4th dimension will correspond to cluster after value assignment
   	  	  	- binary_array (np.ndarray): 3 dimensional array where the dimensions correspond to cluster, X and Y. This will be used for generation of binary masks for each cluster.
   	  	  	- all_mask_array (np.ndarray): 3 dimensional array where the dimensions correspond to X, Y, color. This will be used to generate an image with all clusters shown.


2. Binary Mask Implementation
   	- result_image_generator: Generates 3 arrays from clusters. The first is the each cluster individually overlaid on the original image. The second is a binary mask from each cluster. The third is an array with each pixel colored based on the associated cluster.
   	  	- Inputs:
   	  	  	- img_clust (np.ndarray): a 2D array where the dimensions correspond to X and Y, and the values correspond to the cluster assigned to that pixel
   	  	  	- original_image (np.ndarray): the original image as a 3 dimensional array where dimensions correspond to X, Y, and color
   	  	- Outputs:
   	  	  	- final_arrays (np.ndarray): a 4 dimensional array where dimensions correspond to cluster, X, Y, and color. This can be thought of as a list of images with one for each cluster. The images are the original image with cluster pixels labeled black
   	  	  	- binary_arrays (np.ndarray): a 3 dimensional array where dimensions correspond to cluster, X, and Y. This can be thought of as a list of images with one for each cluster. The images will contain 1s in pixels associated with the cluster and 0s everywhere else.
   	  	  	- all_masks (np.ndarray): a 3 dimensional array where dimensions correspond to X, Y and color. The pixels in the array have various colors associated for each cluster.
   	  	  
   	- mask_only_generator: Generates 1 array from cluster. It is a binary mask from each cluster.
   	  	- Input: img_clust (np.ndarray): a 2D array where the dimensions correspond to X and Y, and the values correspond to the cluster assigned to that pixel
   	  	- Output: binary_arrays: np.ndarray a 3 dimensional array where dimensions correspond to cluster, X, and Y. This can be thought of as a list of images with one for each cluster. The images will contain 1s in pixels associated with the cluster and 0s everywhere else.
   	  
   	- filter_boolean: Determines if a given contour meets the filters that have been defined for TILs
   	  	- Input: contour (np.ndarray): an array of points corresponding to an individual contour
   	  	- Output: meets_crit (bool): boolean that is true if the contour meets the filter and false otherwise
   	  
   	- contour_generator: Creates contours based on an inputted mask and parameters defined here and in the filter_boolean function. These parameters define what will be classified as likely an immune cell cluster and can be varied within filter_bool.
   	  	- Input: img_mask (np.ndarray): binary 2D array where the dimensions represent X, and Y and values are either 0 or 1 based on if the point is contained in the cluster
   	  	- Outputs:
   	  	  	- contours_mod (list): list of arrays of points which defines all filtered contours
   	  	  	- contours_count (int): number of contours that met the determined filters

3. Image Generation:
   	- csv_results_compiler: Generates CSV file with relevant areas, perimeters, and circularities of filtered contours thought to contain TILs
   	  	- Inputs:
   	  	  	- cont_list (list): list of arrays of points corresponding to contours
   	  	  	- filepath (str): the filepath where the CSV file will be saved
   	  	- Outputs: a csv file and the associated path
   	  	  
   	- Immune_cluster_analyzer: This function will generate the contours, identify the relevant cluster
    that contains the immune cells and export the data as a CSV. It will also
    generate an image of the contours overlaid on the image.
   	  	- Input: masks (list): list of masks which are arrays of 0s and 1s corresponding to cluster location
   	  	- Outputs:
   	  	  	- TIL_contour (list): list of arrays that correspond to the contours of the filtered TILs
   	  	  	- max_contour_count (int): maximum number of contours found
   	  	  	- cluster_mask (np.ndarray): 2D array slice of 3D binary mask (num): (clusters by image x-dim by image y-dim) corresponding to cluster with most contours

   	- draw_til_images: This function will generate relevant file paths and save overlaid image and mask from the contours.
   	  	- Inputs:
   	  	  	- img (nd.ndarray): 3 dimensional array containing X, Y, and color data of the image that will be overlaid
   	  	  	- contours (list): list of arrays of points defining the contours that will be overlaid on the images
   	  	  	- filepath (str): directory where the images will be saved
   	  	- Outputs: None

   	- image_postprocessing: This is a wrapper function that will be used to group all postprocessing together. In general postprocessing will generate series of images as well as a CSV with general data derived from contour determination using OpenCV. Also prints the time taken for post-processing.
   	  	- Inputs:
   	  	  	- clusters (np.ndarray): 2D array with dimensions X, and Y and values as the cluster identified via the model
   	  	  	- ori_img (np.ndarray): 3D array with dimensions X, Y, and color with three color channels as RGB. This is the original image clustering was performed on
   	  	  	- gen_all_clusters (bool): determines if image with all clusters visualized will be generated
   	  	  	- gen_overlays (bool): determines if overlaid images will be generated
   	  	  	- gen_tils (bool): determines if overlaid and mask of TILs will be generated
   	  	  	- gen_masks (bool): determines if masks will be generated
   	  	  	- gen_csv (bool): determines if CSV of contours will be generated
   	  	- Output: til_count (int): maximum number of contours found


## SIMILARITY
##### What it does:
Takes the image file of the whole slide image (WSI) and separates it into smaller patches which are then sampled and used for creating the superpatch.
##### Inputs:
WSIs, usually .svs or .ndpi
##### Outputs:
Patches + superpatch (.tif) of the original image(s) that will be used to create the clustering model.
##### Side Effects:
Loss of data (when dividing WSI into patches)

**Subcomponents**
1. Mean Squared Error & Model Fitting
	- image_similarity: This function calculates the mean squared error and image difference between two arrays
		- Inputs:
			- mask1 (np.ndarray): array of first image
     			- mask2 (np.ndarray): array of second image
		- Outputs:
			- mse (float): mean squared error
     			- diff (np.ndarray): image difference as numpy array

	- superpatch_similarity: This iterates through a folder of superpatches and calculates the mean squared error and plots an image of the difference between the superpatch mask and reference mask.
		- Inputs:
			- superpatch_folder (str): path to folder containing superpatch files
     			- reference_patch (str): file of reference patch that model will be applied
 			- output_path (str): path to folder where images are saved
     			- reference_array (np.ndarray): reference patch array after running 
segment_TILs in similarity_use.ipynb
		- Outputs: None, but prints mean squared error and plots difference image
			








