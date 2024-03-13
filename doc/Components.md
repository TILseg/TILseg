## PREPROCESSING
##### What it does: 
Takes the image file of the whole slide image (WSI) and separates it into smaller patches which are then sampled and used for creating the superpatch.
Inputs: WSIs, usually .svs or .ndpi


**Subcomponents**
1. Patch Creation: Divides the whole slide image(s) (WSI) into smaller sections called “patches”
   	- open_slide: opens a slide and returns OpenSlide object and slide’s dimensions
   	  	- Input: slidepath (str): the complete path to the slide file (.svs)
   	  	- Outputs:
   	  	  	- slide (openslide.OpenSlide): slide object created by OpenSlide
   	  	  	- slide_x (int): x dimension of slide
   	  	  	- slide_y (int): y dimension of slide
   	- get_tile_size: A function that takes in a slide dimension and returns the optimal breakdown of each slide into x patches.
   	  	- Inputs:
   	  	  	- maximum (int): the maximum dimension desired
   	  	  	- size (int): the size of the entire slide image
   	  	  	- cutoff (int): the maximum number of pixels to remove (default is 4)
   	  	- Outputs:
   	  	  	- dimension (int): the desired pixel size needed
   	  	  	- slices (int): the number of slices needed in the given direction
   	  	  	- remainder (int): the number of pixels lost with the slicing provided

   	- percent_of_pixels_lost: A function that calculates the total percentage of pixels lost from the whole slide when the slicing occurs.
   	  	- Inputs:
   	  	  	- lost_x (int): the number of pixels lost in the x direction
   	  	  	- patch_x (int): the number of patches that are split in the x direction
   	  	  	- lost_y (int): the number of pixels lost in the y direction
   	  	  	- patch_y (int): the number of patches that are split in the y direction
   	  	  	-  x_size (int): the total number of pixels in the x direction of the slide
   	  	  	-  y_size (int): the total number of pixels in the y direction of the slide
   	  	- Outputs:
   	  	  	- percent (float): the percent of pixels deleted, rounded to two places
   	- save_image:  A function that saves an image given a path.
   	  	- Inputs:
   	  	  	- path (str): the complete path to a directory to which the image should be saved
   	  	  	- name (str): the name of the file, with extension, to save
   	  	  	- image_array (np.array): a numpy array that stores image information
   	  	- Outputs: None
   	 
   	- create_patches: A function that creates patches and yields an numpy array that describes the image patch for each patch in the slide.
   	  	- Inputs:
   	  	  	-  slide (openslide.OpenSlide): the OpenSlide object of the entire slide
   	  	  	-  xpatch (int): the number of the patch in the x direction
   	  	  	-  ypatch (int): the number of the patch in the y direction
   	  	  	-   xdim (int): the size of the patch in the x direction
   	  	  	-   ydim (int): the size of the patch in the y direction
   	  	- Outputs:
   	  	  	- np_patches (lst(np.arrays)): a list of all patches, each as a number array
   	  	  	- patch_position (lst(np.arrays)): a list of tuples containing indices
   
   	  
2. Converting Patch RGB Values to Averaged Grey Scale Values (SKimage):
   	- get_average_color: A function that returns the average RGB color of an input image array (in this case a patch)
   	  	- Inputs: img (np.array): a numpy array containing all information about the RGB colors in a patch
   	  	- Outputs: average (np.array): a numpy array containing the RGB code for the average color of the entire patch
   	  	  
   	- get_grey: A function that calculates the greyscale value of an image given an RGB array.
   	  	- Inputs: rgb (np.array): a numpy array containing three values, one each for R, G, and B
   	  	- Outputs: grey (float): the greyscale value of an image/patch
   	 
   	- save_all_images: A function to save all the images as background or tissue.
   	  	- Inputs:
   	  	  	- df (pd.DataFrame): the dataframe that is already created containing patches,average patch color, and the greyscale value
   	  	  	- path (str): the path to which the folders and subdirectories will be made
   	  	  	- f (str): the slide .svs file name that is currently being read
   	  	- Outputs:
   	  	  	- None, but all images are saved
   	  
3. Filtering Background vs. Tissue: Filters out patches with primarily background and retains patches of interest (i.e., ones with primarily tissue) based on a cutoff between the bimodal distribution (i.e., one average should belong to background patches and one for tissue) of the grey scaled values of all the patches.
   	- compile_patch_data: A function that compiles all relevant data for all patches into a dataframe.
   	  	- Input: slide (openslide.OpenSlide): the OpenSlide object of the entire slide
   	  	  	- ypatch (int): the number of patches in the y direction
   	  	  	- xpatch (int): the number of patches in the x direction
   	  	  	- xdim (int): the size of the patch in the x direction
   	  	  	- ydim (int): the size of the patch in the y direction
   	  	- Outputs:
   	  	  	- patchdf (pd.DataFrame): a pandas dataframe containing the three following
   	  	  	  
   	- is_it_background: A function that tests if a specific image should be classified as a background image or not.
   	  	- Inputs: cutoff (int): the cutoff value for a background image
   	  	- Outputs: background (boolean): a boolean that is True if the patch should be considered background
   	  	  
   	- sort_patches: A function that starts sorting patches based on a KDE, determines a cutoff value, and calculates the finaldataframe for each image
   	  	- Inputs:
   	  	  	- df (pd.DataFrame): the dataframe that is already created containing patches, average patch color, and the greyscale value
   	  	- Outputs:
   	  	  	- df (pd.DataFrame): an updated dataframe with a background column that indicates if a patch should be considered background or not
   	  	  	  
   	- main_preprocessing: The primary function to perform all preprocessing of the data, creating patches and returning a final large dataframe with all information contained.
   	  	- Inputs:
   	  	  	- complete_path (str): the full path to the file containing all svs files that will be used for training the model or a single svs file to get an output value
   	  	  	- training (boolean): a boolean that indicates if this preprocessing is for training data or if it to only be used for the existing model
   	  	  	- save_im (boolean): a boolean that indicates if tissue images should be saved (beware this is a lot of data, at least 10GB per slide)
   	  	  	- max_tile_x (int): the maximum x dimension size, in pixels, of a slide patch (default is 4000)
   	  	  	- max_tile_y (int): the maximum y dimension size, in pixels, of a slide patch (default is 3000)
   	  	- Outputs:
   	  	  	- all_df or sorted_df (pd.DataFrame): a dataframe containing all necessary information for creating superpatches for training (all_df) or for inputting into an already generated model (sorted_df)
   	  
   	- count_images: Count images finds the number of whole slide images available in your current working directory.
   	  	- Inputs: None (path=os.getcwd())
   	  	- Outputs: img_count (int): the number of whole slide images in your directory

4. Superpatch Creation: Bins all of the patches into X bins based on the range of averaged grey scaled values across the WSI and randomly selects one patch from each bin to be incorporated into the superpatch. 
   	- patches_per_img: Patches_per_img calculates the number of patches to be extracted from each image. If there are no images in the current working directory or provided path.
   	  	- Inputs:
   	  	  	- num_patches (int): number of total patches (that make up the entire image)
   	  	  	- path -- optional (str): path in which images might be located
   	  	- Outputs:
   	  	  	- patch_img (int): number of patches to be extraced from each image
   	  
   	- get_superpatch_patches: This function finds the patches to comprise the superpatch. The patches are selected based off of distribution of average color and the source image. This way, the superpatch is not entirely made of patches from one image (unless there is only one image available).
   	  	- Inputs:
   	  	  	- patches_df (pd.DataFrame): MUST be dataframe from main_preprocessing output random_state (int): random state for during sampling (to get consistent patch list)
   	  	  	- patches (int): number of patches
   	  	  	- path=os.getcwd(): 
   	  	  	- random_state: random state for during sampling (to get consistent patch list)
   	  	- Outputs:
   	  	  	- patches_list (list): list of the patches that make up the superpatch
   	- superpatcher: Superpatcher uses the selected patches and converts the individual patches into one patch
   	  	- Inputs:
   	  	  	- patches_list (lst): MUST be output from get_superpatch_patches list of patches
   	  	  	- sp_width (int): the width of a superpatch (how many images, default 3)
   	  	  - Outputs: superpatch (np.array): np.array that contains the superpatch
   	  	    
   	- preprocess: The preprocess function that is called when running the code. Complete details are found in the README file. This only calls other functions and is used as a wrapper
   	  	- Inputs:
   	  	  	- path (str): path to the folder containing the .svs slide files patches (int): number of patches to create superpatch with
   	  	  	- training (boolean): a boolean that indicates if this preprocessing is for training data or if it to only be used for the existing model
   	  	  	- save_im (boolean): a boolean that indicates if tissue images should be saved (beware this is a lot of data, at least 10GB per slide)
   	  	  	- max_tile_x (int): the maximum x dimension size, in pixels, of a slide patch (default is 4000)
   	  	  	- max_tile_y (int): the maximum y dimension size, in pixels, of a slide patch (default is 3000)
   	  	  	- random_state (int): random state to use during sampling of patches
   	  	  - Outputs:
   	  	    	- spatch (pd.DataFrame): a dataframe containing all necessary information for creating superpatches for training (all_df) or for inputting into an already generated model (sorted_df)



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
             	- data (np.array): Contains data used for clustering
             	- Model (list): List of models that will be evaluated against one another
             	- Metric: Metric used to evaluate the hyperparameters. This can be either Sillhouette score, Calinksi Harabasz score, or Davies Bouldin score using the sklearn package.
             	- Metric_direction (str): Sets whether a small or large value is “good” in terms of score, and helps in comparing to other models
             	- Full_return (bool): Determines whether the function will return the a dictionary of the models, or a dictionary of the models mapped to their score
             	- **kwargs: Keyword arguments passed to metric function
           - Outputs:
             	- If full_return = True: model_scores(dict): Dictionary that maps models to their respective scores, If full_return = False: model (sklearn.base.ClusterMixin), model that clusters the data most effectively according to the metrics provided
             	  
      - eval_models_dict: Wrapper function to take in a dictionary of hyperparameters and call eval_models to test a given model
           - Inputs:
             	- data (np.array): Contains data used for clustering
             	- Model_parameter_dict: Dictionary mapping models to parameters
             	- metric: Callable = sklearn.metrics.silhouette_score
             	- Metric_direction (str): Sets whether a small or large value is “good” in terms of score, and helps in comparing to other models
             	- Full_return (bool): Determines whether the function will return the a dictionary of the models, or a dictionary of the models mapped to their score
             	- **kwargs: Keyword arguments passed to metric function
           - Outputs:
             	- Either the model that clusters the data best according to the metric, or a dictionary mapping models to their respective scores (using eval_model)
       
      - eval_models_silhouette_score: Wrapper function to take in a dictionary of hyperparameters and call eval_models to test a given model using Silhouette scores as metric
           - Inputs:
             	- data (np.array): Contains data used for clustering
             	- Models (list): List of models to evaluate
             	- Hyperparameters: List of dictionaries to create a comparison of models
             	- Full_return: Determines whether the function will return the a dictionary of the models, or a dictionary of the models mapped to their Silhouette score
             	- **kwargs: Keyword arguments passed to metric function
           - Outputs:  Either the model that clusters the data best according to the Silhouette Score, or a dictionary mapping models to their Silhouette scores (using eval_model)
             
      - eval_models_calinski_harabasz:
           - Inputs:
             	- data (np.array): Contains data used for clustering
             	- Models (list): List of models to evaluate
             	- Hyperparameters: List of dictionaries to create a comparison of models
             	- Full_return: Determines whether the function will return the a dictionary of the models, or a dictionary of the models mapped to their Calinski Harabasz score
             	- **kwargs: Keyword arguments passed to metric function
           - Outputs: Either the model that clusters the data best according to the Silhouette Score, or a dictionary mapping models to their Calinski Harabasz scores (using eval_model)
       
      - eval_models_davies_bouldin: Wrapper function to take in a dictionary of hyperparameters and call eval_models to test a given model using Davies Bouldin scores as metric
           - Inputs:
             	- data (np.array): Contains data used for clustering
             	- Models (list): List of models to evaluate
             	- Hyperparameters: List of dictionaries to create a comparison of models
             	- Full_return: Determines whether the function will return the a dictionary of the models, or a dictionary of the models mapped to their Davies Bouldin score
             	- **kwargs: Keyword arguments passed to metric function
           - Outputs: Either the model that clusters the data best according to the Silhouette Score, or a dictionary mapping models to their Davies Bouldin scores
 (using eval_model)
       
      - plot_inertia:
           - Inputs:
             	- data (np.array): Contains data used for clustering
             	- N_clusters Ilist): List containing the number of clusters to be used to create the inertial plot
             	- File_path (str): Filepath of location to save the plot produced
             	- Mark_elbow (bool): Typically set to FALSE
             	- R2_cutoff (float): This value is subjective. This is the point where adding more clusters will not significantly affect inertia.
             	- **kwargs: Keyword arguments passed to metric function
           - Outputs: fig: matplotlib plot
             

3. Optimizing Models:
   - opt_kmeans: Function to optimize the number of clusters used in KMeans clustering, wrapper with eval_km_elbow
        - Inputs:
          	- data (np.array): Contains pixel data used for clustering
          	- N_clsuters (list): List of clusters the user wants to test
          	- **kwargs: Keyword arguments passed to metric function
        - Outputs:
          	- Hyperparameter_dict (dict): Dictionary with one key, ‘n_cluster’. More keys could be added later to test other parameters within kmeans
          	  
   - opt_dbscan: Function to optimize the Epsilon hyperparameter used in DBscan clustering, wrapper with eval_model_hyperparameters
        - Inputs:
          	- data (np.array): Contains pixel data used for clustering
          	- eps (list): List of Epsilon (eps) values the user wants to test
          	- metric (str): String with the name of the metric user would like to evaluate model with
          	- Verbose (bool): Whether a verbose output is desired
          	- **kwargs: Keyword arguments passed to metric function
        - Outputs:
          	- Hyperparameter (dict): Dictionary with one key, ‘eps’, corresponding to optimized eps value. More keys could be added later to test other parameters within dbscan
          	  
   - opt_birch: Function to optimize hyperparameters used in Birch model, wrapper with eval_model_hyperparameters
        - Inputs:
          	- data (np.array): Contains pixel data used for clustering
          	- threshold (list): List of values to test
          	- Branching_factor (list): List of values to test
          	- n_clusters (list): List of values to test
          	- metric (str): String with the name of the metric user would like to evaluate model with
          	- Verbose (bool): Whether a verbose output is desired
          	- **kwargs: Keyword arguments passed to metric function
        - Outputs:
          	- Hyperparameter (dict): Dictionary with three keys: ‘threshold’, branching_facttor, n_clusters’. More keys could be added later to test other parameters within birch.
          
   - opt_optics: Function to optimize hyperparameters used in Optics model, wrapper with eval_model_hyperparameters
        - Inputs:
          	- data (np.array): Contains pixel data used for clustering
          	- min_samples (list): List of values to test
          	- max_eps (list): List of values to test
          	- metric (str): String with the name of the metric user would like to evaluate model with
          	- Verbose (bool): Whether a verbose output is desired
          	- **kwargs: Keyword arguments passed to metric function
        - Outputs:
          	- Hyperparameter (dict): Dictionary containing the optimized hyperparameters
     
5. Miscellaneous:
   - sample_patch: Function to sample patch and decrease time to tune parameters
        - Inputs:
          	- Data: numpy array to sample rows from
          	- Sample (int): number of rows in the returned array
        - Outputs:
          	- Sampled_array: numpy array including only rows specified 
   - generate_hyperparameter_combinations: Provides a dictionary of hyperparameters with all combinations of values
        -Inputs: Hyperparameter_dict (dict): dictionary of hyperparameter as key, corresponding to list of values
        - Outputs: Return_dict (dict): list of values with all combinations of the values of the hyperparameters
   - read_json_hyperparameters: Function that reads and store a json file that contains hyperparameters, helps with consistency and compatibility in further use of these parameters within Python
        - Inputs: file_path(str): filepath to the json file
        - Outputs: hyperparameters(dict): Contains contents of json file as hyperparameters in dictionary form



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
   	  	    	- Ch_score (float): Calinksi index; higher value of ch_score means the clusters are dense and well separated- there is no absolute cut-off value, Db_score (float): Davies-Bouldin score; lower values mean better clustering with zero being the minimum value

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
   	  	  	- cluster_index_dict: dictionary containing keys corresponding to the file name and values representing the cluster from kmeans


### REFINED KMEANS
##### What it does:
This folder creates a superpatch from 3 class classified images, optimizes DBSCAN hyperparameters on these images, and then fits this model on a superpatch.

**Subcomponents**
1. Improved KMeans Integration
   	- KMeans_superpatch_fit: Fits a KMeans clustering model to a patch that will be used to cluster other patches
   	  	- Inputs:
   	  	  	- patch_path(str): the directory path to the patch that the model will be fitted to obtain cluster decision boundaries
   	  	  	- hyperparameter_dict (dict): dicitonary of hyperparameters for KMeans containing 'n_clusters' as the only key this dictionary can be obtained by reading the JSON file outputted by tilseg.module_selection
   	  	  	- random_state (int): the random state used in model creation to get reproducible model outputs
   	  	- Outputs:
   	  	  	- model(sklearn.base.ClusterMixin): the fitted model

2. Integration with DBSCAN
	- mask_to_features: Generates the spatial coordinates from a binary mask as features to cluster with DBSCAN
		- Inputs: binary_mask(np.narray): a binary mask with 1's corresponding to the pixels involved in the cluster with the most contours and 0's for pixels not
		- Outputs: features(np.array): an array where each row corresponds to a set of 
    	coordinates (x,y) of the pixels where the binary_mask had a value of 1
    
3. Wrapper Functions
   	- km_dbscan_wrapper: Generates a fitted dbscan model and labels when provided a binary mask  2D array for the KMeans cluster with the highest contour count. A plot ofte dbscan clustering results is printed to the window, with a colorbar and  non-color bar version saved to the "ClusteringResults" directory as "dbscan_result.jpg"
   	  	- Inputs:
   	  	- binary_mask (np.ndarray): a binary mask with 1's corresponding to the pixels involved in the cluser with the most contours and 0's for pixels not
   	  	- hyperparameter_dict(dict): Contains hyperparameters as keys, corresponding to optimized values
          		- print_flag(bool) = True for printing saved plot of dbscan model
   	  	  Outputs:
			- all_labels (np.ndarray): labels of image after dbscan clustering for plotting
     			- dbscan (sklearn.cluster.DBSCAN): fitted dbscan model
   
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
   	  	  	- random_state(int): random state to specify repeatable kmeans model
   	  	- Outputs:
   	  	  	- IM_labels (np.ndarray): labels from fitted sptail model
   	  	  	- dbscan_fit (sklearn.cluster.DBSCAN): fitted spatial model object
   	  	  	- cluster_mask_dict (dict): dictionary containg the filenames of the patches without the extensions as the keys and the binary cluster masks from segment_TILS as the values
   	  	  	- cluster_index_dict (dict): cluster labels from kemans that had the highest contour count in each image. The keys are the filenames and the values are the cluster numbers.

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
      	  	  	- random_state(int): random state to specify repeatable kmeans model
   	  	- Outputs:
   	  	  	- IM_labels (np.ndarray): labels from fitted sptail model
   	  	  	- dbscan_fit (sklearn.cluster.DBSCAN): fitted spatial model object
   	  	  	- cluster_mask_dict (dict): dictionary containg the filenames of the patches without the extensions as the keys and the binary cluster masks from segment_TILS as the values
      	  	  	- cluster_index (int): cluster label from kemans that had the highest contour count. This is the cluster label that was fed into the spatial model for further classification.



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
   	  	  
   	- immune_cluster_analyzer: This function will generate the contours, identify the relevant cluster
    that contains the immune cells and export the data as a CSV. It will also
    generate an image of the contours overlaid on the image.
   	  	- Input: masks (list): list of masks which are arrays of 0s and 1s corresponding to cluster location
   	  	- Outputs:
   	  	  	- TIL_contour (list): list of arrays that correspond to the contours of the filtered TILs
   	  	  	- max_contour_count (int): maximum number of contours found
   	  	  	- cluster_mask (np.ndarray): 2D array slice of 3D binary mask (num): (clusters by image x-dim by image y-dim) corresponding to cluster with most contours
   	  	  	- count_index: cluster index with highest contour count

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
   	  	- Output:
   	  	  	- til_count (int): maximum number of contours found
   	  	  	- cluster_mask (np.ndarray): a binary cluster mask for the cluster that had the highest contour count. It is a 2D array where dimensions correspond to the X and Y pixel dimensions in the original image. The mask will contain 1s in pixels associated with the cluster and 0s everywhere else.
   	  	  	- cluster_index: cluster label that has the highest contour count


## SIMILARITY
##### What it does:
Takes the image file of the whole slide image (WSI) and separates it into smaller patches which are then sampled and used for creating the superpatch.
##### Inputs:
WSIs, usually .svs or .ndpi
##### Outputs:
Patches + superpatch (.tif) of the original image(s) that will be used to create the clustering model.
##### Side Effects:
Loss of data (when dividing WSI into patches)

##### Note:
We did not complete this section. This will be a major interest in further investigations.

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
   	  	  	- reference_array (np.ndarray): reference patch array after running segment_TILs in similarity_use.ipynb
   	  	- Outputs: None, but prints mean squared error and plots difference image
			








