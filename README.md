TILseg README File

Last Updated: March 13th, 2024

## SINCE PREVIOUS UPDATE (March 15th, 2023) ##

### DEVELOPMENTS ###
* _K-Means to DBSCAN_: Previously, K-Means was found to be the best clustering algorithm among DBSCAN, Birch, and Optics. We have built a pipeline in `refine_kmeans.py` that enables the user to optimize, fit, and apply K-Means to patches of a whole slide image and then uses the output cluster masks to feed into DBSCAN for further clustering. 
    * Please see our additions in section *11. K-MEANS TO DBSCAN (REFINE_KMEANS)* for more details. 
* _SuperPatch Scoring_: A module called `similarity.py` was created to generate a similarity score using the MSE between a K-Means model that had been fitted and applied to the same patch (i.e. 'ground truth') and a pre-fitted superpatch model(s) that is applied to the same reference patch. However, it should be noted that it has not yet been integrated to work with the current functions as `preprocessing.py` was updated before the development of `similarity.py`, so the `similarity.py` module has not been updated and tested to verify it's functionality with the current modules. 
    * Please see *Future Directions* at the bottom of this README for more details.
* _Updated Environment_: the user would've had to install OpenSlide separately from the environment.yml file which led to dependency issues and version conflicts since the version of OpenSlide initially used was not specified. We have created a new environment file that includes OpenSlide and all the compatible versions.    
* _Updated Example Jupyter Notebook_: In the `Example` folder of the repository, a new and improved example jupyter notebook `dbscan_kmeans_example.ipynb` has been added to include our latest developments and fixes. A jupyter notebook `similarity_use.ipynb` has also been added to outline how to run the the `similarity.py` module for some sample images.
* _User Cases and Components_: We have updated these documents in our `doc` folder to be more comprehensive while also reflecting our changes. The components document outlines all relevant functions with their completed associated docstrings.
* _New Archive Folder_: A new folder `Archive` has been created at the root of the repository to document and save previous code that had been written before the implementation of our update. This folder should be better utilized for future developers.
* _Bug Fixes Section_: Since this is the first update since the development of the `tilseg` package, we believe it would be useful to document our bug fixes for users and future developers to view and possibly use in their own work. 
    * Please scroll to the bottom of our README for more detailed information about our bug fixes.

## CONTENTS: ##
1. About
2. Methodology
3. Installation
4. Example
5. Example Use Case
6. References
7. Repo Structure
8. Preprocessing Module
9. Model Selection Module
10. Segmentation (Seg) Module
11. KMeans to DBSCAN Module

A. Future Directions
    i. KMeans to DBSCAN Module
    ii. Superpatch Similarity Scores
    iii. Failing Unit Tests

B. Bug Fixes
    March 13th, 2024 Update

### 1. ABOUT: ###
- - - -
TILseg (Tumor-Infiltrating Lymphocyte segmentation) is a software created to segment different types of cells captured on a whole slide image of breast tissue. The tissue is stained using hematoxylin and eosin (H&E), then the resulting images are often used by pathologists to diagnose breast cancer. Tumor-infiltrating lymphocytes (TILs) are often found in high concentrations in breast cancer tissue. Therefore, reliable identification of cell types, and their locations is imperative for accurate diagnoses. This software aims to complement the diagnosis pipeline by automating the segmentation and quantification of these cells from whole slide images. Approaches, like TILseg, are carving out the interface between computational tools and traditional histological, pathological, and medicinal approaches. 

This software provides a straightforward method for analyzing H&E stained microscope images and classifying and quantifying TILs present in the stained image. Briefly, the software takes in a given number of slide images, divides the image into a set of smaller patches, and filters out patches that do not contain significant amounts of tissue. A superpatch consisting of several smaller patches, from multiple slide images, is then passed into the machine-learning model. 

K-Means (n=4) was found to be the optimal clustering algorithm used to segment TILs. However, the model fitting is still imperfect as the TILs cluster may also include other cell types such as fibroblast cells (more fibrous and elongated), plasma (more triangular), etc. due to the similarity in their staining to TILs (more circular). The current algorithm accounts for this by filtering the TILs cluster based on area and circularity criteria that a TIL is assumed to meet. On the other hand, clustering algorithms such as DBSCAN can prove to be a more sophisticated 'filter' since it takes into account the spatial distribution of pixels that K-Means does not. Thus, it may be able to differentiate and cluster cell types based on the spatial arrangement of pixels within each cell (i.e. a cell's shape). However, the spatial algorithms previously tested in this module faced memory issues when initially implemented due to their computational complexity and large image sizes. To circumvent these memory issues, we have developed a tool that can use K-Means as a preliminary step to isolate the pixels corresponding to the assumed TILs cluster. Then, this cluster can be transformed into features that can be further clustered by DBSCAN. 

Though hyperparameter tuning has not yet been done, we hope that this tool can be utilized by future users and/or researchers in tailoring DBSCAN for their own needs.

This software is broken into four different modules that users can call: `preprocessing.py`, `model_selection.py`, `seg.py`, and `refine_kmeans`. The modules are intended to be used sequentially and their main functions/use cases are outlined in the sections below.

### 2. METHODOLOGY: ###
- - - -
![TILseg Workflow](https://user-images.githubusercontent.com/121736782/225461396-af97addf-fbb9-4423-830f-f8e489d03983.png)

### 3. INSTALLATION: ###
- - - -
#### Dependencies: ####
- [matplotlib](https://matplotlib.org) = 3.8.0
- [numpy](https://numpy.org/) = 1.26.4 
- [opencv](https://opencv.org/) = 4.9.0.80
- [openslide](https://openslide.org/) = 3.4.1
- [openslide-python](https://openslide.org/api/python/) = 1.3.1
- [pandas](https://pandas.pydata.org/) = 2.1.4
- [pillow](https://pillow.readthedocs.io/en/stable/) = 10.2.0
- [python](https://www.python.org/) = 3.11.0
- [scikit-image](https://scikit-image.org/) = 0.22.0 
- [scikit-learn](https://scikit-learn.org/stable/) = 1.4.1.post1
- [scipy](https://scipy.org/) = 1.12.0  
These dependencies can be most easily handled using the provided environment.yml file to create a conda virtual environment. To do this:  
1. Install [Anaconda](https://www.anaconda.com/). 
2. Clone this repository
    - For example by running the command `git clone git@github.com:TILseg/TILseg.git` 
3. Creating and running a virtual environment:
    - For Windows and Linux: *not yet configured*
    - For macOS:
        - From the TILseg directory run `conda env create -f environment_mac.yml`
        - The environment can then be activated using `conda activate tilseg_mac`
4. Add TILseg to the PYTHONPATH environment variable
    - To update the environment variable run `export PYTHONPATH = "path/to/TILseg:$PYTHONPATH"` on the command line
        - To update this environment variable more permanently this command can be added to the .bashrc file on linux, or the .profile file on MacOS
    - Alternatively, in a Python file or at the REPL before importing tilseg, run `import sys`, then `sys.path.append("path/to/TILseg")`  

### 4. EXAMPLE: ###
- - - -
Original H&E Patch
:-------------------------:
![Original](https://user-images.githubusercontent.com/121774063/224920422-fb696076-d907-45af-89ab-3f053dd89747.jpg)


Cluster Overlay               |  Cluster Mask
:-------------------------:|:-------------------------:
![Image3](https://user-images.githubusercontent.com/121774063/224920501-9a2b0f81-847a-4e08-8a60-e726f5e4d405.jpg)  |  ![Image3](https://user-images.githubusercontent.com/121774063/224920528-ef4b2c34-5695-46a7-b020-09dc4e068375.jpg)


All Clusters               |  Segmented TILs Overlay
:-------------------------:|:-------------------------:
![AllClusters](https://user-images.githubusercontent.com/121774063/224920465-6b5c79f6-6431-46cf-a16e-59fe66fdbc28.jpg)  |  ![ContourOverlay](https://user-images.githubusercontent.com/121774063/224920555-414d718b-6ce0-4920-9af0-01b1c6cc2b96.jpg)


### 5. EXAMPLE USE CASE: ###
- - - -
Please reference the `dbscan_kmeans_example.ipynb` file for an example of how this software may be used. Additional information and explanations can be found in `dbscan_kmeans_example.ipynb`. The three primary modules are implemented with a sample slide image for illustrative purposes.

### 6. REFERENCES: ###
- - - -
This work is a revision and an extension of a [previous project](https://github.com/ViditShah98/Digital_segmentation_BRCA_547_Capstone) that originated from the CHEME 545/546 course at the University of Washington. Updates have been made from a fork of the [TILseg/TILseg respository](https://github.com/TILseg/TILseg) that originated from the CHEME 545/546 course (Winter 2023) at the University of Washington.

Both the previous project and this work are a continuation of the research performed in [Mittal Research Lab](https://shachimittal.com/) at the University of Washington.

### 7. REPO STRUCTURE ###
- - - -
```
TILseg
-----
├── Archive
│   ├── docs
│   │   ├── COMPONENTS.md
│   │   ├── TILseg Final Presentation.pdf
│   │   ├── USECASES.md
│   │   └── USERSTORIES.md
│   ├── environment.yml
│   ├── environment_mac.yml
│   ├── examples
│   │   ├── tilseg_example.ipynb
│   │   └── tilseg_use.ipynb
│   └── scripts
│       ├── input_hyperparameters
│       │   ├── birch_hyperparameters.json
│       │   ├── dbscan_hyperparameters.json
│       │   ├── kmeans_hyperparameters.json
│       │   └── optics_hyperparameters.json
│       ├── opt_cluster_hyperparameters.py
│       └── output
│           ├── birch_hyperparameters.json
│           ├── dbscan_hyperparameters.json
│           ├── kmeans_hyperparameters.json
│           └── optics_hyperparameters.json
├── Example
│   ├── Notebook_Images
│   │   ├── Image_1.png
│   │   ├── Image_4.png
│   │   ├── Image_9.png
│   │   ├── image_10.png
│   │   ├── image_11.png
│   │   ├── image_12.png
│   │   ├── image_13.png
│   │   ├── image_14.png
│   │   ├── image_15.png
│   │   ├── image_16.png
│   │   ├── image_17.png
│   │   ├── image_18.png
│   │   ├── image_19.png
│   │   ├── image_2.png
│   │   ├── image_20.png
│   │   ├── image_21.png
│   │   ├── image_22.png
│   │   ├── image_23.png
│   │   ├── image_24.png
│   │   ├── image_3.png
│   │   ├── image_5.png
│   │   ├── image_6.png
│   │   ├── image_7.png
│   │   ├── image_8.png
│   │   └── super_before.png
│   ├── dbscan_kmeans_example.ipynb
│   └── similarity_use.ipynb
├── LICENSE
├── README.md
├── __init__.py
├── doc
│   ├── Components.md
│   ├── Use_Cases_and_Components.pdf
│   └── User_Stories&Use_Cases.md
├── environment.yml
├── test
│   ├── __init__.py
│   ├── test_cluster_processing.py
│   ├── test_model_selection.py
│   ├── test_patches
│   │   ├── dummy.csv
│   │   ├── dummy.svs
│   │   ├── patches
│   │   │   ├── test_small_patch.tif
│   │   │   └── test_small_patch_2.tif
│   │   ├── test_img.txt
│   │   ├── test_patch.tif
│   │   └── test_superpatch.tif
│   ├── test_preprocessing.py
│   ├── test_refine_kmeans.py
│   ├── test_seg.py
│   └── test_similarity.py
└── tilseg
    ├── __init__.py
    ├── cluster_processing.py
    ├── model_selection.py
    ├── preprocessing.py
    ├── refine_kmeans.py
    ├── seg.py
    └── similarity.py
```
### 8. PREPROCESSING: ###
- - - -
The preprocessing module is called using one function: `preprocess()`. The function has seven arguments, only one of which is required. The high level functionality is as follows:
- **Input**. A file path that contains all slide image files that will be processed.
- **Output**. A numpy array containing information for the superpatch created, as well as all filtered tissue images saved to their respective directories. In addition, the percentage of pixels lost due to preprocessing is also printed. This should be significantly less than one percent, but is included for the user if it is needed.

More specifically, the arguments taken for the main function are outlined below:
- **path (required):** the path to which all slide image files will be found. A subdirectory within this directory will be created for each slide, and all tissue images (after filtering) will be saved to those folders. The folders will be named the same as the slide image title. The superpatch used for training will be saved in this directory as `superpatch_training.tif`.
    - Required file type for slide images: `.svs`, `.tif`, `.ndpi`, `.vms`, `.vmu`, `.scn`, `.mrxs`, `.tiff`, `.svslide`, `.bif`

- **patches (default=6):** the number of patches to include in the superpatch. A selection of this number of small patches across all slides will be made to best represent the diversity in slide images. The superpatch will then be used for training the model downstream.

- **training (default=True):** a boolean that describes if the preprocessing step is for new slides that will not be put through training (only for model usage) or if they should be used in training, and therefore if a superpatch should be output as a result. 
    - `True` indicates that a superpatch will be created
    - `False` indicates that no superpatch will be created, but filtered patches will still be saved

- **save_im (default=True):** a boolean that describes if the preprocessing step should also save all images after filtering out background, to a subdirectory within the original directory for each slide. For either case, the superpatch will be saved as an image and the numpy array will still be returned.
    - `True` indicates that all filtered images will be saved
    - `False` indicates that all filtered images will not be saved

- **max_tile_x (default=4000):** the maximum number of pixels in the x direction for a patch. The software will attempt to get as close to this number as possible when splitting up the slide image.

- **max_tile_y (default=3000):** the maximum number of pixels in the y direction for a patch. The software will attempt to get as close to this number as possible when splitting up the slide image.

- **random_state (default=None):** the random state can be specified if the user desires to test the function, use other functions that depend on the preprocessing step, or implement it in other functions that require a superpatch to be made froem the same patches each time it is run in a jupyter notebook.

The numpy array of the superpatch is ultimately returned which is then fed into the model selection and segmentation process that occurs in the following modules.

### 9. MODEL SELECTION: ###
- - - -
The `model_selection` module contains functions for optimizing hyperparameters and comparing various clustering algorithms. This module can either be interacted with as a python library, or as a command line interface. To use as a command line interface, from within the conda environment run `python path/to/opt_cluster_hyperparameters.py <args>`. For using this module as a python library the main functions from this module are the opt_\<cluster algorithm\> functions. The high level functionality of this family of functions is:  
- **Input:** The patch or superpatch, along with lists of the       hyperparameter values to test. 
- **Output:** Dictionary of hyperparameter:value pairs of the optimized hyperparameters. 

Specifically, the arguments to these functions are: 
- **data:** A numpy array, of shape 3xP, where the columns are the RGB values of the image, and the rows represent the individual pixels.
- **\<hyperparameter_lists\>:** These arguments vary from function to function, depending on the hyperparameters for a specific clustering algorithm. For example, with opt_dbscan this takes the form of `eps`, and `min_samples`, lists of values to try for the `eps` and `min_samples` parameters for DBSCAN respectively. The hyperparameters for a model are taken as groups based on the order in the list. So, if opt_dbscan is given eps=[0.01, 0.1, 1] and min_samples=[5, 10,20], the hyperparameters for the models would be {eps=0.01, min_samples=5}, {eps=0.1, min_samples=10}, and {eps=1, min_samples=20}. The generation of these lists can be helped with the `generate_hyperparameter_combinations` function.  
- **metric:** Which scoring method to use for evaluating the clustering from each set of hyperparameters.  This is a string describing the method, could be:
    - Silhouette Score
    - Davies Bouldin Score
    - Calinski Harabasz Index  
*note: For KMeans, the [elbow](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) method based on inertia plots will be used*
- **verbose:** Whether a verbose output is desired. When True, the function will print the hyperparameters currently being tried, and a list of the clusters that are generated from them to the standard output.
- **kwargs:** Keyword arguments that are passed to the metric class.

The opt_ family of functions is the main interface to this module, but there are a variety of other methods available. These will be listed with a short description here, for details on arguments see docstrings within the module. 
- `find_elbow`: Finds the elbow of an inertia plot.
- `eval_km_elbow`: Finds the optimum number of clusters for knn using inertia plots and the [elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
- `eval_model_hyperparameters`: Generic function to find hyperparameters for a provided clustering algorithm given a list of dictionaries with the hyperparameters. This is the function which the opt_ family of functions wrap around.
- `eval_models`: Compares how well different clustering algorithms cluster the provided data.
- `eval_models_dict`: Wrapper around `eval_models`, instead of taking seperate lists of algorithms, and hyperparameters, takes a single dictionary of algorithm:hyperparameter-dictionary.
- `eval_models_silhouette_score`: Wrapper around `eval_models`, however uses Silhouette score automatically instead of taking a metric argument.
- `eval_models_calinski_harabasz`: Wrapper around `eval_models`, however uses Calinski Harabasz score automatically instead of taking a metric argument.
- `eval_models_davies_bouldin`: Wrapper around `eval_models`, however uses Davies Bouldin score automatically instead of taking a metric argument.
- `plot_inertia`: Plots the inertia for each of a given list of n_clusters.
- `sample_path`: Takes a sample of a numpy array, useful for optimizing DBSCAN, BIRCH, and OPTICS since those algorithms have very high worst case size complexity that scales with number of observations. This can also be used to speed up the process of hyperparameter optimization.
- `generate_hyperparameter_combinations`: Used to create a dictionary of hyperparameter:list-of-hyperparameter-values with all combinations of the provided hyperparameter levels.
- `read_json_hyperparameters`: Read a json file containing hyperparameters. 

### 10. SEGMENTATION (SEG): ###
- - - - 
There are two main functions in the `seg` module: `clustering_score`, and `segment_TILs`.

`clustering_score` scores clustering models that have been fitted and predicted on a patch, The motive of this function is to test out clustering algorithms on a single patch. The goal of this function is NOT to get high throughput scores from multiple patches in a whole slide image.
- **Input:** The patch, hyperparameters(optional), algorithm, fitted model (optional), output scoring metrics.
- **Output:** Clustering scores. 

Specifically, the arguments for this function are: 
- **patch_path:** String containing directory path to the patch that will be fitted and/or clustered on to produce cluster labels that will be used for the scoring
- **hyperparameter_dict:** Dictionary of hyperparameters for the chosen algorithm. This dictionary can be read by the JSON file outputted by `tilseg.model_selection` module.
    - for KMeans: dictionary should have 'n_clusters' key
    - for DBSCAN: dictionary should have 'eps' and 'min_samples' keys
    - for OPTICS: dictionary should have 'min_samples' and 'max_eps' keys
    - for BIRCH: dictionary should have 'threshold', 'branching_factor', and 'n_clusters' keys
- **algorithm:**  String containing the clustering algorithm to be used: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
- **model:** sklearn KMeans model fitted on a superpatch
- **gen_s_score:** Boolean to generate Silhouette score
- **gen_ch_score:** Boolean to generate Calinski-Harabasz index
- **gen_db_score:** Boolean to generate Davies-Bouldin score



`segment_TILs` applies a clustering model to patches and generates multiple file outputs and TILs counts.: TILs
    overlayed on the original H&E patch, binary segmentation masks of each
    cluster, individual clusters overlayed on the original patch, images of all
    the clusters, and a CSV file containing contour information of each TIL
    segmented from the patch
- **Input:** Directory containing patches, output directory (optional), hyperparameters(optional), algorithm, fitted model (optional).
- **Output:** TIL counts from each patch, TILs overlayed on the original H&E patch, binary segmentation masks of each cluster, individual clusters overlayed on the original patch, image of all the clusters, and a CSV file containing countour information of each TIL segmented from the patch 

Specifically, the arguments for this fucntion are: 
- **in_dir_path:** String containing the directory path to the patches that will clustered and have TILs segmented
- **out_dir_path:** String containing the directory path where output images and CSV files will be saved
- **hyperparameter_dict:** Dicitonary of hyperparameters for the chosen algorithm. This dictionary can be read by the JSON file outputted by `tilseg.model_selection` module.
    - for KMeans: dictionary should have 'n_clusters' key
    - for DBSCAN: dictionary should have 'eps' and 'min_samples' keys
    - for OPTICS: dictionary should have 'min_samples' and 'max_eps' keys
    - for BIRCH: dictionary should have 'threshold', 'branching_factor', and 'n_clusters' keys
- **algorithm:** String containing the clustering algorithm to be used: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
- **model:** sklearn KMeans model fitted on a superpatch
- **save_TILs_overlay:** Boolean to generate image containing TILs overlayed on the original H&E patch
- **save_cluster_masks:** Boolean to generate image showing binary segmentation masks of each cluster
- **save_cluster_overlays:**  Boolean to generate image containing individual clusters overlayed on the original patch
- **save_all_clusters_img:** Boolean to generate image of all the clusters
- **save_csv:** Boolean to generate CSV file containing countour information of each TIL segmented from the patch


### 11. K-MEANS TO DBSCAN (REFINE_KMEANS) ###
- - - - 
There are three main functions in the `refine_kmeans` module: `KMeans_superpatch_fit`, `kmean_to_spatial_model_superpatch_wrapper`, and `kmean_to_spatial_model_patch_wrapper`.

`KMeans_superpatch_fit` fits a KMeans clustering model to a patch that will be used to cluster other patches. KMeans is the only clustering algorithm that allows fitting a model to one patch clustering on another. 
- **Input:** The superpatch, KMeans hyperparameters (default is n_clusters = 4), and the random state (optional).
- **Output:** Fitted model. 

Specifically, the arguments for this function are: 
- **patch_path:** String containing the directory path to the patch that the model will be fitted to obtain cluster decision boundaries
- **hyperparameter_dict:** Dictionary of hyperparameters for KMeans containing 'n_clusters' as the only key this dictionary can be obtained by reading the JSON file outputted by `tilseg.module_selection`
- **random_state (default=None):** The random state can be specified if the user desires to test the function, use other functions that depend on the preprocessing step, or implement it in other functions that require a superpatch to be made froem the same patches each time it is run in a jupyter notebook.

`kmean_to_spatial_model_superpatch_wrapper` sequentially fits a DBSCAN clustering model to a superpatch following K-means clustering which will be used to cluster other patches. This can be used to see if DBSCAN can be used to fit a model to one patch clustering on another. However, it should be noted that DBSCAN is highly dependent on the spatial distribution of pixels. H&E stained images are largely heterogenous by nature, so this function may not prove to be useful but we have developed this tool for verification and/or future research to be done.
- **Input:** The superpatch, the folder containing the patches to be clustered, DBSCAN hyperparameters, list of KMeans hyperparameters to be optimized, the folder that will be used to save the results (optional), the masks and overlays from clustering (optional), and the random state (optional)
- **Output:** The DBSCAN model, model labels, the TILs binary mask from KMeans, and the KMeans cluster label with the highest contour count (i.e. the one identified as the TILs cluster) for each patch

Specifically, the arguments for this function are: 
- **superpatch_path (str):** file path to superpatch image from preprocessing step (.tif)
- **in_dir_path (str):** the directory path to the patches that will be clustered and have TILs segmented from the superpatch model. This directory could be one that contains all the extracted patches containing a significant amount of tissue using the tilseg.preprocessing module.
- **spatial_hyperparameters (dict):** the spatial algorithm's optimized hyperparameters
- **n_clusters (list):** a list of the number clusters to test in KMeans optimization
- **out_dir (str):** the directory path where output images and CSV files will be saved
- **save_TILs_overlay (bool):** generates images containing TILs overlayed on the original H&E patch
- **save_cluster_masks (bool):** generates images showing binary segmentation masks of each cluster
- **save_cluster_overlays (bool):** generates images containing individual clusters overlayed on the original patch
- **save_all_clusters_img (bool):** generates images of all the clusters
- **save_csv (bool):** generates a CSV file containing contour information of each TIL segmented from the patch
- **random_state: (int):** can specify repeatable kmeans model

`kmean_to_spatial_model_patch_wrapper` fits a DBSCAN clustering model to a single patch following K-means clustering which will be applied to cluster the same patch. This function has high functionality as the user is able to manually input the hyperparameters for the DBSCAN model. This allows further implementation into functions that can perform the same clustering on a folder of multiple patches or can perform optimization/scoring metrics.
- **Input:** The patch, the folder containing the patches to be clustered, DBSCAN hyperparameters, the list of KMeans hyperparameters to be optimized, the folder that will be used to save the results (optional), the masks and overlays from clustering (optional), and the random state (optional)
- **Output:** The DBSCAN model, model labels, the TILs binary mask from KMeans, and the KMeans cluster label with the highest contour count (i.e. the one identified as the TILs cluster)

Specifically, the arguments for this function are: 
- **patch_path (str):** file path to a single patch image from the preprocessing step (.tif)
- **in_dir_path (str):** the directory path to the patches that will be clustered and have TILs segmented from superpatch model. This directory could be one that contains all the extracted patches containing significant amount of tissue using the tilseg.preprocessing module.
- **spatial_hyperparameters (dict):** the spatial algorithm's optimized hyperparameters
- **n_clusters (list):** a list of the number clusters to test in KMeans optimization
- **out_dir (str):** the directory path where output images and CSV files will be saved
- **save_TILs_overlay (bool):** generates images containing TILs overlayed on the original H&E patch
- **save_cluster_masks (bool):** generates images showing binary segmentation masks of each cluster
- **save_cluster_overlays (bool):** generates images containing individual clusters overlayed on the original patch
- **save_all_clusters_img (bool):** generates images of all the clusters
- **save_csv (bool):** generates a CSV file containing contour information of each TIL segmented from the patch
- **random_state: (int):** can specify repeatable kmeans model

## A. FUTURE DIRECTIONS ##
- - -
### i. KMeans to DBSCAN Module ###
Currently, our pipeline from K-Means to DBSCAN can only be performed on a single patch using the `kmean_to_spatial_model_patch_wrapper` function. Since the DBSCAN fitting can be highly specific to each patch, we allowed the DBSCAN hyperparameters to be directly modified without optimization and output images for the user to easily observe the results to compare to the KMeans model. This can allow future users/researchers to use our function and adapt the hyperparameters as needed. Additionally, we believe that this tool can be integrated into other functions or simply modified to be performed on multiple patches. By preprocessing using KMeans to isolate our cluster of interest, we have drastically decreased the number of pixels being fed into DBSCAN and, thus, the computation time and expense as opposed to feeding in the raw, RGB patch. We hope DBSCAN can be a useful tool in further clustering the other cell types from the TILs within KMeans. If not, the function could be easily adapted to feed a cluster from KMeans into other spatial algorithms. 

### ii. Superpatch Similarity Scores ###
As noted above, we have also started constructing a pipeline for generating a 'similarity score' based on the mean square error (MSE) between two cluster masks. We have further created a tool that can implement this similarity score to compare: 1) a KMeans model that has been fitted and applied to a chosen reference patch (i.e. the ground truth) and 2) pre-fitted KMeans model(s) that have been fit on a superpatch(es) and then applied to the same chosen reference patch all in one function, `superpatch_similarity`. Since the image from 1) has a Kmeans model fit and clustered on the same image, we expect the model to do relatively well in capturing all the variance -- thus producing the ground truth. Since the superpatches in 2) are sampled across the whole slide image, these models may not capture as much variance as in 1), but we wanted to quantify and qualify how representative these superpatches are. Since the reference image is constant across all the superpatch model clustering, the scores should be somewhat comparable to one another. We also created the function so that it would output a contour overlay showcasing the ground truth (green) and the difference between the ground truth and superpatch model (red) over the original reference patch. Here are some current flaws and points to work on that we recommend:

* Mainly, this function currently is not integrated with the most updated version of `preprocessing.py`. This should be done first before anything else.
* The choice of the reference is arbitrary and subjective at the moment, but further research can be done to optimize this. Alternatively, one could generate similarity scores over a range of reference patches.
* Implement other scoring metrics or modify the function to output other visualizations.

For more information about the code and additional documentation, please see `similarity.py` in the `tilseg.py` folder. To see the preliminary use of this module, please see `similarity_use.ipynb` in the `Example` folder.

### iii. Failing Unit Tests ##
Since we had changed much of the original code in the `tilseg.py` folder, many of the unit tests that were created had failed and we were unable to resolve them at this time. Although we were able to fix the import of the `tilseg` package, we were not able to address the other errors/bugs that had appeared.

The development of our new functions in `refine_kmeans.py` is in its early stages and was built off of the original modules in `tilseg`. Thus, we encountered some errors in these unit tests as well, specifically related to the file path naming. Although we wanted to pursue test-driven development, we were not able to resolve these errors at this time, but you can access our current and working unit tests for all the `tilseg` modules in the `test folder`. Currently, `test_kmean_to_spatial_model_superpatch_wrapper` and `test_kmean_to_spatial_model_patch_wrapper` are failing and receive the following errors:

We used pytest to run our unit test files. Please not that to avoid any errors, ensure that your version of threadpoolctl is above 3.0. with the environment file we provided.

## B. BUG FIXES ##
- - -
### March 13th, 2024 Update ###
* **Hard-coded file Paths**: Initially, file paths in the example jupyter notebooks referred to paths that existed on the developer's local computer. We have updated these paths and have added test images/materials that can be accessed by any user on their computer. Note that due to the size of the test images (~100+ MB), these images were uploaded to a public google drive.
* _preprocessing.py_:
    * Changed the os handling to read in the full file paths of each .svs image since the original code was using only the filename (this led to file path exception errors)
    * **def get_superpatch_patches (def preprocess << def get_superpatch_patches)**: Updated to now have a random state argument to allow for the superpatch to be made from the same patches each time a notebook is run
    * **def sort_patches (def preprocess << def main_preprocessing << def sort_patches)**: Updated to use a Gaussian Mixture distribution to identify the peaks associated with the pink tissue and white background to reduce the background in the returned superpatches. The original method was documented very poorly and did not accurately remove white background.
* _seg.py_:
    * **def segment_TILS**: Updated to take in a `multiple_images` flag to be able to fit a KMeans model to a patch, rather than just a superpatch. This enables use of the predicted clusters on this patch in downstream scoring.
    * **def immune_cluster_analyzer (def segment_TILS << def image_postprocessing << def immune_cluster_analyzer)**: Updated to return the `cluster mask` of the highest TIL contour count to enable further segmentation using DBSCAN.
    * **def draw_til_images (def segment_TILS << def image_postprocessing << def draw_til_images)**: Had a bug for a wrong array type fed to .drawContours package that was fixed
    * **def segment_TILS**: Had a bug fixed to only check for .tif images in a patches folder (avoid errors of hidden .ipynb or files)  
* **Unit Tests**: All unit tests in the `test` folder had issues importing the `tilseg module`. The `test` folder was initially inside of `tilseg`, but was moved outside to be at the same root as `tilseg`. The module was then imported using `import ..tilseg` to access the parent directory containing both `tilseg` and `test`.
