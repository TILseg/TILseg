TILseg README File

Last Updated: March 15th, 2023

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



### 1. ABOUT: ###
- - - -
TILseg (Tumor-Infiltrating Lymphocyte segmentation) is a software created to segment different types of cells capatured on a whole slide image of breast tissue. The tissue is stained using hematoxylin and eosin (H&E), then the resulting images are often used by pathologists to diagnose breast cancer. Tumor-infiltrating lymphocytes (TILs) are often found in high concentrations in breast cancer tissue. Therefore, reliable identification of cell types, and their locations is imperative for accurate diagnoses. This software aims to complement the diangosis pipeline by automating the segmentation and quantification of these cells from whole slide images. Approaches, like TILseg, are carving out the interface between computational tools and traditional histological, pathological, and medicinal approaches. 

This software provides a straightforward method for analyzing H&E stained microscope images and classifying and quantifying TILs present in the stained image. Briefly, the software takes in a given number of slide images, divides the image into a set of smaller patches, and filters out patches that do not contain significant amounts of tissue. A superpatch consisting of several smaller patches, from multiple slide images, is then passed into the machine learning model. Hyper-parameters optimization with various clustering algorithms can be performed by the software. Once training and validation via scoring of clustering is complete, the model is used to identify different cell types and segment and enumerate identified TILs. Detailed images and comma-separated value files with quantifiable details are created for each patch containing tissue.

This software is broken into three different modules that users can call: `preprocessing.py`, `model_selection.py`, and `seg.py`. The modules are intended to be used sequentially and their main functions/use cases are outlined in sections below.

### 2. METHODOLOGY: ###
- - - -
![TILseg Workflow](https://user-images.githubusercontent.com/121736782/225461396-af97addf-fbb9-4423-830f-f8e489d03983.png)

### 3. INSTALLATION: ###
- - - -
#### Dependencies: ####
- [matplotlib](https://matplotlib.org) = 3.6.2
- [numpy](https://numpy.org/) = 1.22.3
- [opencv](https://opencv.org/) = 4.6.0
- [openslide](https://openslide.org/) = 3.4.1
- [openslide-python](https://openslide.org/api/python/) = 1.1.2
- [pandas](https://pandas.pydata.org/) = 1.5.2
- [pillow](https://pillow.readthedocs.io/en/stable/) = 9.3.0
- [python](https://www.python.org/) = 3.10.9
- [scikit-image](https://scikit-image.org/) = 0.19.3 
- [scikit-learn](https://scikit-learn.org/stable/) = 1.0.2
- [scipy](https://scipy.org/) = 1.7.3  
These dependencies can be most easily handled using the provided environment.yml file to create a conda virtual environment. To do this:  
1. Install [Anaconda](https://www.anaconda.com/). 
2. Clone this repository
    - For example by running the command `git clone git@github.com:TILseg/TILseg.git` 
3. Creating and running virtual environment:
    - For Windows and Linux:
        - From the TILseg directory run `conda env create -f environment.yml`
        - The environment can then be activated using `conda activate tilseg`
    - For macOS:
        - From the TILseg directory run `conda env create -f environment_mac.yml`
        - The environment can then be activated using `conda activate tilseg_mac`
4. Add TILseg to the PYTHONPATH environment variable
    - To update the environment variable run `export PYTHONPATH = "path/to/TILseg:$PYTHONPATH"` on the command line
        - To update this environment variable more permanately this command can be added to the .bashrc file on linux, or the .profile file on MacOS
    - Alternatively, in a python file or at the REPL prior to importing tilseg, run `import sys`, then `sys.path.append("path/to/TILseg")`  

Note: Installing OpenSlide can be difficult for some users. The following command prompts were found helpful by these authors during installation. The
utility of these prompts is extremely case dependent. The authors heavily encourage users to exercise discretion when considering what may be helpful for them.

Running these may be useful for Windows WSL and Linux:
1. `pip install openslide-python --no-cache-dir`
2. `sudo atp-get install gcc`
3. `RUN apt-get update && apt-get install ffmpeg libsm6 libtext6 -y`
4. `sudo apt install libgl1-mesa-glx`   


For macOS:
- Follow instructions at [Openslide Python](https://openslide.org/download/)

If there is an issue with the libffi.so.* when trying to import openslide, this can be addressed by creating a symbolic link between libffi.so.6 and the version installed by conda in the environment. To do this, run the command `ln -s path/to/anaconda3/envs/tilseg/lib/libffi.so.7 path/to/anaconda3/envs/tilseg/lib/libffi.so.7`. This is a temporary issue until the openslide conda dependencies are updated.

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
Please reference the `tilseg_use.ipynb` file for an example of how this software may be used. Additional information and explanations can be found in `tilseg_example.ipynb`. The three primary modules are implemented with a sample slide image for illustrative purposes.

### 6. REFERENCES: ###
- - - -
This work is a revision and an extension of a [previous project](https://github.com/ViditShah98/Digital_segmentation_BRCA_547_Capstone) that originated from the CHEM E 545/546 course at the University of Washington.

Both the previous project and this work is a continuation of the research performed in [Mittal Research Lab](https://shachimittal.com/) at the University of Washington.

### 7. REPO STRUCTURE ###
- - - -
```
TILseg
-----
tilseg/
|-__init__.py
|-preprocessing.py
|-model_selection.py
|-cluster_processing.py
|-seg.py
|-test/
||-__init__.py
||-test_preprocessing.py
||-test_model_selection.py
||-test_cluster_processing.py
||-test_seg.py
||-test_birch_hyperparameters.json
||-test_patches/
|||-test_img.txt
|||-test_patch.tif
|||-test_superpatch.tif
|||-dummy.csv
|||-dummy.svs
|||-patches/
||||-test_small_patch.tif
||||-test_small_patch_2.tif
docs/
|-COMPONENTS.md
|-USECASES.md
|-USERSTORIES.md
examples/
|-tilseg_example.ipynb
|-tilseg_use.ipynb
scripts/
|-input_hyperparameters
||-birch_hyperparameters.json
||-dbscan_hyperparameters.json
||-kmeans_hyperparameters.json
||-optics_hyperparameters.json
|-output
||-birch_hyperparameters.json
||-dbscan_hyperparameters.json
||-kmeans_hyperparameters.json
||-optics_hyperparameters.json
|-opt_cluster_hyperparameters.py
environment.yml
environment_mac.yml
LICENSE
README.md
.gitignore
```
### 8. PREPROCESSING: ###
- - - -
The preprocessing module is called using one function: `preprocess()`. The function has six arguments, only one of which is required. The high level functionality is as follows:
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

The numpy array of the superpatch is ultimately returned which is then fed into the model selection and segmentation process that occurs in the following modules.

### 9. MODEL SELECTION: ###
- - - -
The `model_selection` module contains functions for optimizing hyperparameters, and comparing various clustering algorithms. This module can either be interacted with as a python library, or as a command line interface. To use as a command line interface, from within the conda environment run `python path/to/opt_cluster_hyperparameters.py <args>`. For using this module as a python library the main functions from this module are the opt_\<cluster algorithm\> functions. The high level functionality of this family of functions is:  
- **Input:** The patch or superpatch, along with lists of the       hyperparameter values to test. 
- **Output:** Dictionary of hyperparameter:value pairs of the optimized hyperparameters. 

Specifically, the arguments to these functions are: 
- **data:** A numpy array, of shape 3xP, where the columns are the RGB vaues of the image, and the rows represent the individual pixels.
- **\<hyperparameter_lists\>:** These arguments vary from function to function, depending on the hyperparameters for a specific clustering algorithm. For example, with opt_dbscan this takes the form of `eps`, and `min_samples`, lists of values to try for the `eps` and `min_samples` parameters for DBSCAN respectively. The hyperparameters for a model are taken as groups based on the order in the list. So, if opt_dbscan is given eps=[0.01, 0.1, 1] and min_samples=[5, 10,20], the hyperparameters for the models would be {eps=0.01, min_samples=5}, {eps=0.1, min_samples=10}, and {eps=1, min_samples=20}. The generation of these lists can be helped with the `generate_hyperparameter_combinations` function.  
- **metric:** Which scoring method to use for evaluating the clustering from each set of hyperparameters.  This is a string describing the method, could be:
    - Silhouette Score
    - Davies Bouldin Score
    - Calinski Harabasz Index  
*note: For KMeans, the [elbow](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) method based on inertia plots will be used*
- **verbose:** Whether a verbose output is desired. When True, the function will print the hyperparameters currently being tried, and a list of the clusters that are generated from them to the standard output.
- **kwargs:** Keyword arguments which are passed to the metric class.

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
There are three main functions in the `seg` module: `KMeans_superpatch_fit`, `clustering_score`, and `segment_TILs`.


`KMeans_superpatch_fit` fits a KMeans clustering model to a patch that will be used to cluster other patches. KMeans is the only clustering algorithms that allows fitting a model to one patch clustering on another. All other clustering algorithms need to be fitted on the same patch that needs to be clustered. It makes sense to use this function to fit a KMeans clustering model to a superpatch that can capture H&E stain variation across patients and technologies.
- **Input:** The superpatch, KMeans hyperparameters.
- **Output:** Fitted model. 

Specifically, the arguments to these functions are: 
- **patch_path:** String containing the directory path to the patch that the model will be fitted to obtain cluster decision boundaries
- **hyperparameter_dict:** Dicitonary of hyperparameters for KMeans containing 'n_clusters' as the only key this dictionary can be obtained by reading the JSON file outputted by `tilseg.module_selection`



`clustering_score` scores clustering models that have been fitted and predicted on a patch, The motive of this function is to test out clustering algorithms on a single patch. The goal of this function is NOT to get high throughput scores from multiple patches in a whole slide image.
- **Input:** The patch, hyperparameters(optional), algorithm, fitted model (optional), output scoring metrics.
- **Output:** Clustering scores. 

Specifically, the arguments to these functions are: 
- **patch_path:** String containing directory path to the patch that will be fitted and/or clustered on to produce cluster labels that will be used for the scoring
- **hyperparameter_dict:** Dicitonary of hyperparameters for the chosen algorithm. This dictionary can be read by the JSON file outputted by `tilseg.model_selection` module.
    - for KMeans: dictionary should have 'n_clusters' key
    - for DBSCAN: dictionary should have 'eps' and 'min_samples' keys
    - for OPTICS: dictionary should have 'min_samples' and 'max_eps' keys
    - for BIRCH: dictionary should have 'threshold', 'branching_factor', and 'n_clusters' keys
- **algorithm:**  String containing the clustering algorithm to be used: 'KMeans', 'DBSCAN', 'OPTICS', or 'BIRCH'
- **model:** sklearn KMeans model fitted on a superpatch
- **gen_s_score:** Boolean to generate Silhouette score
- **gen_ch_score:** Boolean to generate Calinski-Harabasz index
- **gen_db_score:** Boolean to generate Davies-Bouldin score



`segment_TILs` applies a clustering model to patches and generates multiple files outputs and TILs counts.: TILs
    overlayed on the original H&E patch, binary segmentation masks of each
    cluster, individual clusters overlayed on the original patch, image of all
    the clusters, and a CSV file containing countour information of each TIL
    segmented from the patch
- **Input:** Directory containing patches, output directory (optional), hyperparameters(optional), algorithm, fitted model (optional).
- **Output:** : TIL counts from each patch, TILs overlayed on the original H&E patch, binary segmentation masks of each cluster, individual clusters overlayed on the original patch, image of all the clusters, and a CSV file containing countour information of each TIL segmented from the patch 

Specifically, the arguments to these functions are: 
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
