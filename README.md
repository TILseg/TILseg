TILseg README File

Last Updated: March 12th, 2023

## CONTENTS: ##
1. About
2. Preprocessing
3. Model Selection
4. Segmentation (Seg)
5. Cluster Processing
6. Example Use Case
7. References

### 1. ABOUT: ###
- - - -
TILseg (Tumor-Infiltrating Lymphocyte segmentation) is a software created to segment different types of cells cpatured on a whole slide image of breast tissue. The tissue is stained using hematoxylin and eosin (H&E), then the resulting images are often used by pathologists to diagnose breast cancer. Tumor-infiltrating lymphocytes (TILs) are often found in high concentrations in breast cancer tissue. Therefore, reliable identification of cell types, and their locations is imperative for accurate diagnoses. This software aims to complement the diangosis pipeline by automating the segmentation and quantification of these cells from whole slide images. Approaches, like TILseg, are carving out the interface between computational tools and traditional histological, pathological, and medicinal approaches. 

This software provides a straightforward method for analyzing H&E stained microscope images and classifying and quantifying TILs present in the stained image. Briefly, the software takes in a given number of slide images, divides the image into a set of smaller patches, and filters out patches that do not contain significant amounts of tissue. A superpatch consisting of several smaller patches, from multiple slide images, is then passed into the machine learning model. Additional hyper-parameters can be optimization can be performed by the software. Once training and validation is complete, the model is used to identify different cell types and segment and highlight identified TILs. Detailed images and comma-separated value files with quantifiable details are created for each patch containing tissue.

This software is broken into four different modules: `preprocessing.py`, `model_selection.py`, `seg.py`, `cluster_processing.py`. The modules are intended to be used sequentially and their main functions/use cases are outlined in sections below.

### 2. PREPROCESSING: ###
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

### 3. MODEL SELECTION: ###
- - - -


### 4. SEGMENTATION (SEG): ###
- - - -


### 5. CLUSTER PROCESSING: ###
- - - -


### 6. EXAMPLE USE CASE: ###
- - - -
Please reference the `tilseg_example.ipynb` file for an example of how this software may be used. The four primary modules above are implemented with a sample slide image for illustrative purposes.

### 7. REFERENCES: ###
- - - -
if needed
