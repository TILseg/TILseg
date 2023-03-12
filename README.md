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
TILseg (Tumor-Infiltrating Lymphocyte Segmentation) is a software created to segment different types of cells on a slide image containing various breast tissues. The microscope images are stained by hematoxylin and eosin (H&E) and are often used by pathologists for diagnosing breast cancer. Tumor-infiltrating lymphocytes (TILs) are often found in high concentrations in breast cancer tissue and having ways to highlight and segment these cells digitally is critical to future scientific research in histology, pathology, and medicine.

This software provides a straightforward method for analyzing these H&E stained microscope images and classifying the number of TILs present in the stained image. Briefly, the software takes in a given number of slide images, splits the image into reasonably sized patches, and filters out patches that do not contain significant tissue. A superpatch consisting of several smaller patches from across multiple different slide images is then used as the input for the machine learning model. The model used and its hyper-parameters can be optimized by the software as well, before finally training and creating the model. Finally, the model is used to identify different types of cells and the TILs are segmented and highlighted for all other patches not used in development/testing. Detailed images and comma-separated value files with quantifiable details are created for each patch containing tissue.

This software is broken into four different modules: `preprocessing.py`, `model_selection.py`, `seg.py`, `cluster_processing.py`. The modules are intended to be used sequentially and their main functions/use cases are outlined in sections below.

### 2. PREPROCESSING: ###
- - - -


### 3. MODEL SELECTION: ###
- - - -


### 4. SEGMENTATION (SEG): ###
- - - -


### 5. CLUSTER PROCESSING: ###
- - - -


### 6. EXAMPLE USE CASE: ###
- - - -


### 7. REFERENCES: ###
- - - -
if needed
