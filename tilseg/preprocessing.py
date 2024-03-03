"""
An image preprocessing module that can take in an svs
file as an image and return separate patches of that
image broken up and filtered down to hold patches of
only one type. This is then used for testing/using a
machine learning model or for superpatch creation in
a consequtive module.
"""
# Core library imports
import collections
import math
import os
import uuid

# External library imports
import numpy as np
import openslide  # pylint: disable = import-error
import pandas as pd
import scipy
from skimage import io
pd.options.mode.chained_assignment = None

# pylint: disable=no-else-raise, too-many-lines, too-many-locals, invalid-name
# pylint: disable=too-many-arguments, too-many-branches, useless-return
# pylint: disable=arguments-out-of-order
# noqa: E722,F841, C901


def open_slide(slidepath):
    """
    A function that opens a slide and returns an
    OpenSlide object and the slide's dimensions.

    Parameters
    -----
    slidepath (str): the complete path to the slide file (.svs)

    Returns
    -----
    slide (openslide.OpenSlide): the slide object created by OpenSlide
    slide_x (int): the x dimension of the slide
    slide_y (int): the y dimension of the slide
    """

    # check datatype of filepath
    if isinstance(slidepath, str):
        pass
    else:
        raise TypeError('slidepath must be a string')

    # acc file types
    file_type = ['.svs', '.tif', '.ndpi', '.vms', '.vmu'
                 '.scn', '.mrxs', '.tiff', '.svslide', '.bif',]

    # get the name and extension
    _, file_ext = os.path.splitext(slidepath)

    # check the ext
    if file_ext in file_type:
        pass
    else:
        raise TypeError('File provided in path not supported. \
                        Must be .svs file')

    # check the file path exists
    if os.path.exists(slidepath):
        pass
    else:
        raise ValueError('System cannot find the specified path to slide. \
                         Please ensure os.stat() can run on your path.')

    # get the slide object
    slide = openslide.OpenSlide(slidepath)

    # get x and y dimensions
    slide_x, slide_y = slide.dimensions

    return slide, slide_x, slide_y


def get_tile_size(maximum, size, cutoff=4):
    """
    A function that takes in a slide dimension and returns
    the optimal breakdown of each slide into x patches.

    Parameters
    -----
    maximum (int): the maximum dimension desired
    size (int): the size of the entire slide image
    cutoff (int): the maximum number of pixels to remove (default is 4)

    Returns
    -----
    dimension (int): the desired pixel size needed
    slices (int): the number of slices needed in the given direction
    remainder (int): the number of pixels lost with the slicing provided
    """

    # check maximum datatype
    if isinstance(maximum, int):
        pass
    else:
        raise TypeError('maximum tile size must be an integer')

    # check size of image datatype
    if isinstance(size, int):
        pass
    else:
        raise TypeError('tile size must be an integer')

    # check cutoff datatype
    if isinstance(cutoff, int):
        pass
    else:
        raise TypeError('cutoff must be an integer')

    # check that max is not bigger than the slide
    if maximum >= size:
        raise ValueError('maximum patch size must be smaller than slide size')
    else:
        pass

    remainder = size % maximum
            
    slices = math.trunc(size / maximum)

    # return requested values
    return dimension, slices, remainder


def percent_of_pixels_lost(lost_x, patch_x, lost_y, patch_y, x_size, y_size):
    """
    A function that calculates the total percentage of pixels
    lost from the whole slide when the slicing occurs.

    Parameters
    -----
    lost_x (int): the number of pixels lost in the x direction
    patch_x (int): the number of patches that are split in the x direction
    lost_y (int): the number of pixels lost in the y direction
    patch_y (int): the number of patches that are split in the y direction
    x_size (int): the total number of pixels in the x direction of the slide
    y_size (int): the total number of pixels in the y direction of the slide

    Returns
    -----
    percent (float): the percent of pixels deleted, rounded to two places
    """

    # check xpatch datatype
    if isinstance(patch_x, int):
        pass
    else:
        raise TypeError('patch_x must be an integer')

    # check xloss datatype
    if isinstance(lost_x, int):
        pass
    else:
        raise TypeError('lost_x must be an integer')

    # check xslide datatype
    if isinstance(x_size, int):
        pass
    else:
        raise TypeError('x_size must be an integer')

    # check ypatch datatype
    if isinstance(patch_y, int):
        pass
    else:
        raise TypeError('patch_y must be an integer')

    # check yloss datatype
    if isinstance(lost_y, int):
        pass
    else:
        raise TypeError('lost_y must be an integer')

    # check yslide datatype
    if isinstance(y_size, int):
        pass
    else:
        raise TypeError('y_size must be an integer')

    # check that loss and patch is not bigger than image (x-dir)
    if (lost_x + patch_x) > x_size:
        raise ValueError('(x-dir) patch and loss cannot be \
                         bigger than the image')
    else:
        pass

    # check that loss and patch is not bigger than image (x-dir)
    if (lost_y + patch_y) > y_size:
        raise ValueError('(y-dir) patch and loss cannot \
                         be bigger than the image')
    else:
        pass

    # calculate the percent
    percent = (lost_x * (patch_x - 1) + lost_y * (patch_y - 1)
               + lost_x * lost_y) / (x_size * y_size) * 100

    return percent


def save_image(path, name, image_array):
    """
    A function that saves an image given a path.

    Parameters
    -----
    path (str): the complete path to a directory to which the image should be saved
    name (str): the name of the file, with extension, to save
    image_array (np.array): a numpy array that stores image information
    
    Returns
    -----
    None
    """

    # check datatype of path
    if isinstance(path, str):
        pass
    else:
        raise TypeError('path must be a string')

    # check datatype of name
    if isinstance(name, str):
        pass
    else:
        raise TypeError('name must be a string')

    # check datatype of image_array
    if isinstance(image_array, (np.ndarray, np.generic)):
        pass
    else:
        raise TypeError('image array must be an array')

    # make sure it has more than one dimension
    try:
        _ = image_array.shape[2]  # noqa: F841
    except Exception as exc:  # noqa: E722
        raise IndexError('image array must be an NxMx3 array') from exc

    # check shape of np_array
    if image_array.shape[2] == 3:
        pass
    else:
        raise IndexError('image must NxMx3 array')

    # check to make sure there is a file extension in name
    ext = os.path.splitext(name)[-1].lower()

    if len(ext) == 0:
        raise ValueError('file extension must be included in name')
    else:
        pass

    # check the directory/path exists
    if os.path.exists(path):
        pass
    else:
        raise ValueError('System cannot find the specified path. \
                         Please ensure os.stat() can run on your path.')

    # create the entire saving directory
    save_as = os.path.join(path, name)

    # save the image
    io.imsave(save_as, image_array, check_contrast=False)

    return


def create_patches(slide, xpatch, ypatch, xdim, ydim):
    """
    A function that creates patches and yields an numpy
    array that describes the image patch for each patch
    in the slide.

    Parameters
    -----
    slide (openslide.OpenSlide): the OpenSlide object of the entire slide
    xpatch (int): the number of the patch in the x direction
    ypatch (int): the number of the patch in the y direction
    xdim (int): the size of the patch in the x direction
    ydim (int): the size of the patch in the y direction

    Returns
    -----
    np_patches (lst(np.arrays)): a list of all patches, each as a number array
    patch_position (lst(np.arrays)): a list of tuples containing indices
    """

    # check is it an openslide object
    if 'openslide' in str(type(slide)):
        pass
    else:
        raise TypeError('slide must be an openslide object')

    # check datatype of ypatch
    if isinstance(ypatch, int):
        pass
    else:
        raise TypeError('ypatch (number of patches in y-direction)\
                        must be an int')

    # check datatype of xpatch
    if isinstance(xpatch, int):
        pass
    else:
        raise TypeError('xpatch (number of patches in x-direction) \
                        must be an int')

    # check datatype of xdim
    if isinstance(xdim, int):
        pass
    else:
        raise TypeError('xdim (size of patch in the x-direction) \
                        must be an int')

    # check datatype of ydim
    if isinstance(ydim, int):
        pass
    else:
        raise TypeError('ydim (size of patch in the y-direction) \
                        must be an int')

    # make sure slide is big enough wrt patch size
    xslide, yslide = slide.dimensions

    # checking x patches size
    if xpatch*xdim > xslide:
        raise IndexError('size of total xpatches is bigger than provided \
                         (out of range) slide')
    else:
        pass

    # checking y patches size
    if ypatch*ydim > yslide:
        raise IndexError('size of total ypatches is bigger than provided \
                         (out of range) slide')
    else:
        pass

    # establish an empty patches list that will contain all patches
    np_patches = []

    # establish an empty list that will contain tuples of positions
    patch_position = []

    # iterate through the n x patches that will be made
    for xpatches in range(1, xpatch + 1):

        # get the starting left x coordinate of the patch
        start_x = (xpatches - 1) * xdim

        # iterate through the m y patches that will be made
        for ypatches in range(1, ypatch + 1):

            # get the starting left y coordinate of the patch
            start_y = (ypatches - 1) * ydim

            # convert patch into np array
            npimage = np.asarray(slide.read_region((start_x, start_y), 0,
                                                   (xdim, ydim)))

            # reformat array so it can be read properly
            np_patch = np.array(npimage)[:, :, :3]

            # append new patch to the master list of patches
            np_patches.append(np_patch)

            # append position to patch position list
            patch_position.append((xpatches, ypatches))

    return np_patches, patch_position


def get_average_color(img):
    """
    A function that returns the average RGB color
    of an input image array (in this case a patch).

    Parameters
    -----
    img (np.array): a numpy array containing all information about the
        RGB colors in a patch

    Returns
    -----
    average (np.array): a numpy array containing the RGB code for the average color
        of the entire patch
    """

    # check datatype of image_array
    if isinstance(img, (np.ndarray, np.generic)):
        pass
    else:
        raise TypeError('image array must be an np array')

    # make sure it has more than one dimension
    try:
        _ = img.shape[2]
    except Exception as exc:
        raise IndexError('image array must be an NxMx3 array') from exc

    # check shape of np_array
    if img.shape[2] == 3:
        pass
    else:
        raise IndexError('image must NxMx3 array')

    # calculate the average
    average = img.mean(axis=0).mean(axis=0)

    return average


def get_grey(rgb):
    """
    A function that calculates the greyscale
    value of an image given an RGB array.

    Parameters
    -----
    rgb (np.array): a numpy array containing three values, one each for R, G, and B

    Returns
    -----
    grey (float): the greyscale value of an image/patch
    """

    # check datatype of input
    if isinstance(rgb, (list, pd.core.series.Series, np.ndarray)):
        pass
    else:
        raise TypeError('rgb argument must be a list, \
                        pandas series, or np array')

    # make sure length can be accessed
    try:
        _ = len(rgb)
    except Exception as exc:
        raise TypeError('cannot get length of rgb') from exc

    # make sure it is the correct length
    if len(rgb) == 3:
        pass
    else:
        raise IndexError('input not correct size; must have three entries')

    grey = (rgb[0] + rgb[1] + rgb[2]) / 3

    return grey


def save_all_images(df, path, f):  # pylint disable = invalid-name
    """
    A function to save all the images as background or tissue.

    Parameters
    -----
    df (pd.DataFrame): the dataframe that is already created containing patches,
        average patch color, and the greyscale value
    path (str): the path to which the folders and subdirectories will be made
    f (str): the slide .svs file name that is currently being read

    Returns
    -----
    None, but all images are saved
    """

    # check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError('The file path given does not exist.')
    else:
        pass

    # check if the file name has an extension
    if '.' not in f:
        raise TypeError('The file name provided for the image \
                        has no extension.')
    else:
        pass

    # check that dataframe is actually a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The dataframe entered is not a dataframe object.')
    else:
        pass

    # check that the dataframe has a position column
    if 'patch_xy' not in df.columns:
        raise ValueError('The dataframe does not have positions shown \
                          under a patch_xy column for each patch.')
    else:
        pass

    # check that the dataframe column patch_xy only contains tuples
    for elem in df.patch_xy:
        if not isinstance(elem, tuple):
            raise TypeError('The position column does \
                            not contain only tuples.')
        else:
            pass

    # get name of file without extension
    slide_name = f.split('.')[0]

    # name all used directories
    slide_name_path = os.path.join(path, slide_name)

    # check if the folder does not already exist
    assert not os.path.isfile(slide_name_path), 'An existing \
        folder with the slide name already exists.'

    # make all necessary directories
    os.mkdir(slide_name_path)

    # iterate through all rows of the dataframe
    for _, row in df.iterrows():

        # name the file that will be saved based on
        # its index on the whole slide image
        name = 'position_' + str(row['patch_xy'][0]) + '_' + \
            str(row['patch_xy'][1]) + 'tissue.tif'

        # save the image
        save_image(slide_name_path, name, row['patches'])

    return


def find_max(arr, cutoff, greater):
    """
    A function that finds the max value of a list/array
    within a specific range.

    Parameters
    -----
    arr (collections.abc.Sequence, np.ndarray): the array that contains the list of data in question
    cutoff (int): the value at which you want to start looking for a maximum
    greater (boolean): a boolean that determines if you want the maximum above
        or below the cutoff (above is when greater=False)

    Returns
    -----
    loca: the index (from zero) at which the maximum value occurs
    """

    # check that greater is a boolean
    if not isinstance(greater, bool) or (greater and not greater):
        raise TypeError('The greater argument must be True or False.')
    else:
        pass

    # check that arr is a list or array
    if not isinstance(arr, (collections.abc.Sequence, np.ndarray)):
        raise TypeError('The input list must be an array or list.')
    else:
        pass

    # check that the cutoff value is an integer or float
    if not isinstance(cutoff, (int, float)):
        raise TypeError('The cutoff value must be an integer or float value.')
    else:
        pass

    # check that all list values are positive
    if any(item < 0 for item in arr):
        raise ValueError('The list can only contain non-negative values.')
    else:
        pass

    # a dummy number for the max that will never actually be the max
    maximum = 0

    # iterate through the array, but enumerate so that it is easy to get index
    for index, number in enumerate(arr):

        # if interested in a maximum below the cutoff and the
        # index is greater than this cutoff, then break out of the loop
        if greater and index > cutoff:
            break

        # if interested in a maximum above the cutoff and the index
        # is less than the cutoff, continue looping but do not do anything
        if not greater and index < cutoff:
            continue

        # check if the number in the appropriate range is
        # greater than the maximum
        if number > maximum:

            # if it is, reassign the maximum value at this new
            # value and record the index
            maximum = number
            loca = index

        # if the number is not greater than the maximum do nothing and continue
        else:
            continue

    return loca


def find_min(arr, range_min, range_max):
    """
    A function that finds the min value of a list/array
    within a specific range.

    Parameters
    -----
    arr (collections.abc.Sequence, np.ndarray): the array that contains the list of data in question
    range_min (int, float): the lower bound on which to look for the minimum
    range_max (int, float): the upper bound on which to look for the minimum

    Returns
    -----
    loca (int): the index (from zero) at which the minimum value occurs
    """

    # check that the range_max value is an integer or float
    if not isinstance(range_max, (int, float)):
        raise TypeError('The range_max value must be an \
                        integer or float value.')
    else:
        pass

    # check that the range_min value is an integer or float
    if not isinstance(range_min, (int, float)):
        raise TypeError('The range_min value must be an \
                        integer or float value.')
    else:
        pass

    # check that arr is a list or array
    if not isinstance(arr, (collections.abc.Sequence, np.ndarray)):
        raise TypeError('The input list must be an array or list.')
    else:
        pass

    # check that all list values are positive
    if any(item < 0 for item in arr):
        raise ValueError('The list can only contain non-negative values.')
    else:
        pass

    # check that the range min and range max are less than or greater than
    assert range_min < range_max, 'The range minimum is \
        greater than the maximum.'
    assert range_min != range_max, 'The range minimum and \
        maximum are the same.'

    # a dummy number for the min that will never actually be the min
    minimum = 1000000

    # iterate through the array, but enumerate so that it is easy to get index
    for index, number in enumerate(arr):

        # check if the index is between the desired range
        if range_min < index < range_max:

            # if it is in the correct range then check if the number
            # is less than the current minimum
            if number < minimum:

                # if it is less than the current minimum, reassign
                # the minimum and record the new index
                minimum = number
                loca = index

            # if it is in the correct range but not less than the
            # current minimum then continue through the loop and do nothing
            else:
                continue

        # if the index is out of the desired range,
        # continue and do nothing with that index
        else:
            continue

    return loca


def compile_patch_data(slide, ypatch, xpatch, xdim, ydim):
    """
    A function that compiles all relevant data for
    all patches into a dataframe.

    Parameters
    -----
    slide (openslide.OpenSlide): the OpenSlide object of the entire slide
    ypatch (int): the number of patches in the y direction
    xpatch (int): the number of patches in the x direction
    xdim (int): the size of the patch in the x direction
    ydim (int): the size of the patch in the y direction

    Returns
    -----
    patchdf (pd.DataFrame): a pandas dataframe containing the three following
    """

    # create a dataframe to contain all patch information from a slide
    patchdf = pd.DataFrame()

    # unpack information from created patches
    patch_list, index_list = create_patches(slide, xpatch, ypatch, xdim, ydim)

    # create a column with a numpy array patch in each row of the dataframe
    patchdf['patches'] = patch_list

    # create a column with a tuple indicating the corresponding patch
    patchdf['patch_xy'] = index_list

    # for each patch, calculate the average RGB value of the entire patch
    patchdf['RGB_avg'] = patchdf.apply(lambda row:
                                       get_average_color(row['patches']),
                                       axis=1)

    # add another column that converts the
    # average RGB color to a greyscale color
    patchdf['greys'] = patchdf.apply(lambda row:
                                     get_grey(row['RGB_avg']), axis=1)

    return patchdf


def is_it_background(cutoff, actual):
    """
    A function that tests if a specific image should
    be classified as a background image or not.

    Parameters
    -----
    cutoff (int): the cutoff value for a background image

    Returns
    -----
    background (boolean): a boolean that is True if the patch
        should be considered background
    """

    # test if the actual value is greater than the cutoff
    background = bool(actual > cutoff)
    return background


def sort_patches(df, lin_space=100, approx_between=200):
    """
    A function that starts sorting patches based on a KDE,
    determines a cutoff value, and calculates the final
    dataframe for each image.

    Parameters
    -----
    df (pd.DataFrame): the dataframe that is already created containing patches,
        average patch color, and the greyscale value
    lin_space (int): the multiple by which the KDE axis will be split into
        while it is being formed for a PDF (default is 100)
    approx_between (int): the approximate value at which the grey values
        will be split into two populations in the bimodal distribution.
        This is usually around 200 for slides and is going to be
        set to that as a default.

    Returns
    -----
    df (pd.DataFrame): an updated dataframe with a background column that indicates
        if a patch should be considered background or not
    """

    # check that the input is a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The input dataframe is not a dataframe.')
    else:
        pass

    # check that the dataframe contains a greys column
    if 'greys' not in df.columns:
        raise KeyError('The input dataframe does not contain a greys column.')
    else:
        pass

    # check that the dataframe column greys only contains numeric values
    for elem in df.greys:
        if not isinstance(elem, (int, float)):
            raise TypeError('The position column does not contain only \
                            numeric values.')
        else:
            pass

    # calculate min, max, and range of grey values
    minimum_grey = int(df['greys'].min())
    maximum_grey = int(df['greys'].max())
    range_grey = maximum_grey - minimum_grey

    # put all grey values into a list
    list_of_greys = df['greys'].values.tolist()

    # create a linspace for all grey values for
    # which the PDF will be calculated
    grey_space = np.linspace(minimum_grey, maximum_grey,
                             range_grey * lin_space)

    # create a KDE distribution from the list of greys
    kde_distr = scipy.stats.gaussian_kde(list_of_greys)

    # use the KDE distribution to create a PDF of
    # the grey values along the grey space
    kde_pdf = kde_distr(grey_space)

    # find all local maxima and minima
    color_max = find_max(kde_pdf, (approx_between - minimum_grey)
                         * lin_space, True)
    background_max = find_max(kde_pdf, (maximum_grey - approx_between)
                              * lin_space, False)
    minimum = find_min(kde_pdf, color_max, background_max)

    # complete correct reindexing
    color_max = color_max / lin_space + minimum_grey
    background_max = background_max / lin_space + minimum_grey
    minimum = minimum / lin_space + minimum_grey

    # calculate the cutoff value for greys
    cutoff_value = (background_max + minimum) / 2

    # add column to dataframe that classifies each image as background or not
    df['background'] = df.apply(lambda row: is_it_background(cutoff_value,
                                                             row['greys']),
                                axis=1)

    return df


def main_preprocessing(complete_path, training=True, save_im=True,
                       max_tile_x=3000, max_tile_y=2000):
    """
    The primary function to perform all preprocessing
    of the data, creating patches and returning a final
    large dataframe with all information contained.

    Parameters
    -----
    complete_path (str): the full path to the file containing all svs
                   files that will be used for training the model or a single
                   svs file to get an output value
    training (boolean): a boolean that indicates if this preprocessing is
              for training data or if it to only be used for the
              existing model
    save_im (boolean): a boolean that indicates if tissue images should be saved
             (beware this is a lot of data, at least 10GB per slide)
    max_tile_x (int): the maximum x dimension size, in pixels,
                of a slide patch (default is 4000)
    max_tile_y (int): the maximum y dimension size, in pixels,
                of a slide patch (default is 3000)

    Returns
    -----
    all_df or sorted_df (pd.DataFrame): a dataframe containing all necessary information for
        creating superpatches for training (all_df) or for inputting into an
        already generated model (sorted_df)
    """

    if training:  # pylint: disable=no-else-return

        all_df = pd.DataFrame()
        # iterate through all files in the directory
        for file in os.listdir(complete_path):

            # check that the file is a slide image
            if file.endswith('.svs'):

                # open the slide file
                full_file_path = os.path.join(complete_path, file)
                print(full_file_path)
                slide_file, slide_x, slide_y = open_slide(full_file_path)

                # calculate dimensions and losses
                ydim, ypatch, yloss = get_tile_size(max_tile_y, slide_y)
                xdim, xpatch, xloss = get_tile_size(max_tile_x, slide_x)
                loss_percentage = percent_of_pixels_lost(xloss, xpatch,
                                                         yloss, ypatch,
                                                         slide_x, slide_y)

                # get the dataframe for all patches in the slide
                dataframe_patches = compile_patch_data(slide_file, ypatch,
                                                       xpatch, xdim, ydim)

                # determine if patches are background or not
                sorted_df = sort_patches(dataframe_patches)

                # drop all background images from dataframe
                sorted_df = sorted_df.loc[~sorted_df.background, :]

                # save all images to correct directory if desired
                if save_im:
                    save_all_images(sorted_df, complete_path, file)

                # create a unique id for this slide image
                sorted_df['UUID'] = uuid.uuid4()

                # add the dataframe to the total training dataframe
                all_df = pd.concat([all_df, sorted_df], ignore_index=True)

                # print out the percent of pixels lost
                print(f'Percent of pixels lost in pre-processing for {file}: \
                      {loss_percentage} %')

            # if the file is not a slide image then do nothing and continue
            else:  # pylint: disable=no-else-return
                continue

        # give unique numeric ID to each slide counting from 0 upwards
        all_df['id'] = pd.factorize(all_df['UUID'])[0]

        # remove UUID column across the entire dataframe
        all_df = all_df.drop(columns='UUID')

        return all_df

    else:

        # open the slide file
        slide_file, slide_x, slide_y = open_slide(complete_path)
        path = os.path.dirname(complete_path)
        file = os.path.basename(complete_path)

        # calculate dimensions and losses
        ydim, ypatch, yloss = get_tile_size(max_tile_y, slide_y)
        xdim, xpatch, xloss = get_tile_size(max_tile_x, slide_x)
        loss_percentage = percent_of_pixels_lost(xloss, xpatch,
                                                 yloss, ypatch,
                                                 slide_x, slide_y)

        # get the dataframe for all patches in the slide
        dataframe_patches = compile_patch_data(slide_file, ypatch,
                                               xpatch, xdim, ydim)

        # determine if patches are background or not
        sorted_df = sort_patches(dataframe_patches)

        # drop all background images from dataframe
        sorted_df = sorted_df.loc[~sorted_df.background, :]

        # save all images to correct directory if desired
        if save_im:
            save_all_images(sorted_df, path, file)

        # print out the percent of pixels lost
        print(f'Percent of Pixels Lost in Pre-Processing: {loss_percentage} %')

        return sorted_df


def count_images(path=os.getcwd()):
    """
    Count images finds the number of whole slide images available
    in your current working directory.

    Parameters:
    ------------
    None

    Returns:
    -----------
    img_count (int): the number of whole slide images in your directory
    """
    # check datatype of path
    if isinstance(path, str):
        pass
    else:
        raise TypeError('path must be a string')

    # check the directory/path exists
    if os.path.exists(path):
        pass
    else:
        raise ValueError('System cannot find the specified path.\
                          Please ensure os.stat() can run on your path.')

    file_list = os.listdir(path)

    # count the number of svs images in cwd
    img_count = 0
    for file in file_list:
        if file.endswith('.svs'):
            img_count += 1
        else:
            continue

    return img_count


def patches_per_img(num_patches, path=os.getcwd()):
    """
    Patches_per_img calculates the number of patches
    to be extracted from each image. If there are no
    images in the current working directory or provided
    path.

    Parameters:
    -------------
    num_patches (int): number of total patches (that make up the entire image)
    path -- optional (str): path in which images might be located

    Return:
    --------------
    patch_img (int): number of patches to be extraced from each image
    """

    # check num_patches datatype
    if isinstance(num_patches, int):
        pass
    else:
        raise TypeError('num_patches must be an int')

    # check path datatypes
    if isinstance(path, str):
        pass
    else:
        raise TypeError('path must be a string')

    # check the file path exists
    if os.path.exists(path):
        pass
    else:
        raise ValueError('System cannot find the specified path.\
                          Please ensure os.stat() can run on your path.')

    # find the number of images in cwd
    img_count = count_images(path)

    if img_count == 0:
        # print('There are no images in the current working directory')
        patch_img = 0
    else:
        # find the patches per image
        patch_img = num_patches / img_count

    # return patches per image
    return patch_img


def get_superpatch_patches(patches_df, patches=8, path=os.getcwd()):
    """
    This function finds the patches to comprise the superpatch.
    The patches are selected based off of distribution of
    average color and the source image. This way, the superpatch
    is not entirely made of patches from one image (unless there is
    only one image available).

    Parameters:
    -------------
    df (pd.DataFrame): MUST be dataframe from main_preprocessing output

    Returns:
    -------------
    patches_list (lst): list of the patches to be included in superpatch
                        individual patches are stored as np arrays
    """

    # check datatype of patches_df
    if isinstance(patches_df, pd.DataFrame):
        pass
    else:
        raise TypeError('patches_df must be a pandas dataframe')

    # check datatype of patches
    if isinstance(patches, int):
        pass
    else:
        raise TypeError('patches must be an int')

    # check path datatype
    if isinstance(path, str):
        pass
    else:
        raise TypeError('path must be a string')

    # check the file path exists
    if os.path.exists(path):
        pass
    else:
        raise ValueError('System cannot find the specified path. \
                         Please ensure os.stat() can run on your path.')

    # remove all unnecessary columns
    try:
        df = patches_df.drop(['background', 'RGB_avg'], axis=1)
    except Exception as exc:
        raise KeyError(
            'patches_df must contain background and RGB_avg column') from exc

    # make sure necessary columns exist
    try:
        _ = df['patches']
        _ = df['id']
    except Exception as exc:  # noqa: E722
        raise KeyError('patches_df must have patches and id columns') from exc

    # make sure there are enough patches
    if len(df.index) >= patches:
        pass
    else:
        raise IndexError('Fewer patches available in dataframe than \
                        requested for superpatch')

    # patches list
    patches_list = []

    # calculate patches per image
    # need to change this
    patch_per = math.floor(patches_per_img(patches, path))

    # bin the average values for each patch
    df['grey_binned'] = pd.cut(df['greys'], bins=patches+1)

    # find the bins and the img_labels
    bins = df['grey_binned'].unique()
    img_labels = df['id'].unique()

    for img in img_labels:

        # get only the patches of the image
        img_df = df.loc[df['id'] == img]

        # start counting the number of patches
        # from this image
        patch_count = 0

        for bin_i in bins:

            # get the dataframe that contains the
            # patches with the average color of interest
            bin_df = img_df.loc[img_df['grey_binned'] == bin_i]

            # pick a patch from this set of relevant patches
            patch_df = bin_df.sample()
            actual_patch = patch_df['patches']

            # add patch to list of patches to access later
            patches_list.append(actual_patch)

            # count the patches used for this image
            patch_count += 1

            # get only the number of patches per image
            if patch_count >= patch_per:  # pylint: disable=no-else-break
                # leave the for loop if you have the number of patches
                break
            else:
                # keep looping if you need more patches from this image
                continue

    # return the list of patches that make up the superpatch
    return patches_list


def superpatcher(patches_list, sp_width=3):
    """
    TODO: Update the naming convection for the returned variable!!!
     
    Superpatcher uses the selected patches and
    converts the individual patches into one patch

    Parameters:
    ------------
    patches_list (lst): MUST be output from get_superpatch_patches
                        list of patches
    sp_width (int): the width of a superpatch (how many images, default 3)

    Returns:
    --------------
    superpatch (np.array): np.array that contains the superpatch
    """

    # check sp_width datattype
    if isinstance(sp_width, int):
        pass
    else:
        raise TypeError('sp_width must be an int')

    # check datatype of input
    if isinstance(patches_list, (list, pd.core.series.Series, np.ndarray)):
        pass
    else:
        raise TypeError('patches_list must be a list, pandas series, \
                        or np array')

    num_patches = len(patches_list)
    sp_height_calc = math.ceil(num_patches/sp_width)
    sp_width_calc = int(num_patches/sp_height_calc)

    # initialize the row patch (starting column, adding column wise)
    patch_array_0 = (patches_list[0]).values[0]

    # check patch_array datatype
    if isinstance(patch_array_0, (np.ndarray, np.generic)):
        pass
    else:
        raise TypeError('patches_list must contain np array of patch')

    patch_index = 2

    for j in range(0, sp_height_calc):

        # build the row (build the row)
        for i in range(1, sp_width_calc):

            # get patch at index i (in list)
            patch_array_i = (patches_list[patch_index]).values[0]

            # check patch_array datatype
            if isinstance(patch_array_i, (np.ndarray, np.generic)):
                pass
            else:
                raise ValueError('patches_list must contain np array of patch')

            patch_index += 1

            # update the overall patch to be these patched together
            try:
                patch_array_0 = np.concatenate((patch_array_0, patch_array_i),
                                               axis=1)
            except Exception as exc:
                raise TypeError('patches list does not \
                                contain correct datatypes') from exc

            # save the finished row
            if i == (sp_width_calc-1):
                patch_row_0 = patch_array_0
                patch_array_0 = (patches_list[j+1]).values[0]

        # find the first row (adding row wise)
        if j == 0:
            patch_row_1 = patch_row_0

        # else add the row to the other rows
        else:
            patch_row_1 = np.concatenate((patch_row_0, patch_row_1), axis=0)

    return patch_row_1


def preprocess(path, patches=6, sp_width=3, training=True, save_im=True,
               max_tile_x=4000, max_tile_y=3000):
    """
    #TODO: Continue to edit what the output and input variables definintions are
    
    The preprocess function that is called when running the
    code. Complete details are found in the README file. This
    only calls other functions and is used as a wrapper.
    
    Parameters:
    ------------
    path (str): path to the folder containing the .svs slide files
    patches (int): number of patches to create superpatch with
    training (boolean): a boolean that indicates if this preprocessing is
                        for training data or if it to only be used for the
                        existing model
    save_im (boolean): a boolean that indicates if tissue images should be saved
                    (beware this is a lot of data, at least 10GB per slide)
    max_tile_x (int): the maximum x dimension size, in pixels,
                    of a slide patch (default is 4000)
    max_tile_y (int): the maximum y dimension size, in pixels,
                    of a slide patch (default is 3000)

    Returns:
    --------------
    spatch (pd.DataFrame): a dataframe containing all necessary information for
                        creating superpatches for training (all_df) or for inputting into an
                        already generated model (sorted_df)
    """
    if training:
        dataframe = main_preprocessing(path, patches, training, save_im,
                                       max_tile_x, max_tile_y)
        plist = get_superpatch_patches(dataframe, patches, path)
        spatch = superpatcher(plist, sp_width)
        save_image(path, 'superpatch_training.tif', spatch)

    else:
        #TODO: this is never going to return anything? meant to equal to patch?
        main_preprocessing(path, training, save_im, max_tile_x, max_tile_y)

    return spatch
