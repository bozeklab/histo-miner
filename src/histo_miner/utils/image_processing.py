#Lucas Sancéré -

"""
Here we will add the functions from classic_image_processing
from Misc_Utils othat are needed (probably downsampling code and so on) and are not present anywhere on this repo
"""

import copy
import glob
import json
import math
import multiprocessing as mp
import os
from ast import literal_eval
from itertools import product

import PIL
import cv2
import imagesize
import numpy as np
import shapely.geometry
import yaml
from PIL import Image
from skimage.measure import regionprops, label
from skimage.util import view_as_blocks
from sklearn.preprocessing import binarize
from tqdm import tqdm
from openslide import OpenSlide

PIL.Image.MAX_IMAGE_PIXELS = 10000000000000



## Functions

def downsample_image(imagepath: str, downfactor: int, 
                     savename: str = '_downsampled',
                     savefolder: str= '') -> None:
    """
    Downsample an image with format compatible with PILLOW and save the output image. Use of
    PILLOW to read and process images (because it can read big images, AND CV2 CANNOT).

    Parameters
    ----------
    imagepath: str
        Path to the image to downsample
    downfactor: int
        Value of the downsampling factor.
    savename: str, optional
        Suffix of the name of the file to save. Image will be saved as png.
    savefolder: str, optional
        Name of the subfolder where to save the image. 
        By default there is no name and so the output image will be saved
        In the same folder as the input image.
    Returns
    -------
    """
    # Load image
    image = Image.open(imagepath)
    width, height = image.size[0], image.size[1]
    new_width = width // downfactor
    new_height = height // downfactor
    # Downsampling Operation
    image.thumbnail((new_width, new_height))
    # Now save the downsampled image
    pathtofolder, filename = os.path.split(imagepath)
    filenamecore, ext = os.path.splitext(filename)
    savepath = pathtofolder + '/' + savefolder + filenamecore + savename + ext
    image.save(savepath)


def resize(image: str, newheight: int, newwidth: int, savename: str = '_resized') -> None:
    """
    Resize an image following the new height and widee given as input. Use of
    PILLOW to read and process images (because it can read big images, AND CV2 CANNOT).

    Parameters
    ----------
    image: str
        Path to the image user wants to resize
    newheight: int
        Value of the new height.
    newwidth: int
        Value of the new width.
    savename: str, optional
        Suffix name to add to the name of the image saved. Final name is original name + savename.
    Returns
    -------
    """
    # Load image
    imagetoresize = Image.open(image)
    # Resizing operation
    imageresized = imagetoresize.resize((newwidth, newheight))
    # Having path of the image, savename being a suffix to name saved
    folderpath, namewithext = os.path.split(image)
    name, ext = os.path.splitext(namewithext)
    savepath = folderpath + '/' + name + savename + ext
    imageresized.save(savepath)


def resize_accordingly(image: str, modelimage: str, savename: str = '_resized') -> None:
    """
    Resize an image to match with the modelimage size given as input and save it. Use the cv2.resize function
    and cv2 and PILLOW to read images (can read big images).

    Parameters
    ----------
    image: str
        Path to the image user wants to resize
    modelimage: str
        Path to the image of the desired shape for 'image' to be resized with.
    savename: str, optional
        Suffix name to add to the name of the image saved. Final name is original name + savename.
    Returns
    -------
    """
    # Resizing
    imagetoresize = Image.open(image)
    # No need to open the image, just use get package working for most of image extensions
    modelimage_width, modelimage_heights = imagesize.get(modelimage)
    print('desired shape is', modelimage_width, modelimage_heights)
    print('resizing...', image)
    # Looks like shape as to be inverted for a weird convention reason
    imageresized = imagetoresize.resize((modelimage_width, modelimage_heights))
    # Having path of the image, savename being a suffix to name saved
    folderpath, namewithext = os.path.split(image)
    name, ext = os.path.splitext(namewithext)
    savepath = folderpath + '/' + name + savename + ext
    # save
    print('saving...')
    imageresized.save(savepath)



def downsample_wsi(filename, 
                   output_path,  
                   target_downsample,
                   thumbnail_extension):
    """ 
    Based on Juan Pisula code.

    Save thumbnail of WSI.
    
    Args:

    Returns:
        tuple: The input filename and a boolean indicating success.
    """
    input_path, wsi_fn = os.path.split(filename)[0], os.path.split(filename)[1]
    thumbnail_extension = '.' + thumbnail_extension

    thumbnail_path = os.path.join(output_path, wsi_fn.replace(
        '.{}'.format(wsi_fn.split('.')[-1]), thumbnail_extension))
    if os.path.exists(thumbnail_path):
        return wsi_fn, True
    try:
        slide = OpenSlide(os.path.join(input_path, wsi_fn))
    except BaseException as err:
        return wsi_fn, False


    target_zoom_level = slide.get_best_level_for_downsample(target_downsample)
    zoom_dims = slide.level_dimensions[target_zoom_level]
    rgba_img = slide.read_region((0, 0), target_zoom_level, zoom_dims)
    rgb_img = rgba_img.convert('RGB')
    os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
    rgb_img.save(thumbnail_path)
    return wsi_fn, True




def downsample_image_segmenter(pathtofolder: str,
                               fileext: str = 'ndpi',
                               outputext: str = 'tif', 
                               downfactor: int = 32, 
                               savefolder: str = 'downsampling/',
                               savename: str = '') -> None:
    """
    Downsample an image with format compatible with PILLOW and save the output image. Use of
    PILLOW to read and process images (because it can read big images, AND CV2 CANNOT).

    Parameters
    ----------
    imagepath: str
        Path to the image to downsample
    downfactor: int
        Value of the downsampling factor.
    savename: str, optional
        Suffix of the name of the file to save.
    Returns
    -------
    """
    # loop over all images

    # apply downsample function

    # save new images in a new folder / downsample folder

    # later save segmenter output not there but where it is define in config
    files = os.path.join(pathtofolder, '*.'+ fileext)
    files = glob.glob(files)
    output_path = pathtofolder + '/' + savefolder
    for fname in tqdm(files):
        if os.path.exists(fname):
            downsample_wsi(filename = fname, 
                           output_path=output_path,
                           target_downsample=downfactor,
                           thumbnail_extension=outputext)