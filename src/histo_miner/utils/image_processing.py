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

def downsample_image(imagepath: str, downfactor: int, savename: str = '_downsampled') -> None:
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
    savepath = pathtofolder + '/' + filenamecore + savename + ext
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