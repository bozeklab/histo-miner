#Lucas Sancéré -

from collections import MutableMapping
import os
import imagesize
from openslide import OpenSlide


### Utils Functions

def checkdownsampling(originalimagepath: str, downsampleimagepath: str, downfactor: int):
    """
    Check if for 2 images, one is the downsample version of the first one
    by a factor which is equal to the provided factor

    Parameters:
    -----------
    originalimage: str
        Path to the original image that was downsample (format as to be supported by openslide)
    downsampleimage: str
        Path to the downsample image (format as to be supported by imagesize)
    downfactor: int
        Value of the downsampling factor that we wwant to check.
    Returns
    -------
    """
    originalimage = OpenSlide(originalimagepath)
    originalwidth = originalimage.dimensions[0]
    originalheight = originalimage.dimensions[1]
    # No need to open the downsample image neither,
    # just use get package working for most of image extensions
    downimage_width, downimage_heights = imagesize.get(downsampleimagepath)
    newimage_width = downimage_width * downfactor
    newimage_heights = downimage_heights * downfactor
    if [newimage_width, newimage_heights] == [originalwidth, originalheight]:
        print(os.path.split(downsampleimagepath)[1],
              "is exactly downsampled by a factor", downfactor,
              "from", os.path.split(originalimagepath)[1])
    else:
        print("/!\ WARNING /!\ ",
              os.path.split(downsampleimagepath)[1],
              "is not downsampled by a factor", downfactor,
              "from", os.path.split(originalimagepath)[1])


def convert_flatten_redundant(inputdic: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    Create a flatten dictionnary, meaning from a dictionnary containing nested keys,
    it will generate a dictionnary with simple keys-values pairs.

    Even if some nested keys has the same names the newly generated key won't be the same.

    Examples:
    - This line in the input dict:
    { "Key1-1": {"Key2 : {"Key3-1": 0.01, "Key3-2": 0.05, "Key3-3": 0.002}},  "Key1-2":{}}}
    - Will become in the newly generated JSON:
    {"Key1-1_Key2_Key3-1": 0.01, "Key1-1_Key2_Key3-2": 0.05, "Key1-1_Key2_Key3-3": 0.002}

    Link to original code:
    https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/

    Parameters:
    -----------
    inputdic: dict
        dictionnary the user want to flatten
    parent_key: str, optional
    sep: str, optional
        separation between nested key in the name of the newly created key
    Returns:
    --------
    object : dict
        flatten dictionnary generated
    """
    items = []
    for k, v in inputdic.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, MutableMapping):
            items.extend(convert_flatten_redundant(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_flatten(inputdic: dict, parent_key: str = '') -> dict:
    """
    Create a flatten dictionnary, meaning from a dictionnary containing nested keys,
    it will generate a dictionnary with simple keys-values pairs.

    If some nested keys has the same names the newly generated key will also have the same name!!

    Examples:
    - This line in the input dict:
    { "Key1-1": {"Key2 : {"Key3-1": 0.01, "Key3-2": 0.05, "Key3-3": 0.002}},  "Key1-2":{}}}
    - Will become in the newly generated JSON:
    {"Key3-1": 0.01, "Key3-2": 0.05, "Key3-3": 0.002}

    Link to original code:
    https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/

    Parameters:
    -----------
    inputdic: dict
        dictionnary the user want to flatten
    parent_key: str, optional
    Returns:
    --------
    object : dict
        flatten dictionnary generated
    """
    items = []
    for k, v in inputdic.items():
        new_key = k if parent_key else k

        if isinstance(v, MutableMapping):
            items.extend(convert_flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)