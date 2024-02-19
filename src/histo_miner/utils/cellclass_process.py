
import json

import numpy as np
from PIL import Image


def update_cellclass(classjson: str, maskmap: str, maskmapdownfactor: int) -> None:
    """
    In the loaded json, replace all the cells predicted as tumor cells outside tumor region
    into epithelial cells (class 6) and rewritte previous json with new changes

    Parameters
    ----------
    classjson: str
        Path to the json file (for instance output of hovernet then formatted according to the note above).
        It must contains, inside each object ID key (numbers):
        - a 'centroid' key, containing the centroid coordinates of the object
        - a 'type" key, containing the class type of the object (classification)
    maskmap: str, optional
        Path to the binary image, mask of a specific region (here tumor) of the original image.
        The image must be in PIL supported format.
    maskmapdownfactor: int, optional
        Set what was the downsample factor when the maksmap (tumor regions) were generated
    Returns
    -------
    """
    with open(classjson, 'r') as filename:
        classjsondict = json.load(filename)  # data must be a dictionnary

    maskmap = Image.open(maskmap)
    maskmap = np.array(maskmap)  # The maskmap size is not the same as the input image, it is downsampled
    # Check shape of the input file
    if len(maskmap.shape) != 2 and len(maskmap.shape) != 3:
        raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                         'Image type not supported ')
    elif len(maskmap.shape) == 3:
        if maskmap.shape[2] == 3:
            maskmap = maskmap[:, :, 0]  # Keep only one channel of the image if the image is 3 channels (RGB)
        else:
            raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                             'Image type not supported ')

    # Define more explicity the cell classes
    tumorclass = 5
    epithelialclass = 6

    # Create a new dict where the cancer cells outside tumor regions will be replace
    #With epithelial cells
    classjsonkeys = classjsondict.keys()  # should extract only first level keys
    classdict2update = classjsondict
    for nucleus in classjsonkeys:
        ycoordinate = int(classdict2update[nucleus]['centroid'][0])
        xcoordinate = int(classdict2update[nucleus]['centroid'][1])
        nucleusclass = classdict2update[nucleus]['type']
        # Keep nucleus predicted as tumor but outside tumor region
        # then it will be later transformed to epithelial cells (class 6)
        if nucleusclass == tumorclass and \
                maskmap[int(xcoordinate / maskmapdownfactor), int(ycoordinate / maskmapdownfactor)] == 0:
            classdict2update[nucleus]['type'] = epithelialclass

    # Load the dict as json file
    classdictupdated = classdict2update
    # Save the file with new dictionnary as json and overwritte the previous one
    with open(classjson, 'w') as filename:
        json.dump(classdictupdated, filename)


def cancelupdate(classjson: str) -> None:
    """
    In the loaded json, change all epithelial cells into tumor cells and rewritte previous json with new changes

    Parameters
    ----------
    classjson: str
        Path to the json file (for instance output of hovernet then formatted according to the note above).
        It must contains, inside each object ID key (numbers):
        - a 'centroid' key, containing the centroid coordinates of the object
        - a 'type" key, containing the class type of the object (classification)

    Returns
    -------
    """
    with open(classjson, 'r') as filename:
        classjsondict = json.load(filename)  # data must be a dictionnary

    # Define more explicity the cell classes
    tumorclass = 5
    epithelialclass = 6
    classdict2update = classjsondict

    for nucleus in classdict2update.keys():  # should extract only first level keys
        nucleusclass = classdict2update[nucleus]['type']
        # Keep nucleus predicted as tumor but outside tumor region
        # then it will be later transformed to epithelial cells (class 6)
        if nucleusclass == epithelialclass:
            classdict2update[nucleus]['type'] = tumorclass
    # Load the dict as json file
    new_json = json.loads(classdict2update)
    # Save the file with new dictionnary as json and overwritte the previous one
    with open(classjson, 'w') as filename:
        json.dump(new_json, filename)