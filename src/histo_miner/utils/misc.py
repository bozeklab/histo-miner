#Lucas Sancéré -

import json
import os
import csv
from collections import MutableMapping
#depends on the env, could be
# from collections.abd import MutableMapping
import random
from tqdm import tqdm

import imagesize
import numpy as np
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


def split_featclarrays(pathtofolder: str, splitpourcent: float = 15., 
                       clarrayname: str = 'clarray',
                       featarrayname: str = 'featarray') -> list:
    """
    Split classification array (clarrays) and feature arrays (featarray), that are 
    outputs of Concatenate features and create Pandas DataFrames step of feature selection. 

    It will then create a training set and a test set for the binary classification of WSIs.

    Parameters:
    -----------
    pathtofolder: str

    splitpourcent: float, optionnal

    clarrayname: str, optionnal

    featarrayname: str, optionnal


    Returns:
    --------
    list_train_arrays: 

    list_test_arrays


    """
    # Load data
    ext = '.npy'
    clarray = np.load(pathtofolder  + clarrayname + ext)
    featarray = np.load(pathtofolder + featarrayname + ext) 
    # Define number of WSI represented by the arrays and how much to split (splitpourcent)
    totnbr_wsi = len(clarray)
    nbrwsi2split = int(totnbr_wsi * (splitpourcent/100))
    # Next step, create a list with  the indexes to remove from clarray and featarray
    # Avoid list comprehension in the next line as we need to check the list itself 
    # (. The issue here is that indexlist is not defined yet when you use it within the list comprehension. 
    # Therefore, you cannot check for membership in indexlist at that point.)
    indexlist = []
    while len(indexlist) < nbrwsi2split:
        random_index = random.randint(0, totnbr_wsi - 1)
        if random_index not in indexlist:
            indexlist.append(random_index)

    # VERY IMPORTATNT Remark!:
    # Sort the indices in descending order so that you remove elements from the end to avoid index shifting
    indexlist.sort(reverse=True)
    
    ## Now split clarray into a test and train clarrays
    cllist = list(clarray)
    testcllist = list()
    # Pruning process and append to the test list
    print('Splitting classification array...')
    for index in tqdm(indexlist):
        removedelement = cllist.pop(index)
        testcllist.append(removedelement)
    #Generate the training classifications list from the pruned cllist:
    traincllist = cllist
    # Convert the list to numpy arrays
    testclarray = np.asarray(testcllist)
    trainclarray = np.asarray(traincllist)

    ## Now split featarray into a test and train featarrays
    # featlist = list(featarray)
    testfeatlist = list()
    trainfeatarray = featarray
    # Pruning process and append to the test list
    print('Splitting feature matrix...')
    for index in tqdm(indexlist):
        testfeatlist.append(featarray[:, index])
        trainfeatarray = np.delete(trainfeatarray, index, axis=1)
        # removedcolumn = [row.pop(indexlist) for row in featlist]

    #Generate the test classifications array from the list:
    #Don't forget to transpose it is necessary here!!!
    testfeatarray = np.transpose(np.asarray(testfeatlist))
    #Create one list for training data and one list for test data 
    list_test_arrays = [testfeatarray, testclarray]
    list_train_arrays = [trainfeatarray, trainclarray]

    return list_train_arrays, list_test_arrays



def noheadercsv_to_dict(file_path):
    """
    Create a dictionnary from a csv file with 2 columns:
    Generate first column items as keys and second column items as values
    """
    data_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_dict[row[0]] = row[1]  
    return data_dict



### Utils Classes

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
