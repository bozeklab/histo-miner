#Lucas Sancéré -

import json
import os
import csv
from collections.abc import MutableMapping
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


def rename_with_parent(nested_dict: dict, key_list: list) -> dict:
    """
    Rename the last key in nested dictionaries if the key matches one of the names in a given list.

    This function traverses a nested dictionary, and for each key-value pair, if the key matches 
    a name in the provided list, the key is renamed by concatenating the parent dictionary's key 
    with the current key. The function returns a new dictionary with the modifications.

    Examples:
    - Given the following input dictionary:
    {
        'level1': {
            'level2a': {
                'key1': 'value1',
                'key_to_rename': 'value2'
            },
            'level2b': {
                'key2': 'value3'
            }
        }
    }
    - And the following list of keys to rename:
    ['key_to_rename']
    - The output dictionary will be:
    {
        'level1': {
            'level2a': {
                'key1': 'value1',
                'level2a_key_to_rename': 'value2'
            },
            'level2b': {
                'key2': 'value3'
            }
        }
    }

    Parameters:
    -----------
    nested_dict: dict
        The nested dictionary in which keys will be checked and potentially renamed.
    key_list: list
        A list of keys that should be renamed if they are the last key in a nested dictionary.

    Returns:
    --------
    dict
        A new dictionary with the renamed keys where applicable.
    """
    def recurse(d, parent_key=''):
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, MutableMapping):
                new_dict[k] = recurse(v, k)
            else:
                if k in key_list:
                    new_key = parent_key + '_' + k
                    new_dict[new_key] = v
                else:
                    new_dict[k] = v
        return new_dict

    return recurse(nested_dict)



def rename_with_ancestors(nested_dict: dict, key_list: list) -> dict:
    """
    Rename the last key in nested dictionaries if the key matches one of the names in a given list.

    This function traverses a nested dictionary, and for each key-value pair, if the key matches 
    a name in the provided list, the key is renamed by concatenating the grandparent's key, the parent's key,
    and the current key. The function returns a new dictionary with the modifications.

    Examples:
    - Given the following input dictionary:
    {
        'level1': {
            'level2a': {
                'key1': 'value1',
                'key_to_rename': 'value2'
            },
            'level2b': {
                'key2': 'value3'
            }
        }
    }
    - And the following list of keys to rename:
    ['key_to_rename']
    - The output dictionary will be:
    {
        'level1': {
            'level2a': {
                'key1': 'value1',
                'level1_level2a_key_to_rename': 'value2'
            },
            'level2b': {
                'key2': 'value3'
            }
        }
    }

    Parameters:
    -----------
    nested_dict: dict
        The nested dictionary in which keys will be checked and potentially renamed.
    key_list: list
        A list of keys that should be renamed if they are the last key in a nested dictionary.

    Returns:
    --------
    dict
        A new dictionary with the renamed keys where applicable.
    """
    def recurse(d, grandparent_key='', parent_key=''):
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, MutableMapping):
                new_dict[k] = recurse(v, parent_key=k, grandparent_key=parent_key)
            else:
                if k in key_list:
                    new_key = grandparent_key + '_' + parent_key + '_' + k
                    new_dict[new_key] = v
                else:
                    new_dict[k] = v
        return new_dict

    return recurse(nested_dict)


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



def noheadercsv_to_dict(file_path: str):
    """
    Create a dictionnary from a csv file with 2 columns:
    Generate first column items as keys and second column items as values

    Parameters:
    -----------
    file_path: str
        Path to the csv file to process.
    """
    data_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_dict[row[0]] = row[1]  
    return data_dict



def convert_names_to_integers(name_list: list):
    """
    Convert a list of names into integers, 
    ensuring identical names have the same integer representation.

    Parameters:
    -----------
    name_list: list
        A list of names to be converted.

    Returns:
    -----------
    results: list
        A list of integers representing the names in the same order as the input list.
    """
    name_to_integer = {}
    for name in name_list:
        # Use the hash function to generate an integer representation for the name
        name_integer = hash(name)

        # Store the mapping between the name and its integer representation
        name_to_integer[name] = name_integer

    # Create a list of integers based on the original order of names
    result = [name_to_integer[name] for name in name_list]

    return result



def convert_names_to_orderedint(name_list: list):
    """
    Convert a list of names into integers, 
    ensuring identical names have the same integer representation.

    Parameters:
    -----------
    name_list: list
        A list of names to be converted.

    Returns:
    -----------
    results: list
        A list of ordered integers from 1 to N
        representing the names in the same order as the input list.
    """
    mapping = {}
    current_integer = 1
    patientids_ordered = []

    for num in name_list:
        if num not in mapping:
            mapping[num] = current_integer
            current_integer += 1
        patientids_ordered.append(mapping[num])

    return patientids_ordered


def get_indices_by_value(lst: list ):
    """
    Self explanatory
    """
    indices_by_value = {}

    for i, value in enumerate(lst):
        if value in indices_by_value:
            indices_by_value[value].append(i)
        else:
            indices_by_value[value] = [i]

    return indices_by_value


def find_closest_sublist(nested_list: list, nbr_kept_feat: int):
    """
    Self explanatory
    """
    closest_sublist = None
    smallest_difference = float('inf')  # Initialize to a very large number

    for sublist in nested_list:
        difference = abs(len(sublist) - nbr_kept_feat)
        if difference < smallest_difference:
            smallest_difference = difference
            closest_sublist = sublist

    return closest_sublist



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
