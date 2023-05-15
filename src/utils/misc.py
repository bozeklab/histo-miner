

from collections import MutableMapping


### Utils Functions

def convert_flatten_redundant(inputdic, parent_key='', sep='_'):
    """
    Create a flatten dictionnary, meaning from a dictionnary containing nested keys, it will generate a dictionnary with simple keys-values pairs.

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
    inputdic : dic
        dictionnary the user want to flatten
    sep : str
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


def convert_flatten(inputdic, parent_key=''):
    """
    Create a flatten dictionnary, meaning from a dictionnary containing nested keys, it will generate a dictionnary with simple keys-values pairs.

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
    inputdic : dic
        dictionnary the user want to flatten
    sep : str
        separation between nested key in the name of the newly created key
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