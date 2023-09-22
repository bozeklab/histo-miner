#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import json
import os

import yaml
from attrdict import AttrDict as attributedict
# import numpy as np
# import scipy.stats

from src.histo_miner import hovernet_utils, segmenter_utils
from src.histo_miner.utils import cellclass_process


#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.main
maskmap_downfactor = config.parameters.int.maskmap_downfactor
hovernet_mode = str(config.names.hovernet_mode)
values2change = list(config.parameters.lists.values2change)
newvalues = list(config.parameters.lists.newvalues)


str2replace_tilemode = config.names.managment.str2replace_tilemode
newstr_tilemode = config.names.managment.newstr_tilemode
str2replace2_tilemode = config.names.managment.str2replace2_tilemode
newstr2_tilemode = config.names.managment.newstr2_tilemode
str2replace_wsimode = config.names.managment.str2replace_wsimode
newstr_wsimode = config.names.managment.newstr_wsimode
str2replace2_wsimode = config.names.managment.str2replace2_wsimode
newstr2_wsimode = config.names.managment.newstr2_wsimode



#############################################################
## Processing of inferences results for Tissue Analyser
#############################################################

"""Update each json files to be compatible with Tissue Analyser AND QuPath"""

"""Update each mask output """


#Check if user needs to run the processing of inference 
# runpreprocessing = input(
#     "Update inference outputs to be compatible with Tissue Analyser AND QuPath? \n"
#     "Type 'yes' (Recommanded) or 'no' "
#     "(User should enter no ONLY if the processing was already done):")

# if str(runpreprocessing) != 'yes'and str(runpreprocessing) != 'no':
#     raise ValueError('User should input yes or no.')
#     runpreprocessing = input(
#     "Update inference outputs to be compatible with Tissue Analyser AND QuPath? \n"
#     "Type 'yes' or 'no' (User should enter no ONLY if the processing was already done):")

# Load each parametes as define in ManageJSON script
if hovernet_mode == 'tile':
    string2replace = str(str2replace_tilemode)
    newstring = str(newstr_tilemode)
    string2replace2 = str(str2replace2_tilemode)
    newstring2 = str(newstr2_tilemode)
elif hovernet_mode == 'wsi':
    string2replace = str(str2replace_wsimode)
    newstring = str(newstr_wsimode)
    string2replace2 = str(str2replace2_wsimode)
    newstring2 = str(newstr2_wsimode)
else:
    print("hovernet_mode string is not correct, tile mode choosen by default")
    string2replace = str(str2replace_tilemode)
    newstring = str(newstr_tilemode)
    string2replace2 = str(str2replace2_tilemode)
    newstring2 = str(newstr2_tilemode)


print('The folders must contains only hovernet predictions and segmenter predictions files.')
for root, dirs, files in os.walk(pathtofolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a srting. You create a string with this:
        for file in files:
            path, extension = os.path.splitext(file)
            # Update each JSON file
            filepath = root + '/' + file
            # Knowing that root is the path to the directory of the selected file,
            # root + file is the complete path
            if extension == '.json':
                print('Detected json file:', file)
                print('Path to file :', filepath)
                hovernet_utils.replacestring_json(filepath, string2replace,
                                                  newstring, string2replace2,
                                                  newstring2)
                maskmappath = jsonfilepath.replace(extension, '.png')
                cellclass_process.update_cellclass(filepath, maskmappath, 
                                                   maskmapdownfactor=maskmap_downfactor)
                print('Updated json')
            if extension != '.json':
                print('Detected mask file '
                      '(has to be in a pillow supported format - like .png)', file)
                print('Path to file :', filepath)
                segmenter_utils.change_pix_values(filepath, values2change,
                                                  newvalues)
                print('Updated mask file')



print('All json files updated with mode {}'.format(hovernet_mode))
print('All mask files updated')


