#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import json
import os

import yaml
from attrdictionary import AttrDict as attributedict
from tqdm import tqdm
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
pathtofolder = config.paths.folders.inferences_postproc_main 
maskmap_downfactor = config.parameters.int.maskmap_downfactor
maskmapext = str(config.parameters.str.maskmap_ext)
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
## Update parameters depending on hovernet_mode
#############################################################
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



#############################################################
## Processing of inferences results for Tissue Analyser
#############################################################

"""Update each mask output and  each json files to be compatible with Tissue Analyser AND QuPath"""


########  Create lists with the paths of the files to process
jsonfiles = list()
maskfiles = list()
for root, dirs, files in os.walk(pathtofolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a srting. You create a string with this:
        for file in files:
            namewoext, extension = os.path.splitext(file)
            filepath = root + '/' + file
            # Knowing that root is the path to the directory of the selected file,
            # root + file is the complete path
            if extension == '.json':
                print('Detected hovernet output json file:', file)
                jsonfiles.append(filepath)
            if extension == maskmapext:
                print('Detected segmenter output file:', file)
                maskfiles.append(filepath)


######## Process the files
# The masks have to be updated BEFORE the json files
print('Update of the mask files...')
print('Mask files have to be in a pillow supported format (like .png)')
for maskfile in tqdm(maskfiles):
    segmenter_utils.change_pix_values(maskfile, values2change, newvalues)

#Update of the jsons
print('Update of the json files...')
for jsonfile in tqdm(jsonfiles):
    hovernet_utils.replacestring_json(jsonfile, string2replace,
                                      newstring, string2replace2,
                                      newstring2)
    maskmappath = jsonfile.replace('.json', '.png')
    cellclass_process.update_cellclass(jsonfile, maskmappath, 
                                       maskmapdownfactor=maskmap_downfactor)

print('All mask files updated')
print('All json files updated with mode {}'.format(hovernet_mode))






