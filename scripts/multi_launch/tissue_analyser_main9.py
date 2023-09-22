#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import json
import os

import yaml
from attrdict import AttrDict as attributedict

from src.histo_miner import hovernet_utils, segmenter_utils
from src.histo_miner import tissue_analyser as analyser
from src.utils.misc import NpEncoder


# import numpy as np
# import scipy.stats


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
selectedcls_ratio = list(config.parameters.lists.selectedcls_ratio)
selectedcls_dist = list(config.parameters.lists.selectedcls_dist)
classnames = list(config.parameters.lists.classnames)
classnames_injson = list(config.parameters.lists.classnames_injson)

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
runpreprocessing = input(
    "Update inference outputs to be compatible with Tissue Analyser AND QuPath? \n"
    "Type 'yes' (Recommanded) or 'no' "
    "(User should enter no ONLY if the processing was already done):")

if str(runpreprocessing) != 'yes'and str(runpreprocessing) != 'no':
    raise ValueError('User should input yes or no.')
    runpreprocessing = input(
    "Update inference outputs to be compatible with Tissue Analyser AND QuPath? \n"
    "Type 'yes' or 'no' (User should enter no ONLY if the processing was already done):")

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



if str(runpreprocessing) == 'yes':

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
                    pathtofolder, filename = os.path.split(file)
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


#############################################################
## Tissue Analysis, Extraction of features
#############################################################

"""Tissue Analysis"""

for root, dirs, files in os.walk(pathtofolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a srting. You create a string with this:
        for file in files:
            path, extension = os.path.splitext(file)
            path_to_parentfolder, nameoffile = os.path.split(path)
            if extension == '.json' and not any(keyword in nameoffile for keyword in ['data', 'analysed']):
                # 'data' not in name of files means it is not a json file coming from the analysis
                if os.path.exists(pathtofolder + '/' + nameoffile + '_analysed.json'):
                    print('Detected an already processed file:', nameoffile)
                    continue
                else:
                    print('***** \nDetected hovernet output json file:', file)
                    # Knowing that root is the path to the directory of the selected file,
                    # root + file is the complete path
                    # Creating the dictionnary to count the cells using countjson function
                    jsonfilepath = root + '/' + file
                    print('Process count of cells per cell type in the whole slide image...')
                    classcountsdict = analyser.counthvnjson(
                                                 jsonfilepath, 
                                                 classnames_injson, 
                                                 classnameaskey=classnames)
                    allcells_in_wsi_dict = classcountsdict
                    print('Allcells_inWSI_dict generated as follow:', allcells_in_wsi_dict)
                    # Create the path to Mask map binarized and Class JSON and save it into a variable
                    if os.path.exists(jsonfilepath) and os.path.exists(jsonfilepath.replace(extension, '.png')):

                        # Create path for the maskmap
                        maskmappath = jsonfilepath.replace(extension, '.png')
                        print('Detected mask file:', maskmappath)

                        # Analysis
                        tumor_tot_area = analyser.count_pix_value(maskmappath, 255) * (maskmap_downfactor ** 2)
                        print('Process cells identification '
                              '(number of cells and tot area of cells) inside tumor regions...')

                        cells_inmask_dict = analyser.cells_insidemask_classjson(
                                                    maskmappath, 
                                                    jsonfilepath, 
                                                    selectedcls_ratio,
                                                    maskmapdownfactor=maskmap_downfactor,
                                                    classnameaskey=classnames)

                        print('Cells_inmask_dict generated as follow:', cells_inmask_dict)
                        print('Process distance calculcations inside tumor regions...')

                        cellsdist_inmask_dict = analyser.mpcell2celldist_classjson(
                            jsonfilepath, 
                            selectedcls_dist,
                            cellfilter='Tumor',
                            maskmap=maskmappath,
                            maskmapdownfactor=maskmap_downfactor,
                            tumormargin=None)

                        print('Cellsdist_inmask_dict generated as follow:', cellsdist_inmask_dict)

                    else:
                        cells_inmask_dict = None
                        cellsdist_inmask_dict = None
                        print('Cellsratio_inmask_dict not generated')
                        print('Cellsdist_inmask_dict not generated')

                    jsondata = analyser.hvn_outputproperties(
                                                    allcells_in_wsi_dict,
                                                    cells_inmask_dict, 
                                                    cellsdist_inmask_dict,
                                                    masktype='Tumor',
                                                    areaofmask=tumor_tot_area)

                    # Write information inside a json file and save it
                    with open(pathtofolder + '/' + nameoffile + '_analysed.json', 'w') as outfile:
                        json.dump(jsondata, outfile, cls=NpEncoder)

                    print('Json file written :', path_to_parentfolder + nameoffile + '_analysed.json \n*****')

print('Tissue Analysis Done')
