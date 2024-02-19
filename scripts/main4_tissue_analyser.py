#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import json
import os

import yaml
from attrdictionary import AttrDict as attributedict
# from tqdm import tqdm
# import numpy as np
# import scipy.stats

from src.histo_miner import tissue_analyser as analyser
from src.histo_miner.utils.misc import NpEncoder



###################################################################
## Load configs parameter
###################################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.tissue_analyser_main
calculate_distances = config.parameters.bool.calculate_distances
maskmap_downfactor = config.parameters.int.maskmap_downfactor
maskmapext = str(config.parameters.str.maskmap_ext)
selectedcls_ratio = list(config.parameters.lists.selectedcls_ratio)
selectedcls_dist = list(config.parameters.lists.selectedcls_dist)
classnames = list(config.parameters.lists.classnames)
classnames_injson = list(config.parameters.lists.classnames_injson)



#################################################################
## Tissue Analysis, Extraction of features
#################################################################

"""Tissue Analysis"""


########  Create list with the paths of the files to analyse
jsonfiles = list()
for root, dirs, files in os.walk(pathtofolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a single path string. Create a string with this:
        for file in files:
            namewoext, extension = os.path.splitext(file)
            filepath = root + '/' + file
            if extension == '.json' and not any(keyword in namewoext for keyword in ['data', 'analysed', 'cellnbr']):
                # 'data' not in name of files means it is not a json file coming from the analysis
                if os.path.exists(pathtofolder + '/' + namewoext + '_analysed.json'):
                    print('Detected an already processed file:', file)
                    continue
                else:
                    print('Detected hovernet output json file not already analysed:', file)
                    jsonfiles.append(filepath)



######## Process the files
for jsonfile in jsonfiles:
   # Creating the dictionnary to count the cells using countjson function:
    pathwoext = os.path.splitext(jsonfile)[0]

    print('********** \nProcess file:', pathwoext)
    print('Process count of cells per cell type in the whole slide image...')
    classcountsdict = analyser.counthvnjson(
                                 jsonfile, 
                                 classnames_injson, 
                                 classnameaskey=classnames)
    allcells_in_wsi_dict = classcountsdict
    print('\nAllcells_inWSI_dict generated as follow:', allcells_in_wsi_dict)


    # Write information inside a json file about cell nbr per class and save it
    with open(pathwoext + '_cellnbr.json', 'w') as outfile:
        json.dump(allcells_in_wsi_dict, outfile, cls=NpEncoder)

    print('Json file written :', pathwoext + '_cellnbr.json \n**********')


    # Create the path to Mask map binarized and Class JSON and save it into a variable
    if os.path.exists(jsonfile) and os.path.exists(jsonfile.replace('.json', maskmapext)):

        # Create path for the maskmap
        maskmappath = jsonfile.replace('.json', maskmapext)
        print('Detected mask file:', maskmappath)

        # Analysis
        tumor_tot_area = analyser.count_pix_value(maskmappath, 255) * (maskmap_downfactor ** 2)
        print('Process cells identification '
              '(number of cells and tot area of cells) inside tumor regions...')

        cells_inmask_dict = analyser.cells_insidemask_classjson(
                                    maskmappath, 
                                    jsonfile, 
                                    selectedcls_ratio,
                                    maskmapdownfactor=maskmap_downfactor,
                                    classnameaskey=classnames)

        print('Cells_inmask_dict generated as follow:', cells_inmask_dict)
        
        # Needs to create an empty dict even if the distances are not calculated
        cellsdist_inmask_dict = dict()
        if calculate_distances: 
            print('Process distance calculcations inside tumor regions...')  
            cellsdist_inmask_dict = analyser.mpcell2celldist_classjson(
                jsonfile, 
                selectedcls_dist,
                cellfilter='Tumor',
                maskmap=maskmappath,
                maskmapdownfactor=maskmap_downfactor,
                tumormargin=None)

            print('Cellsdist_inmask_dict generated as follow:', cellsdist_inmask_dict)


    else:
        tumor_tot_area = 0
        cells_inmask_dict = None
        cellsdist_inmask_dict = None
        print('Cellsratio_inmask_dict not generated')
        print('Cellsdist_inmask_dict not generated')


    jsondata = analyser.hvn_outputproperties(
                                    allcells_in_wsi_dict,
                                    cells_inmask_dict, 
                                    cellsdist_inmask_dict,
                                    masktype='Tumor',
                                    areaofmask=tumor_tot_area, 
                                    selectedcls_ratio=selectedcls_ratio, 
                                    selectedcls_dist=selectedcls_dist)


    # Write information inside a json file and save it for the whole analysis
    with open(pathwoext + '_analysed.json', 'w') as outfile:
        json.dump(jsondata, outfile, cls=NpEncoder)

    print('Json file written :', pathwoext + '_analysed.json \n**********')


print('Tissue Analysis Done')
