
# To use only if the tissue analyses is not taking natural logarithm for ratios
# TO remove when code is published (put in archive)


#Lucas Sancéré -

import sys
sys.path.append('../../../')  # Only for Remote use on Clusters

import json
import os

import yaml
from attrdictionary import AttrDict as attributedict
from tqdm import tqdm
import numpy as np
import scipy.stats

from src.histo_miner import tissue_analyser as analyser
from src.histo_miner.utils.misc import NpEncoder



## BE EXTRA CAREFUL WITH THE PATHS IF YOU WANT TO USE IT!!


###################################################################
## Load configs parameter
###################################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.tissue_analyser_main


correction_num = 1

#################################################################
## Correct ratios in json
#################################################################

print('Running here:', pathtofolder)

########  Create list with the paths of the files to analyse
jsonfiles = list()
for root, dirs, files in os.walk(pathtofolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a single path string. Create a string with this:
        for file in files:
            namewoext, extension = os.path.splitext(file)
            filepath = root + '/' + file
            if extension == '.json' :
                   jsonfiles.append(filepath)


###### Now we edit the json and overwritte it
for jsonfile in jsonfiles:
    with open(jsonfile, 'r') as filename:
        # pathwoext = os.path.splitext(jsonfile)[0]
        # path_to_parentfolder, nameoffile = os.path.split(pathwoext)

        # extract information of the JSON as a string
        analysisdata = filename.read()
        # read JSON formatted string and convert it to a dict
        analysisdata = json.loads(analysisdata)

        if correction_num == 1:
	        # changed corresponding values:
	        ratio_key1 = analysisdata['CalculationsforWSI']['Ratios_between_cell_types_WSI']

	        ratio_key1['Ratio_Granulocytes_TumorCells'] = np.log(ratio_key1['Ratio_Granulocytes_TumorCells'])
	        ratio_key1['Ratio_Lymphocytes_TumorCells'] = np.log(ratio_key1['Ratio_Lymphocytes_TumorCells'])
	        ratio_key1['Ratio_PlasmaCells_TumorCells'] = np.log(ratio_key1['Ratio_PlasmaCells_TumorCells'])
	        ratio_key1['Ratio_StromaCells_TumorCells'] = np.log(ratio_key1['Ratio_StromaCells_TumorCells'])
	        ratio_key1['Ratio_EpithelialCells_TumorCells'] = np.log(ratio_key1['Ratio_EpithelialCells_TumorCells'])

	        ratio_key1['Ratio_Granulocytes_Lymphocytes'] = np.log(ratio_key1['Ratio_Granulocytes_Lymphocytes'])
	        ratio_key1['Ratio_PlasmaCells_Lymphocytes'] = np.log(ratio_key1['Ratio_PlasmaCells_Lymphocytes'])
	        ratio_key1['Ratio_StromaCells_Lymphocytes'] = np.log(ratio_key1['Ratio_StromaCells_Lymphocytes'])
	        ratio_key1['Ratio_EpithelialCells_Lymphocytes'] = np.log(ratio_key1['Ratio_EpithelialCells_Lymphocytes'])

	        ratio_key1['Ratio_Granulocytes_PlasmaCells'] = np.log(ratio_key1['Ratio_Granulocytes_PlasmaCells'])
	        ratio_key1['Ratio_StromaCells_PlasmaCells'] = np.log(ratio_key1['Ratio_StromaCells_PlasmaCells'])
	        ratio_key1['Ratio_EpithelialCells_PlasmaCells'] = np.log(ratio_key1['Ratio_EpithelialCells_PlasmaCells'])

	        ratio_key1['Ratio_StromaCells_Granulocytes'] = np.log(ratio_key1['Ratio_StromaCells_Granulocytes'])
	        ratio_key1['Ratio_EpithelialCells_Granulocytes'] = np.log(ratio_key1['Ratio_EpithelialCells_Granulocytes'])

	        ratio_key1['Ratio_EpithelialCells_StromalCells'] = np.log(ratio_key1['Ratio_EpithelialCells_StromalCells'])


	        ratio_key2 = analysisdata['CalculationsRatiosinsideTumor']['Ratios_between_cell_types_Tumor_Regions']

	        ratio_key2['Ratio_Granulocytes_TumorCells_inTumor'] = np.log(ratio_key2['Ratio_Granulocytes_TumorCells_inTumor'])
	        ratio_key2['Ratio_Lymphocytes_TumorCells_inTumor'] = np.log(ratio_key2['Ratio_Lymphocytes_TumorCells_inTumor'])
	        ratio_key2['Ratio_PlasmaCells_TumorCells_inTumor'] = np.log(ratio_key2['Ratio_PlasmaCells_TumorCells_inTumor'])
	        ratio_key2['Ratio_StromaCells_TumorCells_inTumor'] = np.log(ratio_key2['Ratio_StromaCells_TumorCells_inTumor'])

	        ratio_key2['Ratio_Granulocytes_Lymphocytes_inTumor'] = np.log(ratio_key2['Ratio_Granulocytes_Lymphocytes_inTumor'])
	        ratio_key2['Ratio_PlasmaCells_Lymphocytes_inTumor'] = np.log(ratio_key2['Ratio_PlasmaCells_Lymphocytes_inTumor'])
	        ratio_key2['Ratio_StromaCells_Lymphocytes_inTumor'] = np.log(ratio_key2['Ratio_StromaCells_Lymphocytes_inTumor'])

	        ratio_key2['Ratio_Granulocytes_PlasmaCells_inTumor'] = np.log(ratio_key2['Ratio_Granulocytes_PlasmaCells_inTumor'])
	        ratio_key2['Ratio_StromaCells_PlasmaCells_inTumor'] = np.log(ratio_key2['Ratio_StromaCells_PlasmaCells_inTumor'])
	      
	        ratio_key2['Ratio_StromaCells_Granulocytes_inTumor'] = np.log(ratio_key2['Ratio_StromaCells_Granulocytes_inTumor'])
       

        if correction_num == 2:
        	pass

        newjsondata = analysisdata


    with open(jsonfile, 'w') as outfile:
        json.dump(newjsondata, outfile, cls=NpEncoder)