#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import json
import os

import numpy as np
import yaml
from attrdict import AttrDict as attributedict

from src.histo_miner.feature_selection import FeatureSelector
from src.histo_miner.utils.misc import convert_flatten_redundant
from src.histo_miner.utils.filemanagment import anaylser2featselect



###################################################################
## Load configs parameter
###################################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.feature_selection_main
nbr_keptfeat = config.parameters.int.nbr_keptfeat


###################################################################
## Concatenate features and create Pandas DataFrames
###################################################################


"""Concatenate the quantification features all together 
in pandas DataFrame and run the different feature selections"""


####### TO ADD AS ?
# Associate the index of the selected feature to the name of it (use dict probably)


###### WRITE COMMENT about what is anaylser2featselect
anaylser2featselect(pathtofolder)


########  Create list with the paths of the files to analyse
pathto_sortedfolder = pathtofolder + '/' + 'tissue_analysis_sorted'
jsonfiles = list()
cllist = list()
# cllist stands for for classification list (recurrence (1) or norecurrence (0))
for root, dirs, files in os.walk(pathto_sortedfolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a srting. You create a string with this:
        for file in files:
            path, extension = os.path.splitext(file)
            path_to_parentfolder, nameoffile = os.path.split(path)
            if extension == '.json' and 'analysed' in nameoffile:
                print('Detected tissue analysis json file:', file)
                if not 'recurrence' in nameoffile:
                    raise ValueError('Some features are not associated to a recurrence '
                                    'or norecurrence WSI classification. User must sort JSON and rename it'
                                    ' with the corresponding reccurence and noreccurence caracters')
                else:
                    jsonfiles.append(file)


######## Process the files
feature_init = False
# Means Initialisation of feature array not done yet
for jsonfile in jsonfiles:
    with open(jsonfile, 'r') as filename:
        pathwoext = os.path.splitext(jsonfile)[0]
        path_to_parentfolder, nameoffile = os.path.split(pathwoext)

        # extract information of the JSON as a string
        analysisdata = filename.read()
        # read JSON formatted string and convert it to a dict
        analysisdata = json.loads(analysisdata)
        # flatten the dict (with redundant keys in nested dict, see function)
        analysisdata = convert_flatten_redundant(analysisdata)
        analysisdata = {k: v for (k, v) in analysisdata.items() if v != 'Not calculated'}

        #Convert dict values into an array
        valuearray = np.fromiter(analysisdata.values(), dtype=float)
        #Remove nans from arrays and add a second dimension to the array
        # in order to be concatenated later on
        valuearray = valuearray[~np.isnan(valuearray)]
        valuearray = np.expand_dims(valuearray, axis=1)

        # Generate the list of WSI binary classification
        # No list comprehension just to exhibit more clearly the error message
        if 'recurrence' in nameoffile:
            if 'no_recurrence' in nameoffile:
                cllist.append(int(0))
            else:
                cllist.append(int(1))
        else:
            raise ValueError('Some features are not associated to a recurrence '
                             'or norecurrence WSI classification. User must sort JSON and rename it'
                             'with the corresponding reccurence and noreccurence caracters'
                             'User can check src.utils.anaylser2featselect function for more details.')

if not feature_init:
    featarray = valuearray
    feature_init = True
else:
    #MEMORY CONSUMING, FIND BETTER WAY IF POSSIBLE
    featarray = np.concatenate((featarray, valuearray), axis=1)


# Check if there are recurrence and no recurrence data in the training set (both needed)
if cllist:
    if 0 not in cllist:
        raise ValueError(
            'The data contains only no recurrence data or json named as being norecurrence. '
            'To run statistical test we need both recurrence and norecurrence examples')

    if 1 not in cllist:
        raise ValueError(
            'The data contains only recurrence data or json named as being recurrence. '
            'To run statistical test we need both recurrence and norecurrence examples')

clarray = np.asarray(cllist)
# clarray stands for for classification array (recurrence or norecurrence)

##### Display Feature Matrix and its corresponding classification vectors (reccurence or no recurrence)
print("Feature Matrix Shape is", featarray.shape)
print("Feature Matrix is", featarray)
print("Classification Vector is", clarray)


###################################################################
## Run Feature Selections
###################################################################

FeatureSelector = FeatureSelector(featarray, clarray)

print('mR.MR calculations (see https://github.com/smazzanti/mrmr to have more info) '
      'in progress...')
selfeat_mrmr = FeatureSelector.run_mrmr(nbr_keptfeat)
print('Selected Features: {}'.format(selfeat_mrmr[0]))
print('Relevance Matrix: {}'.format(selfeat_mrmr[1]))
print('Redundancy Matrix: {}'.format(selfeat_mrmr[2]))

print('Boruta calculations  (https://github.com/scikit-learn-contrib/boruta_py to have more info)'
      ' in progress...')
selfeat_boruta = FeatureSelector.run_boruta()
print("Selected Feature Matrix Shape: {}".format(selfeat_boruta.shape))
print('Selected Features: {}'.format(selfeat_boruta))

print('Mann Whitney U calculations in progress...')
orderedp_mannwhitneyu = FeatureSelector.run_mannwhitney()
print('Output Ordered from best p-values to worst: {}'.format(orderedp_mannwhitneyu))

print('feature selection finished')
print('***** \n')


############################################################
## Save all numpy files
############################################################

# Save all the files in the tissue analysis folder
# Create the path to folder that will contain the numpy feature selection files
pathnumpy = pathtofolder.replace('tissue_analyses/', 'feature_selection/')
ext = '.npy'

# If the folder doesn't exist create it
if not os.path.exists(pathnumpy):
    os.makedirs(pathnumpy)


print('Save feature selection numpy files...')
pathfeatarray = pathnumpy + 'featarray' + ext
np.save(pathfeatarray, featarray)
pathclarray = pathnumpy + 'clarray' + ext
np.save(pathclarray, clarray)
pathselfeat_mrmr = pathnumpy + 'selfeat_mrmr' + ext
np.save(pathselfeat_mrmr, selfeat_mrmr)
pathselfeat_boruta = pathnumpy + 'selfeat_boruta' + ext
np.save(pathselfeat_boruta, selfeat_boruta)
pathorderedp_mannwhitneyu = pathnumpy + 'orderedp_mannwhitneyu' + ext
np.save(pathorderedp_mannwhitneyu, orderedp_mannwhitneyu)
print('Saving done.')