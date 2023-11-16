#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import json
import os
import csv

import numpy as np
import yaml
from attrdict import AttrDict as attributedict

from src.histo_miner.feature_selection import FeatureSelector
from src.histo_miner.utils.misc import convert_flatten, convert_flatten_redundant, noheadercsv_to_dict
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
patientid_csv = config.paths.files.patientid_csv
patientid_disp = config.parameters.bool.patientid_disp
nbr_keptfeat = config.parameters.int.nbr_keptfeat
boruta_max_depth = config.parameters.int.boruta_max_depth
boruta_random_state = config.parameters.int.nbr_keptfeat



#####################################################################
## Concatenate features and create Pandas DataFrames
## If applicable, create a list of Patient ID keeping feature columns order
#####################################################################

"""Concatenate the quantification features all together 
in pandas DataFrame. Create a corresponding list of patient ID if provided"""


###### Reorganise the folder and naming of files to process the concatenation of feature
tissueanalyser_folder = pathtofolder + '/' + 'tissue_analyses_sorted'
norec_analyse_folder = tissueanalyser_folder + '/' + 'no_recurrence'
rec_analyse_folder = tissueanalyser_folder + '/' + 'recurrence'

# If the 2 folders norec_analyse_folder and rec_analyse_folder don't exsit, 
# or they exist but are empty, we start the re-organization of the files with anaylser2featselect.
if (not os.path.exists(norec_analyse_folder) and not os.path.exists(rec_analyse_folder)) or \
    ((os.path.exists(norec_analyse_folder) and not os.listdir(norec_analyse_folder)) and \
    (os.path.exists(rec_analyse_folder) and not os.listdir(rec_analyse_folder))):
    anaylser2featselect(pathtofolder)
else:
    print('\nRe-organization of the files already performed, so moving the files skipped')


########  Create list with the paths of the files to analyse
pathto_sortedfolder = pathtofolder + '/' + 'tissue_analyses_sorted'
jsonfiles = list()
cllist = list()
# cllist stands for for classification list (recurrence (1) or norecurrence (0))
for root, dirs, files in os.walk(pathto_sortedfolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a srting. You create a string with this:
        for file in files:
            namewoext, extension = os.path.splitext(file)
            filepath = root + '/' + file
            if extension == '.json' and 'analysed' in namewoext:
                if not 'recurrence' in namewoext:
                    raise ValueError('Some features are not associated to a recurrence '
                                     'or norecurrence WSI classification. User must sort JSON and rename it'
                                     ' with the corresponding reccurence and noreccurence caracters')
                else:
                    jsonfiles.append(filepath)



####### If applicable create a dict file from the patient ID csv file 
# And initializa the futur ordered patient ID list
if patientid_disp:
    patientid_dict = noheadercsv_to_dict(patientid_csv)
    patientid_list = list()

######## Process the files
print('Detected {} json files.'.format(len(jsonfiles)))
# If applicable, create a list of patient ID
feature_init = True
# Sort the list by name of files
jsonfiles.sort()
for jsonfile in jsonfiles:
    with open(jsonfile, 'r') as filename:
        pathwoext = os.path.splitext(jsonfile)[0]
        path_to_parentfolder, nameoffile = os.path.split(pathwoext)

        # If applicable, create a list of patient ID with same order of feature array and clarray
        if patientid_disp:
            if 'recurrence' in nameoffile:
                if 'no_recurrence' in nameoffile:
                    namesimplified = nameoffile.replace('_no_recurrence_analysed','')
                else:
                    namesimplified = nameoffile.replace('_recurrence_analysed','')
            patientid_list.append(patientid_dict.get(namesimplified))

        # extract information of the JSON as a string
        analysisdata = filename.read()
        # read JSON formatted string and convert it to a dict
        analysisdata = json.loads(analysisdata)
        # flatten the dict (with redundant keys in nested dict, see function)
        analysisdataflat = convert_flatten_redundant(analysisdata)
        analysisdataflat = {k: v for (k, v) in analysisdataflat.items() if v != 'Not calculated'}

        #Convert dict values into an array
        valuearray = np.fromiter(analysisdataflat.values(), dtype=float)
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

    if  feature_init:
        #Create a list of names of the features, (only done once as all the json have the same features)
        #We create a new dictionnary that is using not the same keys name, but simplified ones.
        #Check the difference between convert_flatten and convert_flatten_redundant docstrings
        simplifieddata =  convert_flatten(analysisdata)
        featnameslist = list(simplifieddata.keys())
        # The first row of the featarray is not concatenated to anything so we have an initialiation
        #step that is slightly different than the other steps.
        featarray = valuearray
        feature_init = False
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

## mr.MR calculations
print('mR.MR calculations (see https://github.com/smazzanti/mrmr to have more info) '
      'in progress...')
selfeat_mrmr_index, mrmr_relevance_matrix, mrmr_redundancy_matrix = FeatureSelector.run_mrmr(nbr_keptfeat)
# Now associate the index of selected features (selfeat_mrmr_index) to the list of names:
selfeat_mrmr_names = [featnameslist[index] for index in selfeat_mrmr_index] 

print('Selected Features Indexes: {}'.format(selfeat_mrmr_index))
print('Selected Features Names: {}'.format(selfeat_mrmr_names))
# print('Relevance Matrix: {}'.format(mrmr_relevance_matrix))
# print('Redundancy Matrix: {}'.format(mrmr_redundancy_matrix))

## Boruta calculations
print('Boruta calculations  (https://github.com/scikit-learn-contrib/boruta_py to have more info)'
      ' in progress...')
selfeat_boruta_index = FeatureSelector.run_boruta(
    max_depth=boruta_max_depth, random_state=boruta_random_state)
# Now associate the index of selected features (selfeat_boruta_index) to the list of names:
selfeat_boruta_names = [featnameslist[index] for index in selfeat_boruta_index] 

print('Selected Features Indexes: {}'.format(selfeat_boruta_index))
print('Selected Features Names: {}'.format(selfeat_boruta_names))

## Mann Whitney U calculations
print('Mann Whitney U calculations in progress...')
selfeat_mannwhitneyu_index, orderedp_mannwhitneyu = FeatureSelector.run_mannwhitney(nbr_keptfeat)
# Now associate the index of selected features (selfeat_mannwhitneyu_index) to the list of names:
selfeat_mannwhitneyu_names = [featnameslist[index] for index in selfeat_mannwhitneyu_index] 

print('Selected Features Indexes: {}'.format(selfeat_mannwhitneyu_index))
print('Selected Features Names: {}'.format(selfeat_mannwhitneyu_names))
print('Output Ordered from best p-values to worst: {}'.format(orderedp_mannwhitneyu))

print('feature selection finished')
print('***** \n')



############################################################
## Save all numpy files
############################################################

# Save all the files in the tissue analyses folder
# Create the path to folder that will contain the numpy feature selection files

pathoutput = pathtofolder + '/feature_selection/'
ext = '.npy'

# If the folder doesn't exist create it
if not os.path.exists(pathoutput):
    os.makedirs(pathoutput)


print('Saving feature array, classification array, summary of selected features for each methods,'
       ' as well as methods output in numpy format to be used for classification step...')
# Create text files with name and indexes of selected feature for 
# every method
summarystr_mrmr = '\n\nFor mR.MR calculations:\n' \
                  + str(selfeat_mrmr_index) + '\n' \
                  + str(selfeat_mrmr_names)

summarystr_boruta = '\n\nFor boruta calculations:\n' \
                  + str(selfeat_boruta_index) + '\n' \
                  + str(selfeat_boruta_names)

summarystr_mannwhitneyu = '\n\nFor Mann Whitney U calculations:\n' \
                          + str(selfeat_mannwhitneyu_index) + '\n' \
                          + str(selfeat_mannwhitneyu_names)

summarystr = summarystr_mrmr + summarystr_boruta + summarystr_mannwhitneyu

# Open the file for writing and save it
summaryfile_path = pathoutput + 'selected_features.txt'
with open(summaryfile_path, "w") as file:
    file.write(summarystr)

#Save feature array and classification array
pathfeatarray = pathoutput + 'featarray' + ext
np.save(pathfeatarray, featarray)
pathclarray = pathoutput + 'clarray' + ext
np.save(pathclarray, clarray)

# If applicable, save the patient_ID list as a ids array:
if patientid_disp:
    patientid_array = np.asarray(patientid_list)
    pathpatientids =  pathoutput + 'patientids' + ext
    np.save(pathpatientids, patientid_array)

# We save the index of selected features for mrmr and mannwhitneyu 
pathselfeat_mrmr = pathoutput + 'selfeat_mrmr_idx' + ext
np.save(pathselfeat_mrmr, selfeat_mrmr_index)
pathselfeat_boruta = pathoutput + 'selfeat_boruta_idx' + ext
np.save(pathselfeat_boruta, selfeat_boruta_index)
pathorderedp_mannwhitneyu = pathoutput + 'selfeat_mannwhitneyu_idx' + ext
np.save(pathorderedp_mannwhitneyu, selfeat_mannwhitneyu_index)


print('Saving done.')
print('Path to the output files: {}'.format(pathoutput))


##### DEV

# ###################################################################
# ## Calculate correlation matrix
# ###################################################################

# path_relevancemat = pathoutput + 'relevance_matrix' + ext
# np.save(path_relevancemat, mrmr_relevance_matrix)
# path_redundancymat = pathoutput + 'redundancy_matrix' + ext
# np.save(path_redundancymat, mrmr_redundancy_matrix)

# path_relevancemat_csv= pathoutput + 'relevance_matrix' + '.csv'
# np.savetxt(path_relevancemat_csv, mrmr_relevance_matrix,  delimiter=",")
# path_redundancymat_csv = pathoutput + 'redundancy_matrix' + '.csv'
# np.savetxt(path_redundancymat_csv, mrmr_redundancy_matrix,  delimiter=",")


corrmat =  np.corrcoef(featarray, clarray)
path_corrmat = pathoutput + 'correlation_matrix' + ext
np.save(path_corrmat, corrmat)
path_corrmat_csv = pathoutput + 'correlation_matrix.csv' 
np.savetxt(path_corrmat_csv, corrmat, delimiter=",")
