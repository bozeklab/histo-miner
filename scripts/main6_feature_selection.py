#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import os

import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict

from src.histo_miner.feature_selection import FeatureSelector

###################################################################
## Load configs parameter
###################################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathfeatselect = config.paths.folders.feature_selection_main
pathfeatselectout = config.paths.folders.feature_selection_output
patientid_avail = config.parameters.bool.patientid_avail
nbr_keptfeat = config.parameters.int.nbr_keptfeat
boruta_max_depth = config.parameters.int.boruta_max_depth
boruta_random_state = config.parameters.int.boruta_random_state



####################################################################
## Load classification arrays, feature vector and patient IDs
####################################################################

ext = '.npy'

pathfeatarray = pathfeatselect + 'repslidesx_featarray' + ext
pathclarray = pathfeatselect + 'repslidesx_clarray' + ext
pathfeatnames = pathfeatselect + 'featnames' + ext
featarray = np.load(pathfeatarray)
clarray = np.load(pathclarray)
featnames = np.load(pathfeatnames)
featnameslist = list(featnames)

# If applicable, save the patient_ID list as a ids array:
if patientid_avail:
    pathpatientids = pathfeatselect + 'patientids' + ext


print("Feature Matrix Shape is", featarray.shape)
print("Feature Matrix is", featarray)
print("Classification Vector is", clarray)



###################################################################
## Run Feature Selections
###################################################################

feature_selector = FeatureSelector(featarray, clarray)

## mr.MR calculations
print('mR.MR calculations (see https://github.com/smazzanti/mrmr to have more info) '
      'in progress...')
selfeat_mrmr_index, mrmr_relevance_matrix, mrmr_redundancy_matrix = feature_selector.run_mrmr(nbr_keptfeat)
# Now associate the index of selected features (selfeat_mrmr_index) to the list of names:
selfeat_mrmr_names = [featnameslist[index] for index in selfeat_mrmr_index] 

print('Selected Features Indexes: {}'.format(selfeat_mrmr_index))
print('Selected Features Names: {}'.format(selfeat_mrmr_names))
print('Relevance Matrix: {}'.format(mrmr_relevance_matrix))
print('Redundancy Matrix: {}'.format(mrmr_redundancy_matrix))

## Boruta calculations
print('Boruta calculations  (https://github.com/scikit-learn-contrib/boruta_py to have more info)'
      ' in progress...')
selfeat_boruta_index = feature_selector.run_boruta(
    max_depth=boruta_max_depth, random_state=boruta_random_state)
# Now associate the index of selected features (selfeat_boruta_index) to the list of names:
selfeat_boruta_names = [featnameslist[index] for index in selfeat_boruta_index] 
print('Selected Features Indexes: {}'.format(selfeat_boruta_index))
print('Selected Features Names: {}'.format(selfeat_boruta_names))

## Mann Whitney U calculations
print('Mann Whitney U calculations in progress...')
selfeat_mannwhitneyu_index, orderedp_mannwhitneyu = feature_selector.run_mannwhitney(nbr_keptfeat)
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

pathoutput = pathfeatselectout 
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
                  + str(selfeat_mrmr_names) \
                  + str(mrmr_relevance_matrix)

summarystr_boruta = '\n\nFor boruta calculations:\n' \
                  + str(selfeat_boruta_index) + '\n' \
                  + str(selfeat_boruta_names)

summarystr_mannwhitneyu = '\n\nFor Mann Whitney U calculations:\n' \
                          + str(selfeat_mannwhitneyu_index) + '\n' \
                          + str(selfeat_mannwhitneyu_names) \
                          + str(orderedp_mannwhitneyu)

# Was edited
summarystr = summarystr_mrmr + summarystr_boruta + summarystr_mannwhitneyu

# Open the file for writing and save it
summaryfile_path = pathoutput + 'selected_features.txt'
with open(summaryfile_path, "w") as file:
    file.write(summarystr)

# We save the index of selected features for mrmr and mannwhitneyu 
pathselfeat_mrmr = pathoutput + 'selfeat_mrmr_idx' + ext
np.save(pathselfeat_mrmr, selfeat_mrmr_index)
pathselfeat_boruta = pathoutput + 'selfeat_boruta_idx_depth20' + ext
np.save(pathselfeat_boruta, selfeat_boruta_index)
pathorderedp_mannwhitneyu = pathoutput + 'selfeat_mannwhitneyu_idx' + ext
np.save(pathorderedp_mannwhitneyu, selfeat_mannwhitneyu_index)


print('Saving done.')
print('Path to the output files: {}'.format(pathoutput))











# NOT TO KEEP For publicaton
# # Calculate Pearson correlation for no-recurrence class
# featarray_noreconly = [
#     featarray[colindex] for colindex in  range(0, featarray.shape[0]) if clarray[colindex] == 0
#     ]
# corrmat_norec =  np.corrcoef(featarray_noreconly)
# path_corrmat_norec = pathoutput + 'norecurrence_correlation_matrix' + ext
# np.save(path_corrmat_norec, corrmat_norec)
# path_corrmat_norec_csv = pathoutput + 'norecurrence_correlation_matrix.csv' 
# np.savetxt(path_corrmat_norec_csv, corrmat_norec, delimiter=",")


# # Calculate Person correlation for recurrence class
# featarray_reconly = [
#     featarray[colindex] for colindex in  range(0, featarray.shape[0]) if clarray[colindex] == 1 
#     ]
# corrmat_rec =  np.corrcoef(featarray_noreconly)
# path_corrmat_rec = pathoutput + 'recurrence_correlation_matrix' + ext
# np.save(path_corrmat_rec, corrmat_rec)
# path_corrmat_rec_csv = pathoutput + 'recurrence_correlation_matrix.csv' 
# np.savetxt(path_corrmat_rec_csv, corrmat_rec, delimiter=",")
