#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import json
import os

import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict

from src.histo_miner.utils.misc import convert_flatten, convert_flatten_redundant, noheadercsv_to_dict, \
                                       convert_names_to_orderedint, get_indices_by_value
from src.histo_miner.utils.filemanagment import anaylser2featselect



###################################################################
## Load configs parameters
###################################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathanalyserout = config.paths.folders.tissue_analyser_output
pathfeatselect = config.paths.folders.feature_selection_main
patientid_csv = config.paths.files.patientid_csv
patientid_avail = config.parameters.bool.patientid_avail
perpatient_feat = config.parameters.bool.perpatient_feat
calculate_vicinity = config.parameters.bool.calculate_vicinity


#####################################################################
## Concatenate features and create Pandas DataFrames
#####################################################################


"""Concatenate the quantification features all together 
   in pandas DataFrame. Create a corresponding list of patient ID if provided.
   2 options are possible: 
        - have one feature column (feature vector) per WSI on the feature matrix
        - have one feature column (feature vector) per patient on the feature matrix
   If perpatient_feat config parameter == False only first option will be generated.
   If perpatient_feat config parameter == True, then all the feature matrices will be produced.
 """




###### Reorganise the folder and naming of files to process the concatenation of feature
tissueanalyser_folder = pathanalyserout 
norec_analyse_folder = tissueanalyser_folder + '/' + 'no_response'
rec_analyse_folder = tissueanalyser_folder + '/' + 'response' 



##### /!\   


# If the 2 folders norec_analyse_folder and rec_analyse_folder don't exsit, 
# or they exist but are empty, we start the re-organization of the files with anaylser2featselect.

# For now we keep this commented as it is a big risk to run it

# if (not os.path.exists(norec_analyse_folder) and not os.path.exists(rec_analyse_folder)) or \
#     ((os.path.exists(norec_analyse_folder) and not os.listdir(norec_analyse_folder)) and \
#     (os.path.exists(rec_analyse_folder) and not os.listdir(rec_analyse_folder))):
#     # anaylser2featselect(pathtofolder)
# else:
#     print('\nRe-organization of the files already performed, so moving the files skipped')

####### END 


########  Create list with the paths of the files to analyse
pathto_sortedfolder = tissueanalyser_folder
jsonfiles = list()
cllist = list()
# cllist stands for for classification list (response (1) or noresponse (0))
for root, dirs, files in os.walk(pathto_sortedfolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a srting. You create a string with this:
        for file in files:
            namewoext, extension = os.path.splitext(file)
            filepath = root + '/' + file
            if extension == '.json' and 'analysed' and not '._' in namewoext:
                if not 'response' in namewoext:
                    raise ValueError('Some features are not associated to a response '
                                     'or noresponse WSI classification. User must sort JSON and rename it'
                                     ' with the corresponding response and no_response caracters')
                else:
                    jsonfiles.append(filepath)

####### If applicable create a dict file from the patient ID csv file 
# And initializa the futur ordered patient ID list
if patientid_avail:
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
        if patientid_avail:
            if 'response' in nameoffile:
                if calculate_vicinity:
                    if 'no_response' in nameoffile:
                        namesimplified = nameoffile.replace('_vicinity_no_response_analysed','')
                    else:
                        namesimplified = nameoffile.replace('_vicinity_response_analysed','')
                else:
                    if 'no_response' in nameoffile:
                        namesimplified = nameoffile.replace('_no_response_analysed','')
                    else:
                        namesimplified = nameoffile.replace('_response_analysed','')
            patientid_list.append(patientid_dict.get(namesimplified))

        # extract information of the JSON as a string
        print(filename)
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
        if 'response' in nameoffile:
            if 'no_response' in nameoffile:
                cllist.append(int(0))
            else:
                cllist.append(int(1))
        else:
            raise ValueError('Some features are not associated to a response '
                             'or noresponse WSI classification. User must sort JSON and rename it'
                             'with the corresponding reccurence and noreccurence caracters'
                             'User can check src.utils.filemanagement.anaylser2featselect function for more details.')

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



# Check if there are response and no response data in the training set (both needed)
if cllist:
    if 0 not in cllist:
        raise ValueError(
            'The data contains only no response data or json named as being noresponse. '
            'To run statistical test we need both response and noresponse examples')

    if 1 not in cllist:
        raise ValueError(
            'The data contains only response data or json named as being response. '
            'To run statistical test we need both response and noresponse examples')

clarray = np.asarray(cllist)
# clarray stands for for classification array (response or noresponse)

##### Display Feature Matrix and its corresponding classification vectors (reccurence or no response)
print("Feature Matrix Shape is", featarray.shape)
print("Feature Matrix is", featarray)
print("Classification Vector is", clarray)


### Maybe better to do the per patient column based on the previous matrix
# just 

if perpatient_feat:
    
    """Concatenate the quantification features all together 
       in pandas DataFrame. Create a corresponding list of patient ID.
       There will be one feature column per patient, and not per WSI"""

    if not patientid_avail:
        raise ValueError(
                'A patient ID csv file is needed to have a feature matrix with one column '
                'per patient. The boolean patientid_avail must be set to True. ')

    patientids_convert = convert_names_to_orderedint(patientid_list)

    # Create a dict with keys being values of the ID and value the list of indexes
    indexes_dict = get_indices_by_value(patientids_convert)

    #Create one vector per patient
    patient_meanfeatlist = list()
    patient_medianfeatlist = list()
    patient_cllist = list() 

    # We extract each set of indexes corresponding to one patient
    for indexes_list in indexes_dict.values():
        # Take only the columns corresponding to the patient
        patient_matr = featarray[:,indexes_list]
        columns_average = np.mean(patient_matr, axis=1)
        columns_median = np.median(patient_matr, axis=1)
        # Add to the list the mean and average columns
        patient_meanfeatlist.append(columns_average)
        patient_medianfeatlist.append(columns_median)
        # Need also to updated the classification vector
        corresponding_cls = clarray[min(indexes_list)]
        patient_cllist.append(corresponding_cls)


    #Transform list into numpy arrays. The arrays will be transposed later on?
    patient_mfeatarray = np.transpose(np.asarray(patient_meanfeatlist))
    patient_medianfeatarray = np.transpose(np.asarray(patient_medianfeatlist))
    patient_clarray = np.asarray(patient_cllist)




############################################################
## Save all numpy files
############################################################

# Save all the files in the tissue analyses folder
# Create the path to folder that will contain the numpy feature selection files

pathoutput = pathfeatselect
ext = '.npy'

# If the folder doesn't exist create it
if not os.path.exists(pathoutput):
    os.makedirs(pathoutput)

print('Saving feature array, classification array')


# Save feature names in a np file
pathfeatnames = pathoutput + 'featnames' + ext
featnames = np.asarray(featnameslist)
np.save(pathfeatnames, featnames)

# Save feature arrays, classification arrays and featnames list
# In the case of feature and classification arrays with one column per wsi
path_perwsifeatarray = pathoutput + 'perwsi_featarray' + ext
np.save(path_perwsifeatarray, featarray)
path_perwsiclarray = pathoutput + 'perwsi_clarray' + ext
np.save(path_perwsiclarray, clarray)

if perpatient_feat:
    # In the case of feature and classification arrays with one column per patient
    # Calculated with the mean of each vectors
    path_perpat_featarray = pathoutput + 'perpat_featarray' + ext
    np.save(path_perpat_featarray, patient_mfeatarray)
    path_perpat_clarray = pathoutput + 'perpat_clarray' + ext
    np.save(path_perpat_clarray, patient_clarray)
    # In the case of feature and classification arrays with one column per patient
    # Calculated with the mediam of each vectors
    path_perpat_median_featarray = pathoutput + 'perpat_median_featarray' + ext
    np.save(path_perpat_median_featarray, patient_medianfeatarray)



# If applicable, save the patient_ID list as a ids array:
if patientid_avail:
    patientid_array = np.asarray(patientid_list)
    pathpatientids =  pathoutput + 'patientids' + ext
    np.save(pathpatientids, patientid_array)


# Calculate Pearson correlation regardless of the class response, no-response
corrmat =  np.corrcoef(featarray)
path_corrmat = pathoutput + 'correlation_matrix' + ext
np.save(path_corrmat, corrmat)
path_corrmat_csv = pathoutput + 'correlation_matrix.csv' 
np.savetxt(path_corrmat_csv, corrmat, delimiter=",")


print('Saving done.')
print('Path to the output files: {}'.format(pathoutput))

