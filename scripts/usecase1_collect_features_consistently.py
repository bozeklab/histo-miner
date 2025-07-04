#Lucas Sancéré -

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(script_dir)   # subdir/
sys.path.append(parent_dir)   # project/

import json
import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict

from src.histo_miner.utils.misc import convert_flatten, convert_flatten_redundant, noheadercsv_to_dict, \
                                       convert_names_to_orderedint, get_indices_by_value, rename_with_ancestors
from src.histo_miner.utils.filemanagment import anaylser2featselect



#####################################################################
## Load configs parameters
#####################################################################

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open(script_dir + "/../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathanalyserout = config.paths.folders.tissue_analyser_output
featarray_folder = config.paths.folders.featarray_folder
cohortid_csv = config.paths.files.cohortid_csv
patientid_csv = config.paths.files.patientid_csv
cohortid_avail = config.parameters.bool.cohortid_avail
patientid_avail = config.parameters.bool.patientid_avail
calculate_vicinity = config.parameters.bool.calculate_vicinity
redundant_feat_names = list(config.parameters.lists.redundant_feat_names)


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


##### /!\   


# If the 2 folders norec_analyse_folder and rec_analyse_folder don't exsit, 
# or they exist but are empty, we start the re-organization of the files with anaylser2featselect.

# For now we keep this commented as it is a big risk to run it

# norec_analyse_folder = tissueanalyser_folder + '/' + 'no_response' 
# rec_analyse_folder = tissueanalyser_folder + '/' + 'response'  
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
            if (
                extension == '.json'  
                and 'analysed' in namewoext 
                and '._' not in namewoext     # in case of MACOS
                and '.DS_S' not in namewoext    # in case of MACOS
            ):
                if not 'response' in namewoext:
                    raise ValueError('Some features are not associated to a response  '
                                     'or no response  WSI classification. User must sort JSON and rename it'
                                     ' with the corresponding response  and no_response  caracters')
                else:
                    jsonfiles.append(filepath)


####### If applicable create a dict file from the patient ID csv file 
# And initializa the futur ordered patient ID list
if patientid_avail:
    patientid_dict = noheadercsv_to_dict(patientid_csv)
    patientid_list = list()

####### If applicable create a dict file from the cohort ID csv file 
# And initializa the futur ordered cohort ID list
if cohortid_avail:
    cohortid_dict = noheadercsv_to_dict(cohortid_csv)
    cohortid_list = list()    


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

        # Generate the cohort list if available for corss-validation;
        if cohortid_avail:
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
            cohortid_list.append(cohortid_dict.get(namesimplified))


    if  feature_init:
        #Create a list of names of the features, (only done once as all the json have the same features)
        #We create a new dictionnary that is using not the same keys name, but simplified ones.
        
        #Be carefukl in the redundancy we have in the areas, circularity, aspect ratio and dis features
        renamed_analysisdata = rename_with_ancestors(analysisdata, redundant_feat_names)

        #Check the difference between convert_flatten and convert_flatten_redundant docstrings
        simplifieddata =  convert_flatten(renamed_analysisdata)
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
if cohortid_avail:
    print("Cohort ID list is", cohortid_list)



############################################################
## Save all numpy files
############################################################

# Save all the files in the tissue analyses folder
# Create the path to folder that will contain the numpy feature selection files

pathoutput = featarray_folder
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


# If applicable, save the patient_ID list as a ids array:
if patientid_avail:
    patientid_array = np.asarray(patientid_list)
    pathpatientids =  pathoutput + 'patientids' + ext
    np.save(pathpatientids, patientid_array)

# If applicable, save the cohort_ID list as a ids array:
if cohortid_avail:
    cohortid_array = np.asarray(cohortid_list)
    pathcohortids =  pathoutput + 'cohortids' + ext
    np.save(pathcohortids, cohortid_array)


# Calculate Pearson correlation regardless of the class response, no-response
corrmat =  np.corrcoef(featarray)
path_corrmat = pathoutput + 'correlation_matrix' + ext
np.save(path_corrmat, corrmat)
path_corrmat_csv = pathoutput + 'correlation_matrix.csv' 
np.savetxt(path_corrmat_csv, corrmat, delimiter=",")


print('Saving done.')
print('Path to the output files: {}'.format(pathoutput))


