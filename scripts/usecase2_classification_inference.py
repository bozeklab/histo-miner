#Lucas Sancéré -

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(script_dir)   # subdir/
sys.path.append(parent_dir)   # project/

import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict
from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import joblib
import json


#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open(script_dir + "/../configs/histo_miner_pipeline.yml", "r") as f:
    confighm = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(confighm)
featarray_folder = confighm.paths.folders.featarray_folder
nbr_keptfeat = confighm.parameters.int.nbr_keptfeat


with open(script_dir + "/../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)

model_folder = config.paths.folders.save_trained_model
training_model_name = config.names.trained_model
predefined_feature_selection = config.parameters.bool.predefined_feature_selection
featsel_path = config.paths.files.feature_selection_file

input_folder = config.paths.folders.inference_input

run_xgboost = config.parameters.bool.run_classifiers.xgboost
run_lgbm = config.parameters.bool.run_classifiers.light_gbm 

displayclass_score = config.parameters.bool.display_classification_scores



############################################################
## Create Paths Strings to Classifiers
############################################################


modelfolder = model_folder
modelext = '.joblib'

if run_xgboost and not run_lgbm:
    classifier_name = 'xgboost'
if run_lgbm and not run_xgboost:
    classifier_name = 'lgbm'

path_to_model = [
    modelfolder + 
    classifier_name + '_'  
    + training_model_name + '_'
    + str(nbr_keptfeat) + 'featkept'
    + modelext
    ]
# path_to_model = str(path_to_model[0])


############################################################
## Sanity checkcs
############################################################



if not os.path.exists(path_to_model) or not os.path.exists(featsel_path):
    raise ValueError(
        'Config path are incorrect and do not point towards the model or the selected features.' 
        ' Model path entered is {}. The feature selected folder entered is {}'
        .format(path_to_model, featsel_path)
        )




############################################################
## Load inference data
############################################################


print('Load feature array')
ext = '.npy'
pathfeatarray = featarray_folder + 'perwsi_featarray' + ext
pathclarray = featarray_folder + 'perwsi_clarray' + ext
#TO DO:
#raise an error here if one of the file doesn't exist

inf_featarray = np.load(pathfeatarray)
inf_clarray = np.load(pathclarray)


## Load the indexes of selected features 

if predefined_feature_selection:
    
    with open(featsel_path, "r") as f:
        data = json.load(f)

    # Get list of first-level keys
    keys = list(data.keys())

    # Load the file with names of features ordered    
    featnames_ordered_npy = np.load(featarray_folder + '/featnames.npy')

    featnames_ordered = list(featnames_ordered_npy)

    selfeat_idx = [featnames_ordered.index(name) for name in keys]

    # !!! Kept given number of best features (previously ordered from best to worst)
    selfeat_idx_final = selfeat_idx[0:nbr_keptfeat]

else:
    selfeat_idx = np.load(featsel_path, allow_pickle=True)

    # !!! Kept given number of best features (previously ordered from best to worst)
    selfeat_idx_final = selfeat_idx[0:nbr_keptfeat]





################################################################
## Keep best features
################################################################

selected_features_matrix = SelectedFeaturesMatrix(inf_featarray)
inf_featarray = selected_features_matrix.mrmr_matr(selfeat_idx_final)    



############################################################
## Classifiers Inference
############################################################


#### Classification training with the features kept by mrmr

model = joblib.load(path_to_model)
# Predict the labels for new data
model_pred = model.predict(inf_featarray)
print('model_pred : {}'.format(model_pred))
if displayclass_score:
    print("Accuracy of classifier:",
    model.score(inf_featarray, inf_clarray))



