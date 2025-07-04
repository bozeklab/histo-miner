#Lucas Sancéré -

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grandparent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(script_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

import os.path
from tqdm import tqdm
import numpy as np
import time
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn import linear_model, ensemble, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, \
cross_validate, cross_val_score, GroupKFold, StratifiedGroupKFold
import joblib
import json

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import src.histo_miner.utils.misc as utils_misc




##############################################################
## Load configs parameter
#############################################################

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open(script_dir + "/../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(config)
featarray_folder = confighm.paths.folders.featarray_folder
nbr_keptfeat = confighm.parameters.int.nbr_keptfeat
patientid_avail = confighm.parameters.bool.patientid_avail



with open(script_dir + "/../../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)

predefined_feature_selection = config.parameters.bool.predefined_feature_selection

folder_output = config.paths.folders.save_trained_model
training_model_name = config.names.trained_model
featsel_path = config.paths.files.feature_selection_file

run_xgboost = config.parameters.bool.run_classifiers.xgboost
run_lgbm = config.parameters.bool.run_classifiers.light_gbm 

xgboost_random_state = config.classifierparam.xgboost.random_state
xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
xgboost_lr = config.classifierparam.xgboost.learning_rate
xgboost_objective = config.classifierparam.xgboost.objective

lgbm_random_state = config.classifierparam.light_gbm.random_state
lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
lgbm_lr = config.classifierparam.light_gbm.learning_rate
lgbm_objective = config.classifierparam.light_gbm.objective
lgbm_numleaves = config.classifierparam.light_gbm.num_leaves




############################################################
## Create Paths Strings to Classifiers
############################################################


modelfolder = folder_output
modelext = '.joblib'

# Create folder is it doesn't exist yet
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)

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




###################################################################
## Load feature selection indexes - for pred-defined selection
###################################################################


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



###################################################################
## Load feature selection indexes - if custom feat selection
###################################################################


else:
    selfeat_idx = np.load(featsel_path, allow_pickle=True)

    # !!! Kept given number of best features (previously ordered from best to worst)
    selfeat_idx_final = selfeat_idx[0:nbr_keptfeat]



################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################

#This is to check but should be fine
print('Load feature array')

ext = '.npy'

path_featarray = featarray_folder + 'perwsi_featarray' + ext
path_clarray = featarray_folder + 'perwsi_clarray' + ext
path_patientids_array = featarray_folder + 'patientids' + ext

train_featarray = np.load(path_featarray)
train_clarray = np.load(path_clarray)


if patientid_avail:
    patientids_load = np.load(path_patientids_array, allow_pickle=True)
    patientids_list = list(patientids_load)
    patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
    patientids = np.asarray(patientids_convert)



################################################################
## Keep best features
################################################################

selected_features_matrix = SelectedFeaturesMatrix(train_featarray)
train_featarray = selected_features_matrix.mrmr_matr(selfeat_idx_final)




##############################################################
## Setting Classifiers
##############################################################


# Define the classifiers
# More information here: #https://scikit-learn.org/stable/modules/linear_model.html

##### XGBOOST
xgboost_clf = xgboost.XGBClassifier(random_state= xgboost_random_state,
                                n_estimators=xgboost_n_estimators, 
                                learning_rate=xgboost_lr, 
                                objective=xgboost_objective,
                                verbosity=0)
##### LIGHT GBM setting
# The use of light GBM classifier is not following the convention of the other one
# Here we will save parameters needed for training, but there are no .fit method
lightgbm_clf = lightgbm.LGBMClassifier(random_state= lgbm_random_state,
                                   n_estimators=lgbm_n_estimators,
                                   learning_rate=lgbm_lr,
                                   objective=lgbm_objective,
                                   num_leaves=lgbm_numleaves,
                                   verbosity=-1)

#RMQ: Verbosity is set to 0 for XGBOOST to avoid printing WARNINGS (not wanted here for sake of
#simplicity)/ In Light GBM, to avoid showing WARNINGS, the verbosity as to be set to -1.
# See parameters documentation to learn about the other verbosity available. 



##############################################################
## Training  Classifiers
##############################################################


print('Start Classifiers trainings...')

if run_xgboost and not run_lgbm:

    # Fit on the entire dataset
    xgboost_clf.fit(train_featarray, train_clarray)

    # Save the trained XGBoost model
    joblib.dump(xgboost_clf, path_to_model)
    print(f"XGBoost model saved to: {path_to_model}\n")



if run_lgbm and not run_xgboost:

    # Fit on the entire dataset
    lightgbm_clf.fit(train_featarray, train_clarray)

    # Save the trained XGBoost model
    joblib.dump(lightgbm_clf, path_to_model)
    print(f"XGBoost model saved to: {path_to_model}\n")



print('\nClassifier(s) trained.')
print('Classifier(s) saved here: ', modelfolder)

