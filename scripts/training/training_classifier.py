#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

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

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import src.histo_miner.utils.misc as utils_misc
import joblib




##############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(config)
featarray_folder = confighm.paths.folders.feature_selection_main
classification_eval_folder = confighm.paths.folders.classification_evaluation
eval_folder_name = confighm.names.eval_folder

patientid_csv = confighm.paths.files.patientid_csv
patientid_avail = confighm.parameters.bool.patientid_avail
nbr_keptfeat = confighm.parameters.int.nbr_keptfeat



with open("./../../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures # see if we remove it
nbr_of_splits = config.parameters.int.nbr_of_splits
run_name = config.names.run_name
run_xgboost = config.parameters.bool.run_classifiers.xgboost
run_lgbm = config.parameters.bool.run_classifiers.light_gbm 



ridge_random_state = config.classifierparam.ridge.random_state
ridge_alpha = config.classifierparam.ridge.alpha
ridge_param_grid_random_state = list(config.classifierparam.ridge.grid_dict.random_state)
ridge_param_grid_alpha = list(config.classifierparam.ridge.grid_dict.alpha)

lregression_random_state = config.classifierparam.logistic_regression.random_state
lregression_penalty = config.classifierparam.logistic_regression.penalty
lregression_solver = config.classifierparam.logistic_regression.solver
lregression_multi_class = config.classifierparam.logistic_regression.multi_class
lregression_class_weight = config.classifierparam.logistic_regression.class_weight
lregression_param_grid_random_state = list(config.classifierparam.logistic_regression.grid_dict.random_state)
lregression_param_grid_penalty = list(config.classifierparam.logistic_regression.grid_dict.penalty)
lregression_param_grid_solver = list(config.classifierparam.logistic_regression.grid_dict.solver)
lregression_param_grid_multi_class = list(config.classifierparam.logistic_regression.grid_dict.multi_class)
lregression_param_grid_class_weight = list(config.classifierparam.logistic_regression.grid_dict.class_weight)

forest_random_state = config.classifierparam.random_forest.random_state
forest_n_estimators = config.classifierparam.random_forest.n_estimators
forest_class_weight = config.classifierparam.random_forest.class_weight
forest_param_grid_random_state = list(config.classifierparam.random_forest.grid_dict.random_state)
forest_param_grid_n_estimators = list(config.classifierparam.random_forest.grid_dict.n_estimators)
forest_param_grid_class_weight = list(config.classifierparam.random_forest.grid_dict.class_weight)

xgboost_random_state = config.classifierparam.xgboost.random_state
xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
xgboost_lr = config.classifierparam.xgboost.learning_rate
xgboost_objective = config.classifierparam.xgboost.objective
xgboost_param_grid_random_state = list(config.classifierparam.xgboost.grid_dict.random_state)
xgboost_param_grid_n_estimators = list(config.classifierparam.xgboost.grid_dict.n_estimators)
xgboost_param_grid_learning_rate = list(config.classifierparam.xgboost.grid_dict.learning_rate)
xgboost_param_grid_objective = list(config.classifierparam.xgboost.grid_dict.objective)

lgbm_random_state = config.classifierparam.light_gbm.random_state
lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
lgbm_lr = config.classifierparam.light_gbm.learning_rate
lgbm_objective = config.classifierparam.light_gbm.objective
lgbm_numleaves = config.classifierparam.light_gbm.num_leaves
lgbm_param_grid_random_state = list(config.classifierparam.light_gbm.grid_dict.random_state)
lgbm_param_grid_n_estimators = list(config.classifierparam.light_gbm.grid_dict.n_estimators)
lgbm_param_grid_learning_rate = list(config.classifierparam.light_gbm.grid_dict.learning_rate)
lgbm_param_grid_objective = list(config.classifierparam.light_gbm.grid_dict.objective)
lgbm_param_grid_num_leaves = list(config.classifierparam.light_gbm.grid_dict.num_leaves)



############################################################
## Create Paths Strings to Classifiers
############################################################



# Folder name to save models (might not be used)
# remove the last folder from pah 
rootmodelfolder = os.path.dirname(classification_eval_folder.rstrip(os.path.sep))
modelfolder = rootmodelfolder + '/classification_models/'

pathridge_vanilla = modelfolder + 'ridge_vanilla.joblib'
pathlr_vanilla = modelfolder + 'lr_vanilla.joblib'
pathforest_vanilla = modelfolder + 'forest_vanilla.joblib'
pathxgboost_vanilla = modelfolder + 'xgboost_vanilla.joblib'
pathlgbm_vanilla = modelfolder + 'lgbm_vanilla.joblib'

pathridge_mrmr = modelfolder + 'ridge_mrmr.joblib'
pathlr_mrmr = modelfolder + 'lr_mrmr.joblib'
pathforest_mrmr = modelfolder + 'forest_mrmr.joblib'
pathxgboost_mrmr = modelfolder + 'xgboost_mrmr.joblib'
pathlgbm_mrmr = modelfolder + 'lgbm_mrmr.joblib'

pathridge_boruta = modelfolder + 'ridge_boruta.joblib'
pathlr_boruta = modelfolder + 'lr_boruta.joblib'
pathforest_boruta = modelfolder + 'forest_boruta.joblib'
pathxgboost_boruta = modelfolder + 'xgboost_boruta.joblib'
pathlgbm_boruta = modelfolder + 'lgbm_boruta.joblib'

pathridge_mannwhitney = modelfolder + 'ridge_mannwhitney.joblib'
pathlr_mannwhitney = modelfolder + 'lr_mannwhitney.joblib'
pathforest_mannwhitney = modelfolder + 'forest_mannwhitney.joblib'
pathxgboost_mannwhitney = modelfolder + 'xgboost_mannwhitney.joblib'
pathlgbm_mannwhitney = modelfolder + 'lgbm_mannwhitney.joblib'



############################################################
## Load feature selection numpy files
############################################################

print('Load feature selection numpy files...')

featsel_folder = classification_eval_folder + eval_folder_name + '/'
ext = '.npy'



# Load feature selection numpy files*
# need to readapt this 
# -->
if run_xgboost and not run_lgbm:
    classifier_name = 'xgboost'
if run_lgbm and not run_xgboost:
    classifier_name = 'lgbm'

name_mrmr_output = '_ba_mrmr_' + str(nbr_of_splits) + 'splits_' + run_name
name_boruta_output = '_ba_boruta_' + str(nbr_of_splits) + 'splits_' + run_name 
name_mannwhitneyu_output = '_ba_mannwhitneyu_' + str(nbr_of_splits) + 'splits_' + run_name 

path_selfeat_mrmr = featsel_folder + 'topselfeatid_' + classifier_name  + name_mrmr_output + ext
path_selfeat_boruta =  featsel_folder + 'topselfeatid_' + classifier_name  + name_boruta_output + ext
path_selfeat_mannwhitneyu = featsel_folder + 'topselfeatid_' + classifier_name  + name_mannwhitneyu_output + ext


# load the indexes fo the top features
if os.path.exists(path_selfeat_mrmr):
    selfeat_mrmr = np.load(path_selfeat_mrmr, allow_pickle=True)

if os.path.exists(path_selfeat_boruta):
    selfeat_boruta = np.load(path_selfeat_boruta, allow_pickle=True)

if os.path.exists(path_selfeat_mannwhitneyu):
    selfeat_mannwhitneyu = np.load(path_selfeat_mannwhitneyu, allow_pickle=True)

print('Loading feature selected indexes done.')



# !!! Kept given number of features

if os.path.exists(path_selfeat_mrmr):
    selfeat_mrmr_idx = selfeat_mrmr[0:nbr_keptfeat]
if os.path.exists(path_selfeat_mannwhitneyu):
    selfeat_mannwhitneyu_idx = selfeat_mrmr[0:nbr_keptfeat]
print('Refinement of feature selected indexes done.')





################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################

#This is to check but should be fine
path_featarray = featarray_folder + 'perwsi_featarray' + ext
path_clarray = featarray_folder + 'perwsi_clarray' + ext
path_patientids_array = featarray_folder + 'patientids' + ext

train_featarray = np.load(path_featarray)
train_clarray = np.load(path_clarray)
# train_clarray = np.transpose(train_clarray)
train_featarray = np.transpose(train_featarray)

if patientid_avail:
    patientids_load = np.load(path_patientids_array, allow_pickle=True)
    patientids_list = list(patientids_load)
    patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
    patientids = np.asarray(patientids_convert)





################################################################
## Keep best features
################################################################


train_featarray = train_featarray[:,selfeat_mrmr_idx]




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

# ##### RIDGE CLASSIFIER
# ridge_clf = linear_model.RidgeClassifier(random_state= ridge_random_state,
#                                      alpha=ridge_alpha)
# ##### LOGISTIC REGRESSION
# lr_clf = linear_model.LogisticRegression(random_state=lregression_random_state,
#                                      penalty=lregression_penalty,
#                                      solver=lregression_solver,
#                                      multi_class=lregression_multi_class,
#                                      class_weight=lregression_class_weight)
# ##### RANDOM FOREST
# forest_clf = ensemble.RandomForestClassifier(random_state= forest_random_state,
#                                          n_estimators=forest_n_estimators,
#                                          class_weight=forest_class_weight)

# Create folder is it doesn't exist yet
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)




##############################################################
## Training  Classifiers
##############################################################


print('Start Classifiers trainings...')

if run_xgboost and not run_lgbm:

    # Fit on the entire dataset
    xgboost_clf.fit(train_featarray, train_clarray)

    # Save the trained XGBoost model
    joblib.dump(xgboost_clf, pathxgboost_mrmr)
    print(f"XGBoost model saved to: {pathxgboost_mrmr}\n")



if run_lgbm and not run_xgboost:

    # Fit on the entire dataset
    lightgbm_clf.fit(train_featarray, train_clarray)

    # Save the trained XGBoost model
    joblib.dump(lightgbm_clf, pathlgbm_mrmr)
    print(f"XGBoost model saved to: {pathlgbm_mrmr}\n")



print('\nClassifier(s) trained.')
print('Classifier(s) saved here: ', modelfolder)

