#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import os
import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import joblib


#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    confighm = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(confighm)
classification_eval_folder = confighm.paths.folders.classification_evaluation
featarray_folder = confighm.paths.folders.featarray_folder
eval_folder_name = confighm.names.eval_folder
nbr_keptfeat = confighm.parameters.int.nbr_keptfeat

with open("./../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
model_name = config.names.trained_model
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures # see if we remove it
nbr_of_splits = config.parameters.int.nbr_of_splits
run_name = config.names.run_name
run_xgboost = config.parameters.bool.run_classifiers.xgboost
run_lgbm = config.parameters.bool.run_classifiers.light_gbm 

displayclass_score = config.parameters.bool.display_classification_scores
displayclass_pred = config.parameters.bool.display_classification_predictions




############################################################
## Create Paths Strings to Classifiers
############################################################


# remove the last folder from pah 
rootmodelfolder = os.path.dirname(classification_eval_folder.rstrip(os.path.sep))
modelfolder = rootmodelfolder + '/classification_models/'
modelext = '.joblib'


model_path = modelfolder + model_name + modelext



############################################################
## Load inference data
############################################################

print('Load feature selection numpy files...')

featsel_folder = classification_eval_folder + eval_folder_name + '/'
ext = '.npy'


# Load feature selection numpy files
if run_xgboost and not run_lgbm:
    classifier_name = 'xgboost'
if run_lgbm and not run_xgboost:
    classifier_name = 'lgbm'

# Only one will be kept here but we take consideration of all feature selection methods 
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



print('Load feature array')
pathfeatarray = featarray_folder + 'perwsi_featarray' + ext
pathclarray = featarray_folder + 'perwsi_clarray' + ext
#TO DO:
#raise an error here if one of the file doesn't exist

inf_featarray = np.load(pathfeatarray)
inf_clarray = np.load(pathclarray)



################################################################
## Keep best features
################################################################

selected_features_matrix = SelectedFeaturesMatrix(inf_featarray)

if os.path.exists(path_selfeat_mrmr):
    inf_featarray = selected_features_matrix.mrmr_matr(selfeat_mrmr_idx)    
if os.path.exists(path_selfeat_boruta):
    inf_featarray = selected_features_matrix.mrmr_matr(selfeat_boruta)
if os.path.exists(path_selfeat_mannwhitneyu):
    inf_featarray = selected_features_matrix.mrmr_matr(selfeat_mannwhitneyu_idx)




############################################################
## Classifiers Inference
############################################################


#### Classification training with the features kept by mrmr

if os.path.exists(path_selfeat_mrmr) and os.path.exists(model_path):
    model = joblib.load(model_path)
    # Predict the labels for new data
    if displayclass_pred:
        model_pred = model.predict(inf_featarray)
        print('model_pred : {}'.format(model_pred))
    if displayclass_score:
        print("Accuracy of classifier:",
              model.score(inf_featarray, inf_clarray))


#### Classification training with the features kept by boruta

if os.path.exists(path_selfeat_boruta) and os.path.exists(model_path):
    model = joblib.load(model_path)
    # Predict the labels for new data
    if displayclass_pred:
        model_pred = model.predict(inf_featarray)
        print('model_pred : {}'.format(model_pred))
    if displayclass_score:
        print("Accuracy of classifier:",
              model.score(inf_featarray, inf_clarray))


#### Classification training with the features kept by mannwhitneyu

if os.path.exists(path_selfeat_mannwhitneyu) and os.path.exists(model_path):
    model = joblib.load(model_path)
    # Predict the labels for new data
    if displayclass_pred:
        model_pred = model.predict(inf_featarray)
        print('model_pred : {}'.format(model_pred))
    if displayclass_score:
        print("Accuracy of classifier:",
              model.score(inf_featarray, inf_clarray))





# DO in a second time
#### Classification  with all features kept 

# if classification_from_allfeatures:
#     # Load test data (no feature selection)
#     inf_globfeatarray = np.transpose(inf_featarray)

#     # Predict the labels for new data
#     ##### nameofmodel
#     if os.path.exists(pathnameofmodel_vanilla):
#         nameofmodel_vanilla = joblib.load(pathnameofmodel_vanilla)
#         if displayclass_pred:
#             nameofmodel_vanilla_pred = nameofmodel_vanilla.predict(inf_globfeatarray)
#             print('nameofmodel_pred : {}'.format(nameofmodel_vanilla_pred))
#         if displayclass_score:
#             print("Accuracy of nameofmodel classifier:",
#               nameofmodel_vanilla.score(inf_globfeatarray, inf_clarray))

