#Lucas Sancéré -

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grandparent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(script_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict
import joblib

from src.histo_miner.feature_selection import SelectedFeaturesMatrix




#############################################################
## Load configs parameter
#############################################################

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open(script_dir + "/../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.tissue_analyser_main
nbr_keptfeat = config.parameters.int.nbr_keptfeat


############################################################
## Create Paths Strings to Classifiers
############################################################


# Maybe in the inference part could be simplified

# Even if the training is not launched or some steps are skipped,
# Always define the path to the following files, corresponding to the trained models
# Because existence or none existence of these files indicate wich previious steps were done or skipped

# Folder name to save models (might not be used)
modelfolder = pathtofolder +'/classification_models/'

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
## Load test/inference data
############################################################

pathfeatselect = pathtofolder + '/feature_selection/'
ext = '.npy'

print('Load feeature selection numpy files...')

pathfeatarray = pathfeatselect + 'featarray_test' + ext
pathclarray = pathfeatselect + 'clarray_test' + ext
#TO DO:
#raise an error here if one of the file doesn't exist

eval_featarray = np.load(pathfeatarray)
eval_clarray = np.load(pathclarray)
eval_clarray = np.transpose(eval_clarray)


# Each time we check if the file exist because all selections are not forced to run
pathselfeat_mrmr = pathfeatselect + 'selfeat_mrmr_idx' + ext
if os.path.exists(pathselfeat_mrmr):
    selfeat_mrmr = np.load(pathselfeat_mrmr, allow_pickle=True)
pathselfeat_boruta = pathfeatselect + 'selfeat_boruta_idx' + ext
if os.path.exists(pathselfeat_boruta):
    selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
pathorderedp_mannwhitneyu = pathfeatselect + 'selfeat_mannwhitneyu_idx' + ext
if os.path.exists(pathorderedp_mannwhitneyu):
    orderedp_mannwhitneyu = np.load(pathorderedp_mannwhitneyu, allow_pickle=True)
print('Loading done.')


############################################################
## Classifiers Inference
############################################################

############ TO ADD
# Add calculation of balanced accuracy to every inference
###################


#### Parse the featarray to the class SelectedFeaturesMatrix 

selected_features_matrix = SelectedFeaturesMatrix(eval_featarray)

#### Classification training with the features kept by mrmr

if os.path.exists(pathselfeat_mrmr):
    # Load test data (that went through mrmr method)
    test_featarray_mrmr = selected_features_matrix.mrmr_matr(selfeat_mrmr)
    # test_featarray_mrmr = np.transpose(test_featarray_mrmr)

    # Predict the labels for new data
    ##### RIDGE CLASSIFIER
    if os.path.exists(pathridge_mrmr):
        ridge_mrmr = joblib.load(pathridge_mrmr)
        # ridge_mrmr_pred = ridge_mrmr.predict(test_featarray_mrmr)
        # print('ridge_mrmr_pred : {}'.format(ridge_mrmr_pred))
        print("Accuracy of RIDGE MRMR classifier:",
                  ridge_mrmr.score(test_featarray_mrmr, eval_clarray))
    ##### LOGISTIC REGRESSION
    if os.path.exists(pathlr_mrmr):
        lr_mrmr = joblib.load(pathlr_mrmr)
        # lr_mrmr_pred = lr_mrmr.predict(test_featarray_mrmr)
        # print('lr_mrmr_pred : {}'.format(lr_mrmr_pred))
        print("Accuracy of LOGISTIC MRMR classifier:",
                  lr_mrmr.score(test_featarray_mrmr, eval_clarray))
    ##### RANDOM FOREST
    if os.path.exists(pathforest_mrmr):
        forest_mrmr = joblib.load(pathforest_mrmr)
        # forest_mrmr_pred = forest_mrmr.predict(test_featarray_mrmr)
        # print('forest_mrmr_pred : {}'.format(forest_mrmr_pred))
        print("Accuracy of RANDOM FOREST MRMR classifier:",
                  forest_mrmr.score(test_featarray_mrmr, eval_clarray))
    ##### XGBOOST
    if os.path.exists(pathxgboost_mrmr):
        xgboost_mrmr = joblib.load(pathxgboost_mrmr)
        # xgboost_mrmr_pred = xgboost_mrmr.predict(test_featarray_mrmr)
        # print('xgboost_mrmr_pred : {}'.format(xgboost_mrmr_pred))
        print("Accuracy of XGBOOST MRMR classifier:",
                  xgboost_mrmr.score(test_featarray_mrmr, eval_clarray))
    ##### LIGHT GBM
    if os.path.exists(pathlgbm_mrmr):
        lgbm_mrmr = joblib.load(pathlgbm_mrmr)
        # lgbm_mrmr_pred = lgbm_mrmr.predict(test_featarray_mrmr)
        # print('lgbm_mrmr_pred : {}'.format(lgbm_mrmr_pred))
        print("Accuracy of LIGHT GBM MRMR classifier:",
                  lgbm_mrmr.score(test_featarray_mrmr, eval_clarray))


#### Classification training with the features kept by boruta

if os.path.exists(pathselfeat_boruta):
    test_featarray_boruta = selected_features_matrix.mrmr_matr(selfeat_boruta)
    # test_featarray_boruta = np.transpose(test_featarray_boruta)

    # Predict the labels for new data
    ##### RIDGE CLASSIFIER
    if os.path.exists(pathridge_boruta):
        ridge_boruta = joblib.load(pathridge_boruta)
        # ridge_boruta_pred = ridge_boruta.predict(test_featarray_boruta)
        # print('ridge_boruta_pred : {}'.format(ridge_boruta_pred))
        print("Accuracy of RIDGE BORUTA classifier:",
                  ridge_boruta.score(test_featarray_boruta, eval_clarray))
    ##### LOGISTIC REGRESSION
    if os.path.exists(pathlr_boruta):
        lr_boruta = joblib.load(pathlr_boruta)
        # lr_boruta_pred = lr_boruta.predict(test_featarray_boruta)
        # print('lr_boruta_pred : {}'.format(lr_boruta_pred))
        print("Accuracy of LOGISTIC BORUTA classifier:",
                  lr_boruta.score(test_featarray_boruta, eval_clarray))
    ##### RANDOM FOREST
    if os.path.exists(pathforest_boruta):
        forest_boruta = joblib.load(pathforest_boruta)
        # forest_boruta_pred = forest_boruta.predict(test_featarray_boruta)
        # print('forest_boruta_pred : {}'.format(forest_boruta_pred))
        print("Accuracy of RANDOM FOREST BORUTA classifier:",
                  forest_boruta.score(test_featarray_boruta, eval_clarray))
    ##### XGBOOST
    if os.path.exists(pathxgboost_boruta):
        xgboost_boruta = joblib.load(pathxgboost_boruta)
        # xgboost_boruta_pred = xgboost_boruta.predict(test_featarray_boruta)
        # print('xgboost_boruta_pred : {}'.format(xgboost_boruta_pred))
        print("Accuracy of XGBOOST BORUTA classifier:",
                  xgboost_boruta.score(test_featarray_boruta, eval_clarray))
    ##### LIGHT GBM
    if os.path.exists(pathlgbm_boruta):
        lgbm_boruta = joblib.load(pathlgbm_boruta)
        # lgbm_boruta_pred = lgbm_boruta.predict(test_featarray_boruta)
        # print('lgbm_boruta_pred : {}'.format(lgbm_boruta_pred))
        print("Accuracy of LIGHT GBM BORUTA classifier:",
                  lgbm_boruta.score(test_featarray_boruta, eval_clarray))


#### Classification training with the features kept by mannwhitneyu

if os.path.exists(pathorderedp_mannwhitneyu):
    test_featarray_mannwhitney = selected_features_matrix.mannwhitney_matr(orderedp_mannwhitneyu)
    # test_featarray_mannwhitney = np.transpose(test_featarray_mannwhitney)

    # Predict the labels for new data
    ##### RIDGE CLASSIFIER
    if os.path.exists(pathridge_mannwhitney):
        ridge_mannwhitney = joblib.load(pathridge_mannwhitney)
        # ridge_mannwhitney_pred = ridge_mannwhitney.predict(test_featarray_mannwhitney)
        # print('ridge_mannwhitney_pred : {}'.format(ridge_mannwhitney_pred))
        print("Accuracy of RIDGE MANN WHITNEY classifier:",
                  ridge_mannwhitney.score(test_featarray_mannwhitney, eval_clarray))
    ##### LOGISTIC REGRESSION
    if os.path.exists(pathlr_mannwhitney):
        lr_mannwhitney = joblib.load(pathlr_mannwhitney)
        # lr_mannwhitney_pred = lr_mannwhitney.predict(test_featarray_mannwhitney)
        # print('lr_mannwhitney_pred : {}'.format(lr_mannwhitney_pred))
        print("Accuracy of LOGISTIC MANN WHITNEY classifier:",
                  lr_mannwhitney.score(test_featarray_mannwhitney, eval_clarray))
    ##### RANDOM FOREST
    if os.path.exists(pathforest_mannwhitney):
        forest_mannwhitney = joblib.load(pathforest_mannwhitney)
        # forest_mannwhitney_pred = forest_mannwhitney.predict(test_featarray_mannwhitney)
        # print('forest_mannwhitney_pred : {}'.format(forest_mannwhitney_pred))
        print("Accuracy of RANDOM FOREST MANN WHITNEY classifier:",
                  forest_mannwhitney.score(test_featarray_mannwhitney, eval_clarray))
    ##### XGBOOST
    if os.path.exists(pathxgboost_mannwhitney):
        xgboost_mannwhitney = joblib.load(pathxgboost_mannwhitney)
        # xgboost_mannwhitney_pred = xgboost_mannwhitney.predict(test_featarray_mannwhitney)
        # print('xgboost_mannwhitney_pred : {}'.format(xgboost_mannwhitney_pred))
        print("Accuracy of XGBOOST MANN WHITNEY classifier:",
               xgboost_mannwhitney.score(test_featarray_mannwhitney, eval_clarray))
    ##### LIGHT GBM
    if os.path.exists(pathlgbm_mannwhitney):
        lgbm_mannwhitney = joblib.load(pathlgbm_mannwhitney)
        # lgbm_mannwhitney_pred = lgbm_mannwhitney.predict(test_featarray_mannwhitney)
        # print('lgbm_mannwhitney_pred : {}'.format(lgbm_mannwhitney_pred))
        print("Accuracy of LIGHT GBM MANN WHITNEY classifier:",
                  lgbm_mannwhitney.score(test_featarray_mannwhitney, eval_clarray))
