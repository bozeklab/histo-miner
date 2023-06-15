#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import os.path

import numpy as np
import yaml
from attrdict import AttrDict as attributedict
from sklearn import linear_model, ensemble

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import joblib

# import scipy.stats
# import sys
# import pandas as pd
# import mrmr
# import boruta
# from sklearn.ensemble import RandomForestClassifier
# import json
# import os
# import time
# import subprocess



#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.main
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures

ridge_alpha = config.classifierparam.ridge.alpha
lr_solver = config.classifierparam.logistic_regression.solver
lr_multi_class = config.classifierparam.logistic_regression.multi_class
forest_n_estimators = config.classifierparam.random_forest.n_estimators
forest_class_weight = config.classifierparam.random_forest.class_weight

saveclassifier_ridge = config.parameters.bool.saving_classifiers.ridge
saveclassifier_lr = config.parameters.bool.saving_classifiers.logistic_regression
saveclassifier_forest = config.parameters.bool.saving_classifiers.random_forest



############################################################
## Load feature selection numpy files
############################################################


pathnumpy = pathtofolder.replace('tissue_analyses/', 'feature_selection/')
ext = '.npy'

print('Load feeature selection numpy files...')

pathfeatarray = pathnumpy + 'featarray' + ext
pathclarray = pathnumpy + 'clarray' + ext
#raise an error here if one of the file doesn't exist

featarray = np.load(pathfeatarray)
clarray = np.load(pathclarray)

# Each time we check iif the file exist because all selections are not forced to run
pathselfeat_mrmr = pathnumpy + 'selfeat_mrmr' + ext
if os.path.exists(pathselfeat_mrmr):
    selfeat_mrmr = np.load(pathselfeat_mrmr, allow_pickle=True)
pathselfeat_boruta = pathnumpy + 'selfeat_boruta' + ext
if os.path.exists(pathselfeat_boruta):
    selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
pathorderedp_mannwhitneyu = pathnumpy + 'orderedp_mannwhitneyu' + ext
if os.path.exists(pathorderedp_mannwhitneyu):
    orderedp_mannwhitneyu = np.load(pathorderedp_mannwhitneyu, allow_pickle=True)
print('Loading done.')



############################################################
## Create Paths Strings to Classifiers
############################################################


# Even if the training is not launched or some steps are skipped,
# Always define the path to the following files, corresponding to the trained models
# Because existence or none existence of these files indicate wich previious steps were done or skipped

# Folder name to save models (might not be used)
modelfolder = pathtofolder.replace('tissue_analyses/', 'classification_models/')

pathridge_vanilla = modelfolder + 'ridge_vanilla.joblib'
pathlr_vanilla = modelfolder + 'lr_vanilla.joblib'
pathforest_vanilla = modelfolder + 'forest_vanilla.joblib'
pathridge_mrmr = modelfolder + 'ridge_mrmr.joblib'
pathlr_mrmr = modelfolder + 'lr_mrmr.joblib'
pathforest_mrmr = modelfolder + 'forest_mrmr.joblib'
pathridge_boruta = modelfolder + 'ridge_boruta.joblib'
pathlr_boruta = modelfolder + 'lr_boruta.joblib'
pathforest_boruta = modelfolder + 'forest_boruta.joblib'
pathridge_mannwhitney = modelfolder + 'ridge_mannwhitney.joblib'
pathlr_mannwhitney = modelfolder + 'lr_mannwhitney.joblib'
pathforest_mannwhitney = modelfolder + 'forest_mannwhitney.joblib'



############################################################
## Traininig Classifiers
############################################################



# Define the classifiers
# More information here: #https://scikit-learn.org/stable/modules/linear_model.html
##### RIDGE CLASSIFIER
Ridge = linear_model.RidgeClassifier(alpha=ridge_alpha)
##### LOGISTIC REGRESSION
LR = linear_model.LogisticRegression(solver=lr_solver,
                                     multi_class=lr_multi_class)
##### RANDOM FOREST
Forest = ensemble.RandomForestClassifier(n_estimators=forest_n_estimators,
                                         class_weight=forest_class_weight)

if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)

print('Start Classifiers trainings...')
if classification_from_allfeatures:
    # Use all the feature (no selection) as input
    genfeatarray = np.transpose(featarray)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    RidgeVanilla = Ridge
    ridge_vanilla = RidgeVanilla.fit(genfeatarray, clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_vanilla, pathridge_vanilla)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    LRVanilla = LR
    lr_vanilla = LRVanilla.fit(genfeatarray, clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_vanilla, pathlr_vanilla)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    ForestVanilla = Forest
    forest_vanilla = ForestVanilla.fit(genfeatarray, clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_vanilla, pathforest_vanilla)


SelectedFeaturesMatrix = SelectedFeaturesMatrix(featarray)

if os.path.exists(pathselfeat_mrmr):
    # Generate the matrix with selected feature for mrmr
    featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr(selfeat_mrmr)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    RidgeMrmr = Ridge
    ridge_mrmr = RidgeMrmr.fit(featarray_mrmr, clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_mrmr, pathridge_mrmr)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    LRMrmr = LR
    lr_mrmr = LRMrmr.fit(featarray_mrmr, clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_mrmr, pathlr_mrmr)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    ForestMrmr = Forest
    forest_mrmr = ForestMrmr.fit(featarray_mrmr, clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_mrmr, pathforest_mrmr)


if os.path.exists(pathselfeat_boruta):
    # Here the matrix with selected feature is already done (directly the output)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    RidgeBoruta = Ridge
    ridge_boruta = RidgeBoruta.fit(selfeat_boruta, clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_boruta, pathridge_boruta)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    LRBoruta = LR
    lr_boruta = LRBoruta.fit(selfeat_boruta, clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_boruta, pathlr_boruta)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    ForestBoruta = Forest
    forest_boruta = ForestBoruta.fit(selfeat_boruta, clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_boruta, pathforest_boruta)


if os.path.exists(pathorderedp_mannwhitneyu):
    # Generate the matrix with selected feature for mannwhitney
    featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(orderedp_mannwhitneyu)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    RidgeMannWhitney = Ridge
    ridge_mannwhitney = RidgeMannWhitney.fit(featarray_mannwhitney, clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_mannwhitney, pathridge_mannwhitney)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    LRMannWhitney = LR
    lr_mannwhitney = LRMannWhitney.fit(featarray_mannwhitney, clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_mannwhitney, pathlr_mannwhitney)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    ForestMannWhitney = Forest
    forest_mannwhitney = ForestMannWhitney.fit(featarray_mannwhitney, clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_mannwhitney, pathforest_mannwhitney)

print('All classifiers trrained.')
print('Classifiers saved here: ', modelfolder)
