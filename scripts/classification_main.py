#Lucas Sancéré -

# import sys
# sys.path.append('../')  # Only for Remote use on Clusters
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


# -----> PREDICTION ON TRAINING DATA§ Not good!! Split the set into 2 when more data!!!!!
#For now we just check that the code is working, it is a debugging step



#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.main
nbr_keptfeat = config.parameters.int.nbr_keptfeat
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures

ridge_alpha = config.classifierparam.ridge.alpha
lr_solver = config.classifierparam.logistic_regression.solver
lr_multi_class = config.classifierparam.logistic_regression.multi_class
forest_n_estimators = config.classifierparam.random_forest.n_estimators
forest_class_weight = config.classifierparam.random_forest.class_weight

training_classifiers = config.parameters.bool.training_classifier
saveclassifier_ridge = config.parameters.bool.saving_classifier.ridge
saveclassifier_lr = config.parameters.bool.saving_classifier.logistic_regression
saveclassifier_forest = config.parameters.bool.saving_classifier.random_forest



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
    selfeat_mrmr = np.load(pathselfeat_mrmr)
pathselfeat_boruta = pathnumpy + 'selfeat_boruta' + ext
if os.path.exists(pathselfeat_boruta):
    selfeat_boruta = np.load(pathselfeat_boruta)
pathorderedp_mannwhitneyu = pathnumpy + 'orderedp_mannwhitneyu' + ext
if os.path.exists(pathorderedp_mannwhitneyu):
    orderedp_mannwhitneyu = np.load(pathorderedp_mannwhitneyu)
print('Loading done.')



############################################################
## Create Paths Strings to Classifiers
############################################################


# Even if the training is not launched or some steps are skipped,
# Always define the path to the following files, corresponding to the trained models
# Because existence or none existence of these files indicate wich previious steps were done or skipped

# Folder name to save models (might not be used)
modelfolder = 'ClassificationModels/'

pathridge_vanilla = pathtofolder + modelfolder + 'ridge_vanilla.joblib'
pathlr_vanilla = pathtofolder + modelfolder + 'lr_vanilla.joblib'
pathforest_vanilla = pathtofolder + modelfolder + 'forest_vanilla.joblib'
pathridge_mrmr = pathtofolder + modelfolder + 'ridge_mrmr.joblib'
pathlr_mrmr = pathtofolder + modelfolder + 'lr_mrmr.joblib'
pathforest_mrmr = pathtofolder + modelfolder + 'forest_mrmr.joblib'
pathridge_boruta = pathtofolder + modelfolder + 'ridge_boruta.joblib'
pathlr_boruta = pathtofolder + modelfolder + 'lr_boruta.joblib'
pathforest_boruta = pathtofolder + modelfolder + 'forest_boruta.joblib'
pathridge_mannwhitney = pathtofolder + modelfolder + 'ridge_mannwhitney.joblib'
pathlr_mannwhitney = pathtofolder + modelfolder + 'lr_mannwhitney.joblib'
pathforest_mannwhitney = pathtofolder + modelfolder + 'forest_mannwhitney.joblib'



############################################################
## Traininig Classifiers
############################################################


if training_classifiers:

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




############################################################
## Classifiers Inference
############################################################


# Maybe change the if condition in case we are not training the classifiers


if classification_from_allfeatures:
    # Load test data (no feature selection)
    test_featarray = np.transpose(featarray)

    # Predict the labels for new data
    ##### RIDGE CLASSIFIER
    if os.path.exists(pathridge_vanilla):
        ridge_vanilla = joblib.load(pathridge_vanilla)
        ridge_vanilla_pred = ridge_vanilla.predict(test_featarray)
        print('ridge_pred : {}'.format(ridge_vanilla_pred))
        print("Accuracy of RIDGE classifier:",
              RidgeVanilla.score(genfeatarray, clarray))
    ##### LOGISTIC REGRESSION
    if os.path.exists(pathlr_vanilla):
        lr_vanilla = joblib.load(pathlr_vanilla)
        lr_vanilla_pred = lr_vanilla.predict(test_featarray)
        print('lr_pred : {}'.format(lr_vanilla_pred))
        print("Accuracy of LOGISTIC classifier:",
              LRVanilla.score(genfeatarray, clarray))
    ##### RANDOM FOREST
    if os.path.exists(pathforest_vanilla):
        forest_vanilla = joblib.load(pathforest_vanilla)
        forest_vanilla_pred = forest_vanilla.predict(test_featarray)
        print('forest_pred : {}'.format(forest_vanilla_pred))
        print("Accuracy of RANDOM FOREST classifier:",
              ForestVanilla.score(genfeatarray, clarray))


if os.path.exists(pathselfeat_mrmr):
    # Load test data (that went through mrmr method)
    test_featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr(selfeat_mrmr)

    # Predict the labels for new data
    ##### RIDGE CLASSIFIER
    if os.path.exists(pathridge_mrmr):
        ridge_mrmr = joblib.load(pathridge_mrmr)
        ridge_mrmr_pred = ridge_mrmr.predict(test_featarray_mrmr)
        print('ridge_mrmr_pred : {}'.format(ridge_mrmr_pred))
        print("Accuracy of RIDGE MRMR classifier:",
              RidgeMrmr.score(featarray_mrmr, clarray))
    ##### LOGISTIC REGRESSION
    if os.path.exists(pathlr_mrmr):
        lr_mrmr = joblib.load(pathlr_mrmr)
        lr_mrmr_pred = lr_mrmr.predict(test_featarray_mrmr)
        print('lr_mrmr_pred : {}'.format(lr_mrmr_pred))
        print("Accuracy of LOGISTIC MRMR classifier:",
              LRMrmr.score(featarray_mrmr, clarray))
    ##### RANDOM FOREST
    if os.path.exists(pathforest_mrmr):
        forest_mrmr = joblib.load(pathforest_mrmr)
        forest_mrmr_pred = forest_mrmr.predict(test_featarray_mrmr)
        print('forest_mrmr_pred : {}'.format(forest_mrmr_pred))
        print("Accuracy of RANDOM FOREST MRMR classifier:",
              ForestMrmr.score(featarray_mrmr, clarray))


if os.path.exists(pathselfeat_boruta):
    # Load test data (that went through boruta method)
    test_featarray_boruta = selfeat_boruta

    # Predict the labels for new data
    ##### RIDGE CLASSIFIER
    if os.path.exists(pathridge_boruta):
        ridge_boruta = joblib.load(pathridge_boruta)
        ridge_boruta_pred = ridge_boruta.predict(test_featarray_boruta)
        print('ridge_boruta_pred : {}'.format(ridge_boruta_pred))
        print("Accuracy of RIDGE BORUTA classifier:",
              RidgeBoruta.score(selfeat_boruta, clarray))
    ##### LOGISTIC REGRESSION
    if os.path.exists(pathlr_boruta):
        lr_boruta = joblib.load(pathlr_boruta)
        lr_boruta_pred = lr_boruta.predict(test_featarray_boruta)
        print('lr_boruta_pred : {}'.format(lr_boruta_pred))
        print("Accuracy of LOGISTIC BORUTA classifier:",
              LRBoruta.score(selfeat_boruta, clarray))
    ##### RANDOM FOREST
    if os.path.exists(pathforest_boruta):
        forest_boruta = joblib.load(pathforest_boruta)
        forest_boruta_pred = forest_boruta.predict(test_featarray_boruta)
        print('forest_boruta_pred : {}'.format(forest_boruta_pred))
        print("Accuracy of RANDOM FOREST BORUTA classifier:",
              ForestBoruta.score(selfeat_boruta, clarray))


if os.path.exists(pathorderedp_mannwhitneyu):
    # Load test data (that went through Mann Whitney U rank test)
    test_featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(orderedp_mannwhitneyu)

    # Predict the labels for new data
    ##### RIDGE CLASSIFIER
    if os.path.exists(pathridge_mannwhitney):
        ridge_mannwhitney = joblib.load(pathridge_mannwhitney)
        ridge_mannwhitney_pred = ridge_mannwhitney.predict(test_featarray_mannwhitney)
        print('ridge_mannwhitney_pred : {}'.format(ridge_mannwhitney_pred))
        print("Accuracy of RIDGE MANN WHITNEY classifier:",
              RidgeMannWhitney.score(ridge_mannwhitney_pred, clarray))
    ##### LOGISTIC REGRESSION
    if os.path.exists(pathlr_mannwhitney):
        lr_mannwhitney = joblib.load(pathlr_mannwhitney)
        lr_mannwhitney_pred = lr_mannwhitney.predict(test_featarray_mannwhitney)
        print('lr_mannwhitney_pred : {}'.format(lr_mannwhitney_pred))
        print("Accuracy of LOGISTIC MANN WHITNEY classifier:",
              LRMannWhitney.score(lr_mannwhitney_pred, clarray))
    ##### RANDOM FOREST
    if os.path.exists(pathforest_mannwhitney):
        forest_mannwhitney = joblib.load(pathforest_mannwhitney)
        forest_mannwhitney_pred = forest_mannwhitney.predict(test_featarray_mannwhitney)
        print('forest_mannwhitney_pred : {}'.format(forest_mannwhitney_pred))
        print("Accuracy of RANDOM FOREST MANN WHITNEY classifier:",
              ForestMannWhitney.score(featarray_mannwhitney, clarray))
