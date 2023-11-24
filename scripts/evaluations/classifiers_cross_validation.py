#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import os.path

import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn import linear_model, ensemble
from sklearn.model_selection import train_test_split
 

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import src.histo_miner.utils.misc as utils_misc
import joblib



# #############################################################
# ## Load configs parameter
# #############################################################


# # Import parameters values from config file by generating a dict.
# # The lists will be imported as tuples.
# with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# # Create a config dict from which we can access the keys with dot syntax
# configfolder = attributedict(config)
# pathtofolder = configfolder.paths.folders.feature_selection_main

# # Import parameters values from config file by generating a dict.
# # The lists will be imported as tuples.
# with open("./../../configs/classification_training.yml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# # Create a config dict from which we can access the keys with dot syntax
# config = attributedict(config)
# classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
# perform_split = config.parameters.bool.perform_split
# split_pourcentage = config.parameters.int.split_pourcentage

# ridge_alpha = config.classifierparam.ridge.alpha
# lregression_solver = config.classifierparam.logistic_regression.solver
# lregression_multi_class = config.classifierparam.logistic_regression.multi_class
# forest_n_estimators = config.classifierparam.random_forest.n_estimators
# forest_class_weight = config.classifierparam.random_forest.class_weight
# xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
# xgboost_lr = config.classifierparam.xgboost.learning_rate
# xgboost_objective = config.classifierparam.xgboost.objective
# lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
# lgbm_lr = config.classifierparam.light_gbm.learning_rate
# lgbm_objective = config.classifierparam.light_gbm.objective
# lgbm_numleaves = config.classifierparam.light_gbm.num_leaves

# saveclassifier_ridge = config.parameters.bool.saving_classifiers.ridge
# saveclassifier_lr = config.parameters.bool.saving_classifiers.logistic_regression
# saveclassifier_forest = config.parameters.bool.saving_classifiers.random_forest
# saveclassifier_xgboost = config.parameters.bool.saving_classifiers.xgboost
# saveclassifier_lgbm = config.parameters.bool.saving_classifiers.light_gbm 



# ############################################################
# ## Create Paths Strings to Classifiers
# ############################################################


# # Even if the training is not launched or some steps are skipped,
# # Always define the path to the following files, corresponding to the trained models
# # Because existence or none existence of these files indicate wich previious steps were done or skipped

# # Folder name to save models (might not be used)
# modelfolder = pathtofolder +'/classification_models/'

# pathridge_vanilla = modelfolder + 'ridge_vanilla.joblib'
# pathlr_vanilla = modelfolder + 'lr_vanilla.joblib'
# pathforest_vanilla = modelfolder + 'forest_vanilla.joblib'
# pathxgboost_vanilla = modelfolder + 'xgboost_vanilla.joblib'
# pathlgbm_vanilla = modelfolder + 'lgbm_vanilla.joblib'

# pathridge_mrmr = modelfolder + 'ridge_mrmr.joblib'
# pathlr_mrmr = modelfolder + 'lr_mrmr.joblib'
# pathforest_mrmr = modelfolder + 'forest_mrmr.joblib'
# pathxgboost_mrmr = modelfolder + 'xgboost_mrmr.joblib'
# pathlgbm_mrmr = modelfolder + 'lgbm_mrmr.joblib'

# pathridge_boruta = modelfolder + 'ridge_boruta.joblib'
# pathlr_boruta = modelfolder + 'lr_boruta.joblib'
# pathforest_boruta = modelfolder + 'forest_boruta.joblib'
# pathxgboost_boruta = modelfolder + 'xgboost_boruta.joblib'
# pathlgbm_boruta = modelfolder + 'lgbm_boruta.joblib'

# pathridge_mannwhitney = modelfolder + 'ridge_mannwhitney.joblib'
# pathlr_mannwhitney = modelfolder + 'lr_mannwhitney.joblib'
# pathforest_mannwhitney = modelfolder + 'forest_mannwhitney.joblib'
# pathxgboost_mannwhitney = modelfolder + 'xgboost_mannwhitney.joblib'
# pathlgbm_mannwhitney = modelfolder + 'lgbm_mannwhitney.joblib'



# ############################################################
# ## Load feature selection numpy files
# ############################################################


# pathfeatselect = pathtofolder + '/feature_selection/'
# ext = '.npy'

# print('Load feature selection numpy files...')

# # Load feature selection numpy files
# # Each time we check if the file exist because all selections are not forced to run
# pathselfeat_mrmr = pathfeatselect + 'selfeat_mrmr_idx' + ext
# if os.path.exists(pathselfeat_mrmr):
#     selfeat_mrmr = np.load(pathselfeat_mrmr, allow_pickle=True)
# pathselfeat_boruta = pathfeatselect + 'selfeat_boruta_idx' + ext
# if os.path.exists(pathselfeat_boruta):
#     selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
# pathselfeat_mannwhitneyu = pathfeatselect + 'selfeat_mannwhitneyu_idx' + ext
# if os.path.exists(pathselfeat_mannwhitneyu):
#     selfeat_mannwhitneyu = np.load(pathselfeat_mannwhitneyu, allow_pickle=True)
# print('Loading feature selected indexes done.')



# ############################################################
# ## Split data into training and test sets if not done and
# ## Load feat array and classification arrays
# ############################################################

# # create all the path for the classifications and feature arrays
# path_train_featarray = pathfeatselect + 'featarray_train' + ext
# path_train_clarray = pathfeatselect + 'clarray_train' + ext
# path_test_featarray = pathfeatselect + 'featarray_test' + ext
# path_test_clarray = pathfeatselect + 'clarray_test' + ext


# #split the arrays into test and train arrays
# if perform_split:
#     print('Splitting set into training classification array + feature matrix and'
#            ' test classification array + feature matrix')
#     list_train_arrays, list_test_arrays =  utils_misc.split_featclarrays(
#                                                 pathtofolder = pathfeatselect,
#                                                 splitpourcent = split_pourcentage,
#                                                 )
#     train_featarray = list_train_arrays[0]
#     train_clarray = list_train_arrays[1]
#     np.save(path_train_featarray, train_featarray)
#     np.save(path_train_clarray, train_clarray)

#     test_featarray = list_test_arrays[0]
#     test_clarray = list_test_arrays[1]
#     np.save(path_test_featarray, test_featarray)
#     np.save(path_test_clarray, test_clarray)



# if not perform_split:
#     if not os.path.exists(path_train_featarray) or not os.path.exists(path_train_clarray):
#         raise ValueError('Training classification array at {} and/or trainnig feature matrix at {} was not found.'
#                           'User can check the naming of files or launch the feature selection to generate them. ')


# #This is to check but should be fine
# train_featarray = np.load(path_train_featarray)
# train_clarray = np.load(path_train_clarray)
# train_clarray = np.transpose(train_clarray)



# ############################################################
# ## Traininig Classifiers
# ############################################################


# # Define the classifiers
# # More information here: #https://scikit-learn.org/stable/modules/linear_model.html
# ##### RIDGE CLASSIFIER
# Ridge = linear_model.RidgeClassifier(alpha=ridge_alpha)
# ##### LOGISTIC REGRESSION
# LR = linear_model.LogisticRegression(solver=lregression_solver,
#                                      multi_class=lregression_multi_class)
# ##### RANDOM FOREST
# Forest = ensemble.RandomForestClassifier(n_estimators=forest_n_estimators,
#                                          class_weight=forest_class_weight)
# ##### XGBOOST
# XGBoost = xgboost.XGBClassifier(n_estimators=xgboost_n_estimators, 
#                                 learning_rate=xgboost_lr, 
#                                 objective=xgboost_objective,
#                                 verbosity=0)
# ##### LIGHT GBM setting
# # The use of light GBM classifier is not following the convention of the other one
# # Here we will save parameters needed for training, but there are no .fit method
# LightGBM = lightgbm.LGBMClassifier(n_estimators=lgbm_n_estimators,
#                                   learning_rate=lgbm_lr,
#                                   objective=lgbm_objective,
#                                   num_leaves=lgbm_numleaves,
#                                   verbosity=-1)

# #RMQ: Verbosity is set to 0 for XGBOOST to avoid printing WARNINGS (not wanted here for sake of
# #simplicity)/ In Light GBM, to avoid showing WARNINGS, the verbosity as to be set to -1.
# # See parameters documentation to learn about the other verbosity available. 


# if not os.path.exists(modelfolder):
#     os.makedirs(modelfolder)

# print('Start Classifiers trainings...')


# #### Classification training with all features kept 

# if classification_from_allfeatures:
#     # Use all the feature (no selection) as input
#     genfeatarray = np.transpose(train_featarray)
#     ##### RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier and fit (train) it to the data
#     RidgeVanilla = Ridge
#     ridge_vanilla = RidgeVanilla.fit(genfeatarray, train_clarray)
#     # If saving:
#     if saveclassifier_ridge:
#         joblib.dump(ridge_vanilla, pathridge_vanilla)
#     ##### LOGISTIC REGRESSION
#     # Initialize the Logistic Regression and fit (train) it to the data
#     LRVanilla = LR
#     lr_vanilla = LRVanilla.fit(genfeatarray, train_clarray)
#     # If saving:
#     if saveclassifier_lr:
#         joblib.dump(lr_vanilla, pathlr_vanilla)
#     ##### RANDOM FOREST
#     # Initialize the Random Forest and fit (train) it to the data
#     ForestVanilla = Forest
#     forest_vanilla = ForestVanilla.fit(genfeatarray, train_clarray)
#     # If saving:
#     if saveclassifier_forest:
#         joblib.dump(forest_vanilla, pathforest_vanilla)
#     ##### XGBOOST
#     XGBoostVanilla = XGBoost
#     xgboost_vanilla = XGBoostVanilla.fit(genfeatarray, train_clarray)
#     # If saving:
#     if saveclassifier_xgboost:
#         joblib.dump(xgboost_vanilla, pathxgboost_vanilla)
#     ##### LIGHT GBM
#     # lgbm_traindata_vanilla = lightgbm.Dataset(genfeatarray, label=train_clarray) 
#     # #lgbm_valdata_vanilla = lgbm_traindata_vanilla.create_valid()
#     # lgbm_vanilla = lightgbm.train(lightgbm_paramters, 
#     #                               lgbm_traindata_vanilla, 
#     #                               lgbm_n_estimators)
#     LightGBMVanilla = LightGBM
#     lgbm_vanilla = LightGBMVanilla.fit(genfeatarray, train_clarray) 
#     # If saving:
#     if saveclassifier_lgbm:
#         # Don't know if joblib works
#         joblib.dump(lgbm_vanilla, pathlgbm_vanilla)



# #### Parse the featarray to the class SelectedFeaturesMatrix 

# SelectedFeaturesMatrix = SelectedFeaturesMatrix(train_featarray)


# #### Classification training with the features kept by mrmr

# if os.path.exists(pathselfeat_mrmr):
#     # Generate the matrix with selected feature for mrmr
#     featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr(selfeat_mrmr)
#     ##### RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier and fit (train) it to the data
#     RidgeMrmr = Ridge
#     ridge_mrmr = RidgeMrmr.fit(featarray_mrmr, train_clarray)
#     # If saving:
#     if saveclassifier_ridge:
#         joblib.dump(ridge_mrmr, pathridge_mrmr)
#     ##### LOGISTIC REGRESSION
#     # Initialize the Logistic Regression and fit (train) it to the data
#     LRMrmr = LR
#     lr_mrmr = LRMrmr.fit(featarray_mrmr, train_clarray)
#     # If saving:
#     if saveclassifier_lr:
#         joblib.dump(lr_mrmr, pathlr_mrmr)
#     ##### RANDOM FOREST
#     # Initialize the Random Forest and fit (train) it to the data
#     ForestMrmr = Forest
#     forest_mrmr = ForestMrmr.fit(featarray_mrmr, train_clarray)
#     # If saving:
#     if saveclassifier_forest:
#         joblib.dump(forest_mrmr, pathforest_mrmr)
#     ##### XGBOOST
#     XGBoostMrmr = XGBoost
#     xgboost_mrmr = XGBoostMrmr.fit(featarray_mrmr, train_clarray)
#     # If saving:
#     if saveclassifier_xgboost:
#         joblib.dump(xgboost_mrmr, pathxgboost_mrmr)
#     ##### LIGHT GBM
#     LightGBMMrmr = LightGBM
#     lgbm_mrmr = LightGBMMrmr.fit(featarray_mrmr, train_clarray) 
#     # If saving:
#     if saveclassifier_lgbm:
#         # Don't know if joblib works
#         joblib.dump(lgbm_mrmr, pathlgbm_mrmr)


# #### Classification training with the features kept by boruta

# if os.path.exists(pathselfeat_boruta):
#     # Generate the matrix with selected feature for boruta
#     featarray_boruta = SelectedFeaturesMatrix.boruta_matr(selfeat_boruta)
#     ##### RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier and fit (train) it to the data
#     RidgeBoruta = Ridge
#     ridge_boruta = RidgeBoruta.fit(featarray_boruta, train_clarray)
#     # If saving:
#     if saveclassifier_ridge:
#         joblib.dump(ridge_boruta, pathridge_boruta)
#     ##### LOGISTIC REGRESSION
#     # Initialize the Logistic Regression and fit (train) it to the data
#     LRBoruta = LR
#     lr_boruta = LRBoruta.fit(featarray_boruta, train_clarray)
#     # If saving:
#     if saveclassifier_lr:
#         joblib.dump(lr_boruta, pathlr_boruta)
#     ##### RANDOM FOREST
#     # Initialize the Random Forest and fit (train) it to the data
#     ForestBoruta = Forest
#     forest_boruta = ForestBoruta.fit(featarray_boruta, train_clarray)
#     # If saving:
#     if saveclassifier_forest:
#         joblib.dump(forest_boruta, pathforest_boruta)
#     ##### XGBOOST
#     XGBoostBoruta = XGBoost
#     xgboost_boruta = XGBoostBoruta.fit(featarray_boruta, train_clarray)
#     # If saving:
#     if saveclassifier_xgboost:
#         joblib.dump(xgboost_boruta, pathxgboost_boruta)
#     ##### LIGHT GBM
#     LightGBMBoruta = LightGBM
#     lgbm_boruta = LightGBMBoruta.fit(featarray_boruta, train_clarray) 
#     # If saving:
#     if saveclassifier_lgbm:
#         # Don't know if joblib works
#         joblib.dump(lgbm_boruta, pathlgbm_boruta)


# #### Classification training with the features kept by mannwhitneyu

# if os.path.exists(pathselfeat_mannwhitneyu):
#     # Generate the matrix with selected feature for mannwhitney
#     featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(selfeat_mannwhitneyu)
#     ##### RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier and fit (train) it to the data
#     RidgeMannWhitney = Ridge
#     ridge_mannwhitney = RidgeMannWhitney.fit(featarray_mannwhitney, train_clarray)
#     # If saving:
#     if saveclassifier_ridge:
#         joblib.dump(ridge_mannwhitney, pathridge_mannwhitney)
#     ##### LOGISTIC REGRESSION
#     # Initialize the Logistic Regression and fit (train) it to the data
#     LRMannWhitney = LR
#     lr_mannwhitney = LRMannWhitney.fit(featarray_mannwhitney, train_clarray)
#     # If saving:
#     if saveclassifier_lr:
#         joblib.dump(lr_mannwhitney, pathlr_mannwhitney)
#     ##### RANDOM FOREST
#     # Initialize the Random Forest and fit (train) it to the data
#     ForestMannWhitney = Forest
#     forest_mannwhitney = ForestMannWhitney.fit(featarray_mannwhitney, train_clarray)
#     # If saving:
#     if saveclassifier_forest:
#         joblib.dump(forest_mannwhitney, pathforest_mannwhitney)
#     ##### XGBOOST
#     XGBoostMannWhitney = XGBoost
#     xgboost_mannwhitney = XGBoostMannWhitney.fit(featarray_mannwhitney, train_clarray)
#     # If saving:
#     if saveclassifier_xgboost:
#         joblib.dump(xgboost_mannwhitney, pathxgboost_mannwhitney)
#     ##### LIGHT GBM
#     LightGBMMannWhitney = LightGBM
#     lgbm_mannwhitney = LightGBMMannWhitney.fit(featarray_mannwhitney, train_clarray) 
#     # If saving:
#     if saveclassifier_lgbm:
#         # Don't know if joblib works
#         joblib.dump(lgbm_mannwhitney, pathlgbm_mannwhitney)


# print('All classifiers trained.')
# print('Classifiers saved here: ', modelfolder)
# print('Classifiers saved are the ones that have saving_classifiers set as True in the config.')
