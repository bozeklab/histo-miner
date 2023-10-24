#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import os.path

import numpy as np
import time
import yaml
import xgboost 
import lightgbm
from attrdict import AttrDict as attributedict
from sklearn import linear_model, ensemble
from sklearn.model_selection import train_test_split, GridSearchCV

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import src.histo_miner.utils.misc as utils_misc
import joblib



#############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
configfolder = attributedict(config)
pathtofolder = configfolder.paths.folders.feature_selection_main

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
perform_split = config.parameters.bool.perform_split
split_pourcentage = config.parameters.int.split_pourcentage

ridge_alpha = config.classifierparam.ridge.alpha

lregression_solver = config.classifierparam.logistic_regression.solver
lregression_multi_class = config.classifierparam.logistic_regression.multi_class

forest_n_estimators = config.classifierparam.random_forest.n_estimators
forest_class_weight = config.classifierparam.random_forest.class_weight
forest_param_grid_n_estimators = list(config.classifierparam.random_forest.grid_dict.n_estimators)
forest_param_grid_class_weight = list(config.classifierparam.random_forest.grid_dict.class_weight)

xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
xgboost_lr = config.classifierparam.xgboost.learning_rate
xgboost_objective = config.classifierparam.xgboost.objective
xgboost_param_grid_n_estimators = list(config.classifierparam.xgboost.grid_dict.n_estimators)
xgboost_param_grid_learning_rate = list(config.classifierparam.xgboost.grid_dict.learning_rate)
xgboost_param_grid_objective = list(config.classifierparam.xgboost.grid_dict.objective)

lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
lgbm_lr = config.classifierparam.light_gbm.learning_rate
lgbm_objective = config.classifierparam.light_gbm.objective
lgbm_numleaves = config.classifierparam.light_gbm.num_leaves
lgbm_param_grid_n_estimators = list(config.classifierparam.light_gbm.grid_dict.n_estimators)
lgbm_param_grid_learning_rate = list(config.classifierparam.light_gbm.grid_dict.learning_rate)
lgbm_param_grid_objective = list(config.classifierparam.light_gbm.grid_dict.objective)
lgbm_param_grid_num_leaves = list(config.classifierparam.light_gbm.grid_dict.num_leaves)

saveclassifier_ridge = config.parameters.bool.saving_classifiers.ridge
saveclassifier_lr = config.parameters.bool.saving_classifiers.logistic_regression
saveclassifier_forest = config.parameters.bool.saving_classifiers.random_forest
saveclassifier_xgboost = config.parameters.bool.saving_classifiers.xgboost
saveclassifier_lgbm = config.parameters.bool.saving_classifiers.light_gbm 



############################################################
## Create Paths Strings to Classifiers
############################################################


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
## Load feature selection numpy files
############################################################


pathfeatselect = pathtofolder + '/feature_selection/'
ext = '.npy'

print('Load feature selection numpy files...')

# Load feature selection numpy files
# Each time we check if the file exist because all selections are not forced to run
pathselfeat_mrmr = pathfeatselect + 'selfeat_mrmr_idx' + ext
if os.path.exists(pathselfeat_mrmr):
    selfeat_mrmr = np.load(pathselfeat_mrmr, allow_pickle=True)
pathselfeat_boruta = pathfeatselect + 'selfeat_boruta_idx' + ext
if os.path.exists(pathselfeat_boruta):
    selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
pathselfeat_mannwhitneyu = pathfeatselect + 'selfeat_mannwhitneyu_idx' + ext
if os.path.exists(pathselfeat_mannwhitneyu):
    selfeat_mannwhitneyu = np.load(pathselfeat_mannwhitneyu, allow_pickle=True)
print('Loading feature selected indexes done.')



############################################################
## Split data into training and test sets if not done and
## Load feat array and classification arrays
############################################################

# create all the path for the classifications and feature arrays
path_train_featarray = pathfeatselect + 'featarray_train' + ext
path_train_clarray = pathfeatselect + 'clarray_train' + ext
path_test_featarray = pathfeatselect + 'featarray_test' + ext
path_test_clarray = pathfeatselect + 'clarray_test' + ext


#split the arrays into test and train arrays
if perform_split:
    print('Splitting set into training classification array + feature matrix and'
           ' test classification array + feature matrix')
    list_train_arrays, list_test_arrays =  utils_misc.split_featclarrays(
                                                pathtofolder = pathfeatselect,
                                                splitpourcent = split_pourcentage,
                                                )
    train_featarray = list_train_arrays[0]
    train_clarray = list_train_arrays[1]
    np.save(path_train_featarray, train_featarray)
    np.save(path_train_clarray, train_clarray)

    test_featarray = list_test_arrays[0]
    test_clarray = list_test_arrays[1]
    np.save(path_test_featarray, test_featarray)
    np.save(path_test_clarray, test_clarray)



if not perform_split:
    if not os.path.exists(path_train_featarray) or not os.path.exists(path_train_clarray):
        raise ValueError('Training classification array at {} and/or trainnig feature matrix at {} was not found.'
                          'User can check the naming of files or launch the feature selection to generate them. ')


#This is to check but should be fine
train_featarray = np.load(path_train_featarray)
train_clarray = np.load(path_train_clarray)
train_clarray = np.transpose(train_clarray)



############################################################
## Traininig Classifiers
############################################################


# Define the classifiers
# More information here: #https://scikit-learn.org/stable/modules/linear_model.html
##### RIDGE CLASSIFIER
ridge = linear_model.RidgeClassifier(alpha=ridge_alpha)
##### LOGISTIC REGRESSION
lr = linear_model.LogisticRegression(solver=lregression_solver,
                                     multi_class=lregression_multi_class)
##### RANDOM FOREST
forest = ensemble.RandomForestClassifier(n_estimators=forest_n_estimators,
                                         class_weight=forest_class_weight)
##### XGBOOST
xgboost = xgboost.XGBClassifier(n_estimators=xgboost_n_estimators, 
                                learning_rate=xgboost_lr, 
                                objective=xgboost_objective,
                                verbosity=0)
##### LIGHT GBM setting
# The use of light GBM classifier is not following the convention of the other one
# Here we will save parameters needed for training, but there are no .fit method
lightgbm = lightgbm.LGBMClassifier(n_estimators=lgbm_n_estimators,
                                  learning_rate=lgbm_lr,
                                  objective=lgbm_objective,
                                  num_leaves=lgbm_numleaves,
                                  verbosity=-1)

#RMQ: Verbosity is set to 0 for XGBOOST to avoid printing WARNINGS (not wanted here for sake of
#simplicity)/ In Light GBM, to avoid showing WARNINGS, the verbosity as to be set to -1.
# See parameters documentation to learn about the other verbosity available. 

# Create folder is it doesn't exist yet
if not os.path.exists(modelfolder):
    os.makedirs(modelfolder)


###### Load all paramters into a dictionnary for Grid Search
forest_param_grid = {
                      'n_estimators': forest_param_grid_n_estimators,
                      'class_weight': forest_param_grid_class_weight
}
xgboost_param_grid = {
                      'n_estimators': xgboost_param_grid_n_estimators,
                      'learning_rate': xgboost_param_grid_learning_rate,
                      'objective': xgboost_param_grid_objective
}
lgbm_param_grid = {
                    'n_estimators': lgbm_param_grid_n_estimators,
                    'learning_rate': lgbm_param_grid_learning_rate,
                    'objective': lgbm_param_grid_objective,
                    'num_leaves': lgbm_param_grid_num_leaves
}


print('Start Classifiers trainings...')

#### Classification training with all features kept 

if classification_from_allfeatures:
    # Use all the feature (no selection) as input
    genfeatarray = np.transpose(train_featarray)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgevanilla = ridge
    ridge_vanilla = ridgevanilla.fit(genfeatarray, train_clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_vanilla, pathridge_vanilla)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    lrvanilla  = lr
    lr_vanilla = lrvanilla.fit(genfeatarray, train_clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_vanilla, pathlr_vanilla)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestvanilla = forest
    # use Gridsearch to find the best set of HPs
    grid_forestvanilla =  GridSearchCV(forestvanilla, forest_param_grid)
    forest_vanilla = grid_forestvanilla.fit(genfeatarray, train_clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_vanilla, pathforest_vanilla)
    ##### XGBOOST
    xgboostvanilla = xgboost
    # use Gridsearch to find the best set of HPs
    grid_xgboostvanilla =  GridSearchCV(xgboostvanilla, xgboost_param_grid)
    xgboost_vanilla = grid_forestvanilla.fit(genfeatarray, train_clarray)
    # If saving:
    if saveclassifier_xgboost:
        joblib.dump(xgboost_vanilla, pathxgboost_vanilla)
    ##### LIGHT GBM
    # lgbm_traindata_vanilla = lightgbm.Dataset(genfeatarray, label=train_clarray) 
    # #lgbm_valdata_vanilla = lgbm_traindata_vanilla.create_valid()
    # lgbm_vanilla = lightgbm.train(lightgbm_paramters, 
    #                               lgbm_traindata_vanilla, 
    #                               lgbm_n_estimators)
    lightgbmvanilla = lightgbm
    # use Gridsearch to find the best set of HPs
    grid_lightgbmvanilla =  GridSearchCV(lightgbmvanilla, lgbm_param_grid)
    lgbm_vanilla = grid_lightgbmvanilla.fit(genfeatarray, train_clarray) 
    # If saving:
    if saveclassifier_lgbm:
        # Don't know if joblib works
        joblib.dump(lgbm_vanilla, pathlgbm_vanilla)



#### Parse the featarray to the class SelectedFeaturesMatrix 

SelectedFeaturesMatrix = SelectedFeaturesMatrix(train_featarray)


#### Classification training with the features kept by mrmr

if os.path.exists(pathselfeat_mrmr):
    # Generate the matrix with selected feature for mrmr
    featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr(selfeat_mrmr)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgemrmr = ridge
    ridge_mrmr = ridgemrmr.fit(featarray_mrmr, train_clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_mrmr, pathridge_mrmr)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    lrmrmr = lr
    lr_mrmr = lrmrmr.fit(featarray_mrmr, train_clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_mrmr, pathlr_mrmr)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestmrmr = forest
    # use Gridsearch to find the best set of HPs
    grid_forestmrmr =  GridSearchCV(forestmrmr, forest_param_grid)
    forest_mrmr = grid_forestvanilla.fit(featarray_mrmr, train_clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_mrmr, pathforest_mrmr)
    ##### XGBOOST
    xgboostmrmr = xgboost
    # use Gridsearch to find the best set of HPs
    grid_xgboostmrmr =  GridSearchCV(xgboostmrmr, xgboost_param_grid)
    xgboost_mrmr = grid_forestvanilla.fit(featarray_mrmr, train_clarray)
    # If saving:
    if saveclassifier_xgboost:
        joblib.dump(xgboost_mrmr, pathxgboost_mrmr)
    ##### LIGHT GBM
    lightgbmmrmr = lightgbm
    # use Gridsearch to find the best set of HPs
    grid_lightgbmmrmr =  GridSearchCV(lightgbmmrmr, lgbm_param_grid)
    lgbm_mrmr = grid_lightgbmmrmr.fit(featarray_mrmr, train_clarray) 
    # If saving:
    if saveclassifier_lgbm:
        # Don't know if joblib works
        joblib.dump(lgbm_mrmr, pathlgbm_mrmr)


#### Classification training with the features kept by boruta

if os.path.exists(pathselfeat_boruta):
    # Generate the matrix with selected feature for boruta
    featarray_boruta = SelectedFeaturesMatrix.boruta_matr(selfeat_boruta)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgeboruta = ridge
    ridge_boruta = ridgeboruta.fit(featarray_boruta, train_clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_boruta, pathridge_boruta)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    lrboruta = lr
    lr_boruta = lrboruta.fit(featarray_boruta, train_clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_boruta, pathlr_boruta)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestboruta = forest
    # use Gridsearch to find the best set of HPs
    grid_forestboruta =  GridSearchCV(forestboruta, forest_param_grid)
    forest_boruta = grid_forestboruta.fit(featarray_boruta, train_clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_boruta, pathforest_boruta)
    ##### XGBOOST
    xgboostboruta = xgboost
    # use Gridsearch to find the best set of HPs
    grid_xgboostboruta =  GridSearchCV(xgboostboruta, xgboost_param_grid)
    xgboost_boruta = grid_xgboostboruta.fit(featarray_boruta, train_clarray)
    # If saving:
    if saveclassifier_xgboost:
        joblib.dump(xgboost_boruta, pathxgboost_boruta)
    ##### LIGHT GBM
    lightgbmboruta = lightgbm
    # use Gridsearch to find the best set of HPs
    grid_lightgbmboruta =  GridSearchCV(lightgbmboruta, lgbm_param_grid)
    lgbm_boruta = grid_lightgbmboruta.fit(featarray_boruta, train_clarray) 
    # If saving:
    if saveclassifier_lgbm:
        # Don't know if joblib works
        joblib.dump(lgbm_boruta, pathlgbm_boruta)


#### Classification training with the features kept by mannwhitneyu

if os.path.exists(pathselfeat_mannwhitneyu):
    # Generate the matrix with selected feature for mannwhitney
    featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(selfeat_mannwhitneyu)
    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgemannwhitney = ridge
    ridge_mannwhitney = ridgemannwhitney.fit(featarray_mannwhitney, train_clarray)
    # If saving:
    if saveclassifier_ridge:
        joblib.dump(ridge_mannwhitney, pathridge_mannwhitney)
    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    lrmannwhitney = lr
    lr_mannwhitney = lrmannwhitney.fit(featarray_mannwhitney, train_clarray)
    # If saving:
    if saveclassifier_lr:
        joblib.dump(lr_mannwhitney, pathlr_mannwhitney)
    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestmannwhitney = forest
    # use Gridsearch to find the best set of HPs
    grid_forestmannwhitney =  GridSearchCV(forestmannwhitney, forest_param_grid)
    forest_mannwhitney = grid_forestmannwhitney.fit(featarray_mannwhitney, train_clarray)
    # If saving:
    if saveclassifier_forest:
        joblib.dump(forest_mannwhitney, pathforest_mannwhitney)
    ##### XGBOOST
    xgboostmannwhitney = xgboost
    # use Gridsearch to find the best set of HPs
    grid_xgboostmannwhitney =  GridSearchCV(xgboostmannwhitney, xgboost_param_grid)
    xgboost_mannwhitney = grid_xgboostmannwhitney.fit(featarray_mannwhitney, train_clarray)
    # If saving:
    if saveclassifier_xgboost:
        joblib.dump(xgboost_mannwhitney, pathxgboost_mannwhitney)
    ##### LIGHT GBM
    lightgbmmannwhitney = lightgbm
    # use Gridsearch to find the best set of HPs
    grid_lightgbmmannwhitney =  GridSearchCV(lightgbmmannwhitney, lgbm_param_grid)
    lgbm_mannwhitney = grid_lightgbmmannwhitney.fit(featarray_mannwhitney, train_clarray) 
    # If saving:
    if saveclassifier_lgbm:
        # Don't know if joblib works
        joblib.dump(lgbm_mannwhitney, pathlgbm_mannwhitney)


display_bestparam = False
if display_bestparam:
    print(f'\nBest parameters found by grid search for random forest - all features are: {forest_vanilla.best_params_}')
    time.sleep(2)
    print(f'\nBest parameters found by grid search for xgboost - all features are: {xgboost_vanilla.best_params_}')
    # print(f'\nBest parameters found by grid search for light gbm - all features are: {lgbm_vanilla.best_params_}')
    print(f'\nBest parameters found by grid search for random forest - mrmr features are: {forest_mrmr.best_params_}')
    time.sleep(2)
    print(f'\nBest parameters found by grid search for xgboost - mrmr features are: {xgboost_mrmr.best_params_}')
    # print(f'\nBest parameters found by grid search for light gbm - mrmr features are: {lgbm_mrmr.best_params_}')
    print(f'\nBest parameters found by grid search for random forest - boruta features are: {forest_boruta.best_params_}')
    print(f'\nBest parameters found by grid search for xgboost - boruta features are: {xgboost_boruta.best_params_}')
    # print(f'\nBest parameters found by grid search for light gbm - boruta features are: {lgbm_boruta.best_params_}')
    print(f'\nBest parameters found by grid search for random forest - mann whitney features are: {forest_mannwhitney.best_params_}')
    print(f'\nBest parameters found by grid search for xgboost - mann whitney features are: {xgboost_mannwhitney.best_params_}')
    # print(f'\nBest parameters found by grid search for light gbm - mann whitney features are: {lgbm_mannwhitney.best_params_}')




print('All classifiers trained.')
print('Classifiers saved here: ', modelfolder)
print('Classifiers saved are the ones that have saving_classifiers set as True in the config.')
