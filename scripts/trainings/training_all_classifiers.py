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
from sklearn import linear_model, ensemble, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, \
cross_validate, cross_val_score

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


### THIS SO BIG, NEED A FUNCTION TO MAKE IT READABLE

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
## Optionnal: Split data into training and test sets if not done and
############################################################

# create all the path for the classifications and feature arrays
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



###########################################################
## Load feat array and classification arrays
############################################################

#This is to check but should be fine
path_featarray = pathfeatselect + 'featarray' + ext
path_clarray = pathfeatselect + 'clarray' + ext

train_featarray = np.load(path_featarray)
train_clarray = np.load(path_clarray)
train_clarray = np.transpose(train_clarray)



############################################################
## Traininig Classifiers
############################################################


# Define the classifiers
# More information here: #https://scikit-learn.org/stable/modules/linear_model.html
##### RIDGE CLASSIFIER
ridge = linear_model.RidgeClassifier(random_state= ridge_random_state,
                                     alpha=ridge_alpha)
##### LOGISTIC REGRESSION
lr = linear_model.LogisticRegression(random_state=lregression_random_state,
                                     penalty=lregression_penalty,
                                     solver=lregression_solver,
                                     multi_class=lregression_multi_class,
                                     class_weight=lregression_class_weight)
##### RANDOM FOREST
forest = ensemble.RandomForestClassifier(random_state= forest_random_state,
                                         n_estimators=forest_n_estimators,
                                         class_weight=forest_class_weight)
##### XGBOOST
xgboost = xgboost.XGBClassifier(random_state= xgboost_random_state,
                                n_estimators=xgboost_n_estimators, 
                                learning_rate=xgboost_lr, 
                                objective=xgboost_objective,
                                verbosity=0)
##### LIGHT GBM setting
# The use of light GBM classifier is not following the convention of the other one
# Here we will save parameters needed for training, but there are no .fit method
lightgbm = lightgbm.LGBMClassifier(random_state= lgbm_random_state,
                                   n_estimators=lgbm_n_estimators,
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
ridge_param_grd = {
                    'random_state': ridge_param_grid_random_state,
                     'alpha':  ridge_param_grid_alpha
}
lregression_param_grid = {
                          'random_state': lregression_param_grid_random_state,
                          'penalty': lregression_param_grid_penalty,
                          'solver': lregression_param_grid_solver,
                          'multi_class': lregression_param_grid_multi_class,
                          'class_weight': lregression_param_grid_class_weight
}
forest_param_grid = {
                      'random_state': forest_param_grid_random_state,
                      'n_estimators': forest_param_grid_n_estimators,
                      'class_weight': forest_param_grid_class_weight
}
xgboost_param_grid = {
                      'random_state': xgboost_param_grid_random_state,
                      'n_estimators': xgboost_param_grid_n_estimators,
                      'learning_rate': xgboost_param_grid_learning_rate,
                      'objective': xgboost_param_grid_objective
}
lgbm_param_grid = {
                    'random_state': lgbm_param_grid_random_state,
                    'n_estimators': lgbm_param_grid_n_estimators,
                    'learning_rate': lgbm_param_grid_learning_rate,
                    'objective': lgbm_param_grid_objective,
                    'num_leaves': lgbm_param_grid_num_leaves
}


print('Start Classifiers trainings...')


### Create a new permutation and save it
# permutation_index = np.random.permutation(train_clarray.size)
# np.save(pathfeatselect + 'random_permutation_index_new2.npy', permutation_index)


### Load permutation index not to have 0 and 1s not mixed
permutation_index = np.load(pathfeatselect + 'random_permutation_index_best.npy')

### Shuffle classification arrays using the permutation index
train_clarray = train_clarray[permutation_index]






##fro dev
# print(sorted(metrics.SCORERS.keys()))

#### Classification training with all features kept 

if classification_from_allfeatures:
# Use all the feature (no selection) as input
    genfeatarray = np.transpose(train_featarray)

    #Shuffle feature arrays using the permutation index 
    genfeatarray = genfeatarray[permutation_index,:]


    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgevanilla = ridge
    # ridge_vanilla = ridgevanilla.fit(genfeatarray, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation 
    for paramset in ParameterGrid(ridge_param_grd):
        ridgevanilla.set_params(**paramset)
        # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        crossvalid_results = cross_val_score(ridgevanilla, 
                                             genfeatarray, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            ridgevanilla_bestset = paramset

    # If saving:
    # if saveclassifier_ridge:
    #     joblib.dump(ridge_vanilla, pathridge_vanilla)
    print('\nBest set of parameters for best_ridgevanilla is:',ridgevanilla_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### LOGISTIC REGRESSION
    # # Initialize the Logistic Regression and fit (train) it to the data
    # lrvanilla  = lr
    # # lr_vanilla = lrvanilla.fit(genfeatarray, train_clarray)
    # # use Grid Search to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestscore = 0 #cv stands for cross-validation 
    # for paramset in ParameterGrid(lregression_param_grid):
    #     lrvanilla.set_params(**paramset)
    #     # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     crossvalid_results = cross_val_score(lrvanilla, 
    #                                          genfeatarray, 
    #                                          train_clarray,  
    #                                          cv=10,
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     #save if best
    #     if crossvalid_meanscore > cv_bestscore:
    #         cv_bestscorevect = crossvalid_results
    #         cv_bestscore = crossvalid_meanscore
    #         lrvanilla_bestset = paramset

    # # If saving:
    # # if saveclassifier_lr:
    #     # joblib.dump(lr_vanilla, pathlr_vanilla)
    # print('\nBest set of parameters for best_lr_vanilla is:',lrvanilla_bestset)
    # print('The scores for all splits are:', cv_bestscorevect)
    # print('The average accuracy is:',cv_bestscore) 


    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestvanilla = forest
    # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation 
    for paramset in ParameterGrid(forest_param_grid):
        forestvanilla.set_params(**paramset)
        # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        crossvalid_results = cross_val_score(forestvanilla, 
                                             genfeatarray, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            forest_vanilla_bestset = paramset

    # If saving:
    # if saveclassifier_forest:
        # joblib.dump(best_forest_vanilla, pathforest_vanilla)
    print('\nBest set of parameters for best_forest_vanilla is:',forest_vanilla_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### XGBOOST
    xgboostvanilla = xgboost
    # xgboost_vanilla = xgboostvanilla.fit(genfeatarray, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    for paramset in ParameterGrid(xgboost_param_grid):
        xgboostvanilla.set_params(**paramset)
        # xgboost_vanilla = xgboostvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(xgboostvanilla, genfeatarray, train_clarray, cv=10)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(xgboostvanilla, 
                                             genfeatarray, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            xgboost_vanilla_bestset = paramset

    # If saving:
    # if saveclassifier_xgboost:
        # joblib.dump(best_xgboost_vanilla, pathxgboost_vanilla)
    print('\nBest set of parameters for best_xgboost_vanilla is:',xgboost_vanilla_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### LIGHT GBM
    # lgbm_traindata_vanilla = lightgbm.Dataset(genfeatarray, label=train_clarray) 
    # #lgbm_valdata_vanilla = lgbm_traindata_vanilla.create_valid()
    # lgbm_vanilla = lightgbm.train(lightgbm_paramters, 
    #                               lgbm_traindata_vanilla, 
    #                               lgbm_n_estimators)
    lightgbmvanilla = lightgbm
    # lgbm_vanilla = lightgbmvanilla.fit(genfeatarray, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(lgbm_param_grid):
        lightgbmvanilla.set_params(**paramset)
        # lgbm_vanilla = lightgbmvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(lightgbmvanilla, genfeatarray, train_clarray, cv=10)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(lightgbmvanilla, 
                                             genfeatarray, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            lgbm_vanilla_bestset = paramset

    # If saving:
    # if saveclassifier_lgbm:
        # Don't know if joblib works
        # joblib.dump(best_lgbm_vanilla, pathlgbm_vanilla)
    print('\nBest set of parameters for best_lgbm_vanilla is:',lgbm_vanilla_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore)  












#### Parse the featarray to the class SelectedFeaturesMatrix 

SelectedFeaturesMatrix = SelectedFeaturesMatrix(train_featarray)


#### Classification training with the features kept by mrmr

if os.path.exists(pathselfeat_mrmr):
    # Generate the matrix with selected feature for mrmr
    featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr(selfeat_mrmr)

    #Shuffle feature arrays using the permutation index 
    featarray_mrmr = featarray_mrmr[permutation_index,:]
  

    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgemrmr = ridge
    #ridge_mrmr = ridgemrmr.fit(featarray_mrmr, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation 
    for paramset in ParameterGrid(ridge_param_grd):
        ridgemrmr.set_params(**paramset)
        # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        crossvalid_results = cross_val_score(ridgemrmr, 
                                             featarray_mrmr, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            ridgemrmr_bestset = paramset

    # If saving:
    # if saveclassifier_ridge:
    #     joblib.dump(ridge_mrmr, pathridge_mrmr)
    print('\nBest set of parameters for best_ridgemrmr is:',ridgemrmr_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### LOGISTIC REGRESSION
    # # Initialize the Logistic Regression and fit (train) it to the data
    # lrmrmr = lr
    # #lr_mrmr = lrmrmr.fit(featarray_mrmr, train_clarray)
    # # use Grid Search to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestscore = 0 #cv stands for cross-validation 
    # for paramset in ParameterGrid(lregression_param_grid):
    #     lrmrmr.set_params(**paramset)
    #     # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     crossvalid_results = cross_val_score(lrmrmr, 
    #                                          featarray_mrmr, 
    #                                          train_clarray,  
    #                                          cv=10,  
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     #save if best
    #     if crossvalid_meanscore > cv_bestscore:
    #         cv_bestscorevect = crossvalid_results
    #         cv_bestscore = crossvalid_meanscore
    #         lrmrmr_bestset = paramset
           
    # # # If saving:
    # # if saveclassifier_lr:
    # #     joblib.dump(lr_mrmr, pathlr_mrmr)    
    # print('\nBest set of parameters for best_lrmrmr is:',lrmrmr_bestset)
    # print('The scores for all splits are:', cv_bestscorevect)
    # print('The average accuracy is:',cv_bestscore) 


    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestmrmr = forest
    # forest_mrmr = forestmrmr.fit(featarray_mrmr, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    for paramset in ParameterGrid(forest_param_grid):
        forestmrmr.set_params(**paramset)
        # forest_mrmr = forestmrmr.fit(featarray_mrmr, train_clarray)
        #evaluate the model with cross validation
        crossvalid_results = cross_val_score(forestmrmr, 
                                             featarray_mrmr, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
             cv_bestscorevect = crossvalid_results
             cv_bestscore = crossvalid_meanscore
             forest_mrmr_bestset = paramset

    # If saving:
    # if saveclassifier_forest:
        # joblib.dump(best_forest_mrmr, pathforest_mrmr)
    print('\nBest set of parameters for best_forest_mrmr is:',forest_mrmr_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 
  

    ##### XGBOOST
    xgboostmrmr = xgboost
    # xgboost_mrmr = xgboostmrmr.fit(featarray_mrmr, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    for paramset in ParameterGrid(xgboost_param_grid):
        xgboostmrmr.set_params(**paramset)
        # xgboost_mrmr = xgboostmrmr.fit(featarray_mrmr, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(xgboostmrmr, featarray_mrmr, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(xgboostmrmr, 
                                             featarray_mrmr, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscore = crossvalid_meanscore
            cv_bestscore = crossvalid_meanscore
            xgboost_mrmr_bestset = paramset

    # If saving:
    # if saveclassifier_xgboost:
        # joblib.dump(best_xgboost_mrmr, pathxgboost_mrmr)
    print('\nBest set of parameters for best_xgboost_mrmr is:',xgboost_mrmr_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### LIGHT GBM
    lightgbmmrmr = lightgbm
    # lgbm_mrmr = lightgbmmrmr.fit(featarray_mrmr, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(lgbm_param_grid):
        lightgbmmrmr.set_params(**paramset)
        # lgbm_mrmr = lightgbmmrmr.fit(featarray_mrmr, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(lightgbmmrmr, featarray_mrmr, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(lightgbmmrmr, 
                                             featarray_mrmr, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            lgbm_mrmr_bestset = paramset
           # best_lgbm_mrmr = lgbm_mrmr
    # If saving:
    # if saveclassifier_lgbm:
        # Don't know if joblib works
        # joblib.dump(best_lgbm_mrmr, pathlgbm_mrmr)
    print('\nBest set of parameters for best_lgbm_mrmr is:',lgbm_mrmr_bestset)    
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore)   









#### Classification training with the features kept by boruta

if os.path.exists(pathselfeat_boruta):
    # Generate the matrix with selected feature for boruta
    featarray_boruta = SelectedFeaturesMatrix.boruta_matr(selfeat_boruta)

    #Shuffle feature arrays using the permutation index 
    featarray_boruta = featarray_boruta[permutation_index,:]
 

    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgeboruta = ridge
    # ridge_boruta = ridgeboruta.fit(featarray_boruta, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation 
    for paramset in ParameterGrid(ridge_param_grd):
        ridgeboruta.set_params(**paramset)
        # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        crossvalid_results = cross_val_score(ridgeboruta, 
                                             featarray_boruta, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            ridgeboruta_bestset = paramset

    # # If saving:
    # if saveclassifier_ridge:
    #     joblib.dump(ridge_boruta, pathridge_boruta)
    print('\nBest set of parameters for best_ridgeboruta is:',ridgeboruta_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    # lrboruta = lr
    # # lr_boruta = lrboruta.fit(featarray_boruta, train_clarray)
    # # use Grid Search to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestscore = 0 #cv stands for cross-validation 
    # for paramset in ParameterGrid(lregression_param_grid):
    #     lrboruta.set_params(**paramset)
    #     # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     crossvalid_results = cross_val_score(lrboruta, 
    #                                          featarray_boruta, 
    #                                          train_clarray,  
    #                                          cv=10,  
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     #save if best
    #     if crossvalid_meanscore > cv_bestscore:
    #         cv_bestscorevect = crossvalid_results
    #         cv_bestscore = crossvalid_meanscore
    #         lrboruta_bestset = paramset
           
    # # # If saving:
    # # if saveclassifier_lr:
    # #     joblib.dump(lr_boruta, pathlr_boruta)
    # print('\nBest set of parameters for best_lrboruta is:',lrboruta_bestset)
    # print('The scores for all splits are:', cv_bestscorevect)
    # print('The average accuracy is:',cv_bestscore) 
 

    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestboruta = forest
    # forest_boruta = forestboruta.fit(featarray_boruta, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(forest_param_grid):
        forestboruta.set_params(**paramset)
        # forest_boruta = forestboruta.fit(featarray_boruta, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(forestboruta, featarray_boruta, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(forestboruta, 
                                             featarray_boruta, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            forest_boruta_bestset = paramset
    #        best_forest_boruta = forest_boruta
    # # If saving:
    # if saveclassifier_forest:
    #     joblib.dump(best_forest_boruta, pathforest_boruta)
    print('\nBest set of parameters for best_forest_boruta is:',forest_boruta_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 
  

    ##### XGBOOST
    xgboostboruta = xgboost
    # xgboost_boruta = xgboostboruta.fit(featarray_boruta, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(xgboost_param_grid):
        xgboostboruta.set_params(**paramset)
        # xgboost_boruta = xgboostboruta.fit(featarray_boruta, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(xgboostboruta, featarray_boruta, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(xgboostboruta, 
                                             featarray_boruta, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            xgboost_boruta_bestset = paramset
    #        best_xgboost_boruta = xgboost_boruta
    # # If saving:
    # if saveclassifier_xgboost:
    #     joblib.dump(best_xgboost_boruta, pathxgboost_boruta)
    print('\nBest set of parameters for best_xgboost_boruta is:',xgboost_boruta_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### LIGHT GBM
    lightgbmboruta = lightgbm
    # lgbm_boruta = lightgbmboruta.fit(featarray_boruta, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(lgbm_param_grid):
        lightgbmboruta.set_params(**paramset)
        # lgbm_boruta = lightgbmboruta.fit(featarray_boruta, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(lightgbmboruta, featarray_boruta, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(lightgbmboruta, 
                                             featarray_boruta, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            lgbm_boruta_bestset = paramset
    #        best_lgbm_boruta = lgbm_boruta
    # # If saving:
    # if saveclassifier_lgbm:
    #     # Don't know if joblib works
    #     joblib.dump(best_lgbm_boruta, pathlgbm_boruta)
    print('\nBest set of parameters for best_lgbm_boruta is:',lgbm_boruta_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore)  






#### Classification training with the features kept by mannwhitneyu

if os.path.exists(pathselfeat_mannwhitneyu):
    # Generate the matrix with selected feature for mannwhitney
    featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(selfeat_mannwhitneyu)
    
    #Shuffle feature arrays using the permutation index 
    featarray_mannwhitney = featarray_mannwhitney[permutation_index,:]


    ##### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier and fit (train) it to the data
    ridgemannwhitney = ridge
    # ridge_mannwhitney = ridgemannwhitney.fit(featarray_mannwhitney, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation 
    for paramset in ParameterGrid(ridge_param_grd):
        ridgemannwhitney.set_params(**paramset)
        # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        crossvalid_results = cross_val_score(ridgemannwhitney, 
                                             featarray_boruta, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            ridgemannwhitney_bestset = paramset

    # # If saving:
    # if saveclassifier_ridge:
    #     joblib.dump(ridge_mannwhitney, pathridge_mannwhitney)
    print('\nBest set of parameters for best_ridgemannwhitney is:',ridgemannwhitney_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 


    ##### LOGISTIC REGRESSION
    # Initialize the Logistic Regression and fit (train) it to the data
    # lrmannwhitney = lr
    # # lr_mannwhitney = lrmannwhitney.fit(featarray_mannwhitney, train_clarray)
    # # use Gridsearch to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestscore = 0 #cv stands for cross-validation 
    # for paramset in ParameterGrid(lregression_param_grid):
    #     lrmannwhitney.set_params(**paramset)
    #     # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     crossvalid_results = cross_val_score(lrmannwhitney, 
    #                                          featarray_mannwhitney, 
    #                                          train_clarray,  
    #                                          cv=10,  
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     #save if best
    #     if crossvalid_meanscore > cv_bestscore:
    #         cv_bestscorevect = crossvalid_results
    #         cv_bestscore = crossvalid_meanscore
    #         lrmannwhitney_bestset = paramset
           
    # # # If saving:
    # # if saveclassifier_lr:
    # #     joblib.dump(lr_mannwhitney, pathlr_mannwhitney)
    # print('\nBest set of parameters for best_lrmannwhitney is:',lrmannwhitney_bestset)
    # print('The scores for all splits are:', cv_bestscorevect)
    # print('The average accuracy is:',cv_bestscore) 


    ##### RANDOM FOREST
    # Initialize the Random Forest and fit (train) it to the data
    forestmannwhitney = forest
    # forest_mannwhitney = forestmannwhitney.fit(featarray_mannwhitney, train_clarray)
    # use Gridsearch to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(forest_param_grid):
        forestmannwhitney.set_params(**paramset)
        # forest_mannwhitney = forestmannwhitney.fit(featarray_mannwhitney, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(forestmannwhitney, featarray_mannwhitney, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(forestmannwhitney, 
                                             featarray_mannwhitney, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscore = crossvalid_meanscore
            forest_mannwhitney_bestset = paramset
            cv_bestscorevect = crossvalid_results
    #        best_forest_mannwhitney = forest_mannwhitney
    # # If saving:
    # if saveclassifier_forest:
    #     joblib.dump(best_forest_mannwhitney, pathforest_mannwhitney)
    print('\nBest set of parameter for best_forest_mannwhitney is:',forest_mannwhitney_bestset)
    print('The scores for all splits are:', crossvalid_all_scores)
    print('The average accuracy is:',crossvalid_meanscore) 
   

    ##### XGBOOST
    xgboostmannwhitney = xgboost
    # xgboost_mannwhitney = xgboostmannwhitney.fit(featarray_mannwhitney, train_clarray)
    # use Gridsearch to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(xgboost_param_grid):
        xgboostmannwhitney.set_params(**paramset)
        # xgboost_mannwhitney = xgboostmannwhitney.fit(featarray_mannwhitney, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(xgboostmannwhitney, featarray_mannwhitney, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(xgboostmannwhitney, 
                                             featarray_mannwhitney, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            xgboost_mannwhitney_bestset = paramset
    #        best_xgboost_mannwhitney = xgboost_mannwhitney
    # # If saving:
    # if saveclassifier_xgboost:
    #     joblib.dump(best_xgboost_mannwhitney, pathxgboost_mannwhitney)
    print('\nBest set of parameters for best_xgboost_mannwhitney is:',xgboost_mannwhitney_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore) 
  

    ##### LIGHT GBM
    lightgbmmannwhitney = lightgbm
    # lgbm_mannwhitney = lightgbmmannwhitney.fit(featarray_mannwhitney, train_clarray)
    # use Gridsearch to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestscore = 0 #cv stands for cross-validation
    crossvalid_all_scores = 0 
    for paramset in ParameterGrid(lgbm_param_grid):
        lightgbmmannwhitney.set_params(**paramset)
        # lgbm_mannwhitney = lightgbmmannwhitney.fit(featarray_mannwhitney, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(lightgbmmannwhitney, featarray_mannwhitney, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        crossvalid_results = cross_val_score(lightgbmmannwhitney, 
                                             featarray_mannwhitney, 
                                             train_clarray,  
                                             cv=10,  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        #save if best
        if crossvalid_meanscore > cv_bestscore:
            cv_bestscorevect = crossvalid_results
            cv_bestscore = crossvalid_meanscore
            lgbm_mannwhitney_bestset = paramset
    #        best_lgbm_mannwhitney = lgbm_mannwhitney
    # # If saving:
    # if saveclassifier_lgbm:
    #     joblib.dump(best_lgbm_mannwhitney, pathlgbm_mannwhitney)
    print('\nBest set of parameters for best_lgbm_mannwhitney is:',lgbm_mannwhitney_bestset)
    print('The scores for all splits are:', cv_bestscorevect)
    print('The average accuracy is:',cv_bestscore)


# display_bestparam = True
# if display_bestparam:
#     print(f'\nBest parameters found by grid search for random forest - all features are: {forest_vanilla.best_params_}')
#     print(f'\nBest parameters found by grid search for xgboost - all features are: {xgboost_vanilla.best_params_}')
#     print(f'\nBest parameters found by grid search for light gbm - all features are: {lgbm_vanilla.best_params_}')
#     print(f'\nBest parameters found by grid search for random forest - mrmr features are: {forest_mrmr.best_params_}')
#     print(f'\nBest parameters found by grid search for xgboost - mrmr features are: {xgboost_mrmr.best_params_}')
#     print(f'\nBest parameters found by grid search for light gbm - mrmr features are: {lgbm_mrmr.best_params_}')
#     print(f'\nBest parameters found by grid search for random forest - boruta features are: {forest_boruta.best_params_}')
#     print(f'\nBest parameters found by grid search for xgboost - boruta features are: {xgboost_boruta.best_params_}')
#     print(f'\nBest parameters found by grid search for light gbm - boruta features are: {lgbm_boruta.best_params_}')
#     print(f'\nBest parameters found by grid search for random forest - mann whitney features are: {forest_mannwhitney.best_params_}')
#     print(f'\nBest parameters found by grid search for xgboost - mann whitney features are: {xgboost_mannwhitney.best_params_}')
#     print(f'\nBest parameters found by grid search for light gbm - mann whitney features are: {lgbm_mannwhitney.best_params_}')



# print('\nAll classifiers trained.')
# print('Classifiers saved here: ', modelfolder)
# print('Classifiers saved are the ones that have saving_classifiers set as True in the config.')

print('\nCross validation of all classifiers done.')