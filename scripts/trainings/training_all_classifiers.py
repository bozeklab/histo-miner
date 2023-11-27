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



#############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(config)
pathtofolder = confighm.paths.folders.feature_selection_main
patientid_csv = confighm.paths.files.patientid_csv
patientid_avail = confighm.parameters.bool.patientid_avail
nbr_keptfeat = confighm.parameters.int.nbr_keptfeat

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
perform_split = config.parameters.bool.perform_split
split_pourcentage = config.parameters.int.split_pourcentage

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
# pathselfeat_boruta = "donnot/exist/"
if os.path.exists(pathselfeat_boruta):
    selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
pathselfeat_mannwhitneyu = pathfeatselect + 'selfeat_mannwhitneyu_idx' + ext
if os.path.exists(pathselfeat_mannwhitneyu):
    selfeat_mannwhitneyu = np.load(pathselfeat_mannwhitneyu, allow_pickle=True)
print('Loading feature selected indexes done.')

# Kept given number of features

if os.path.exists(pathselfeat_mrmr):
    selfeat_mrmr = selfeat_mrmr[0:nbr_keptfeat]
if os.path.exists(pathselfeat_mannwhitneyu):
    selfeat_mannwhitneyu = selfeat_mrmr[0:nbr_keptfeat]
print('Refinement of feature selected indexes done.')


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



################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################

#This is to check but should be fine
path_featarray = pathfeatselect + 'featarray' + ext
path_clarray = pathfeatselect + 'clarray' + ext
path_patientids_array = pathfeatselect + 'patientids' + ext

train_featarray = np.load(path_featarray)
train_clarray = np.load(path_clarray)
train_clarray = np.transpose(train_clarray)
if patientid_avail:
    patientids_load = np.load(path_patientids_array, allow_pickle=True)
    patientids_list = list(patientids_load)
    patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
    patientids = np.asarray(patientids_convert)



##############################################################
## Traininig Classifiers
##############################################################


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

# ### Load permutation index not to have 0 and 1s not mixed
# permutation_index = np.load(pathfeatselect + 'random_permutation_index_11_27_bestmean.npy')

# ### Shuffle classification arrays using the permutation index
# train_clarray = train_clarray[permutation_index]

# ### Shuffle patient IDs arrays using the permutation index 
# if patientid_avail:
#     patientids = patientids[permutation_index]

#     # Create a mapping of unique elements to positive integers
#     mapping = {}
#     current_integer = 1
#     patientids_ordered = []

#     for num in patientids:
#         if num not in mapping:
#             mapping[num] = current_integer
#             current_integer += 1
#         patientids_ordered.append(mapping[num])


### Create Stratified Group  instance to be used later 
### for the cross validation:
# sgkf = StratifiedGroupKFold(n_splits=10, shuffle=False)


for index in tqdm(range(0,200)):
   
    train_featarray = np.load(path_featarray)
    train_clarray = np.load(path_clarray)
    train_clarray = np.transpose(train_clarray)


    if patientid_avail:
        patientids_load = np.load(path_patientids_array, allow_pickle=True)
        patientids_list = list(patientids_load)
        patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
        patientids = np.asarray(patientids_convert)


    permutation_index = np.random.permutation(train_clarray.size)
    train_clarray = train_clarray[permutation_index]

    if patientid_avail:
        patientids = patientids[permutation_index]

        # Create a mapping of unique elements to positive integers
        mapping = {}
        current_integer = 1
        patientids_ordered = []

        for num in patientids:
            if num not in mapping:
                mapping[num] = current_integer
                current_integer += 1
            patientids_ordered.append(mapping[num])


#### Classification training with all features kept 

# if classification_from_allfeatures:
# Use all the feature (no selection) as input
    genfeatarray = np.transpose(train_featarray)

    #Shuffle feature arrays using the permutation index 
    genfeatarray = genfeatarray[permutation_index,:]

    # ##### RIDGE CLASSIFIER
    # # Initialize the RidgeClassifier and fit (train) it to the data
    # ridgevanilla = ridge
    # # ridge_vanilla = ridgevanilla.fit(genfeatarray, train_clarray)
    # # use Grid Search to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestmean = 0 #cv stands for cross-validation 
    # cv_bestsplit = 0
    # for paramset in ParameterGrid(ridge_param_grd):
    #     ridgevanilla.set_params(**paramset)
    #     # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     sgkf = StratifiedGroupKFold(n_splits=10, shuffle=False)
    #     cvsplit= sgkf.split(genfeatarray, train_clarray, groups=patientids)
    #     crossvalid_results = cross_val_score(ridgevanilla, 
    #                                          genfeatarray, 
    #                                          train_clarray,  
    #                                          groups=patientids,
    #                                          cv=cvsplit,  
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     crossvalid_maxscore = np.max(crossvalid_results)
    #     # Keep best trainings score and parameters set
    #     if crossvalid_maxscore > cv_bestsplit:
    #         cv_bestsplit = crossvalid_maxscore
    #         ridge_vanilla_bestmaxset = paramset
    #         cv_bestmax_scorevect = crossvalid_results
    #     if crossvalid_meanscore > cv_bestmean:
    #         cv_bestmean = crossvalid_meanscore 
    #         ridge_vanilla_bestmeanset = paramset
    #         cv_bestmean_scorevect = crossvalid_results   

    # # If saving:
    # # if saveclassifier_ridge:
    # #     joblib.dump(ridge_vanilla, pathridge_vanilla)
    # print('\n\n ** ridge_vanilla **')
    # print('The best split average accuracy is:',cv_bestsplit)  
    # print('Corresponding set of parameters for ridge_vanilla_bestmaxset is:',
    #         ridge_vanilla_bestmaxset)
    # print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
    # print('The best mean average accuracy is:',cv_bestmean)
    # print('Corresponding set of parameters for ridge_vanilla_bestmeanset is:',
    #         ridge_vanilla_bestmeanset)
    # print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


    # ##### LOGISTIC REGRESSION
    # # Initialize the Logistic Regression and fit (train) it to the data
    # lrvanilla  = lr
    # # lr_vanilla = lrvanilla.fit(genfeatarray, train_clarray)
    # # use Grid Search to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestmean = 0 #cv stands for cross-validation 
    # cv_bestsplit = 0
    # for paramset in ParameterGrid(lregression_param_grid):
    #     lrvanilla.set_params(**paramset)
    #     # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     sgkf = StratifiedGroupKFold(n_splits=10, shuffle=False)
    #     cvsplit= sgkf.split(genfeatarray, train_clarray, groups=patientids)
    #     crossvalid_results = cross_val_score(lrvanilla, 
    #                                          genfeatarray, 
    #                                          train_clarray,  
    #                                          groups=patientids,
    #                                          cv=cvsplit,  
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     crossvalid_maxscore = np.max(crossvalid_results)
    #     # Keep best trainings score and parameters set
    #     if crossvalid_maxscore > cv_bestsplit:
    #         cv_bestsplit = crossvalid_maxscore
    #         lr_vanilla_bestmaxset = paramset
    #         cv_bestmax_scorevect = crossvalid_results
    #     if crossvalid_meanscore > cv_bestmean:
    #         cv_bestmean = crossvalid_meanscore 
    #         lr_vanilla_bestmeanset = paramset
    #         cv_bestmean_scorevect = crossvalid_results   

    # # If saving:
    # # if saveclassifier_ridge:
    # #     joblib.dump(ridge_vanilla, pathridge_vanilla)
    # print('\n\n ** lr_vanilla **')
    # print('The best split average accuracy is:',cv_bestsplit)  
    # print('Corresponding set of parameters for lr_vanilla_bestmaxset is:',
    #         lr_vanilla_bestmaxset)
    # print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
    # print('The best mean average accuracy is:',cv_bestmean)
    # print('Corresponding set of parameters for lr_vanilla_bestmeanset is:',
    #         lr_vanilla_bestmeanset)
    # print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


    ##### RANDOM FOREST
    # # Initialize the Random Forest and fit (train) it to the data
    # forestvanilla = forest
    # # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    # # use Grid Search to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestmean = 0 #cv stands for cross-validation 
    # cv_bestsplit = 0
    # for paramset in ParameterGrid(forest_param_grid):
    #     forestvanilla.set_params(**paramset)
    #     # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     sgkf = StratifiedGroupKFold(n_splits=10, shuffle=False)
    #     cvsplit= sgkf.split(genfeatarray, train_clarray, groups=patientids)
    #     crossvalid_results = cross_val_score(forestvanilla, 
    #                                          genfeatarray, 
    #                                          train_clarray,  
    #                                          groups=patientids,
    #                                          cv=cvsplit,  
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     crossvalid_maxscore = np.max(crossvalid_results)
    #     # Keep best trainings score and parameters set
    #     if crossvalid_maxscore > cv_bestsplit:
    #         cv_bestsplit = crossvalid_maxscore
    #         forest_vanilla_bestmaxset = paramset
    #         cv_bestmax_scorevect = crossvalid_results
    #     if crossvalid_meanscore > cv_bestmean:
    #         cv_bestmean = crossvalid_meanscore 
    #         forest_vanilla_bestmeanset = paramset
    #         cv_bestmean_scorevect = crossvalid_results   

    # # If saving:
    # # if saveclassifier_ridge:
    # #     joblib.dump(ridge_vanilla, pathridge_vanilla)
    # print('\n\n ** forest_vanilla **')
    # print('The best split average accuracy is:',cv_bestsplit)  
    # print('Corresponding set of parameters for forest_vanilla_bestmaxset is:',
    #         forest_vanilla_bestmaxset)
    # print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
    # print('The best mean average accuracy is:',cv_bestmean)
    # print('Corresponding set of parameters for forest_vanilla_bestmeanset is:',
    #         forest_vanilla_bestmeanset)
    # print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


    ##### XGBOOST
    xgboostvanilla = xgboost
    # xgboost_vanilla = xgboostvanilla.fit(genfeatarray, train_clarray)
    # use Grid Search to find the best set of HPs 
    # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    cv_bestmean = 0 #cv stands for cross-validation 
    cv_bestsplit = 0
    cv_bestmean_perm = 0
    bestmean_permutation_index = []
    cv_bestsplit_perm = 0
    bestsplit_permutation_index = []
    for paramset in ParameterGrid(xgboost_param_grid):
        xgboostvanilla.set_params(**paramset)
        # xgboost_vanilla = xgboostvanilla.fit(genfeatarray, train_clarray)
        #evaluate the model with cross validation
        # crossvalid_results = cross_validate(xgboostvanilla, genfeatarray, train_clarray, cv=5)
        # crossvalid_all_scores = crossvalid_results['test_score']
        # sgkf = StratifiedGroupKFold(n_splits=10, shuffle=False)
        # cvsplit= sgkf.split(genfeatarray, train_clarray, groups=patientids)
        crossvalid_results = cross_val_score(xgboostvanilla, 
                                             genfeatarray, 
                                             train_clarray,  
                                             groups=patientids_ordered,
                                             cv=StratifiedGroupKFold(n_splits=10, shuffle=False),  
                                             scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        crossvalid_maxscore = np.max(crossvalid_results)
        # Keep best trainings score and parameters set
        if crossvalid_maxscore > cv_bestsplit:
            cv_bestsplit = crossvalid_maxscore
            xgboost_vanilla_bestmaxset = paramset
            cv_bestmax_scorevect = crossvalid_results
        if crossvalid_meanscore > cv_bestmean:
            cv_bestmean = crossvalid_meanscore 
            xgboost_vanilla_bestmeanset = paramset
            cv_bestmean_scorevect = crossvalid_results
    if cv_bestmean > cv_bestmean_perm:
        cv_bestmean_perm = cv_bestmean
        bestmean_permutation_index = permutation_index
    if cv_bestsplit > cv_bestsplit_perm:
        cv_bestsplit_perm = cv_bestsplit
        bestsplit_permutation_index = permutation_index

    # If saving:
    # if saveclassifier_ridge:
    #     joblib.dump(ridge_vanilla, pathridge_vanilla)
    print('\n\n ** xgboost_vanilla **')
    print('The best split average accuracy is:',cv_bestsplit)  
    print('Corresponding set of parameters for xgboost_vanilla_bestmaxset is:',
            xgboost_vanilla_bestmaxset)
    print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
    print('The best mean average accuracy is:',cv_bestmean)
    print('Corresponding set of parameters for xgboost_vanilla_bestmeanset is:',
            xgboost_vanilla_bestmeanset)
    print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


    #### LIGHT GBM
    # # lgbm_traindata_vanilla = lightgbm.Dataset(genfeatarray, label=train_clarray) 
    # # #lgbm_valdata_vanilla = lgbm_traindata_vanilla.create_valid()
    # # lgbm_vanilla = lightgbm.train(lightgbm_paramters, 
    # #                               lgbm_traindata_vanilla, 
    # #                               lgbm_n_estimators)
    # lightgbmvanilla = lightgbm
    # # lgbm_vanilla = lightgbmvanilla.fit(genfeatarray, train_clarray)
    # # use Grid Search to find the best set of HPs 
    # # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
    # cv_bestmean = 0 #cv stands for cross-validation 
    # cv_bestsplit = 0 
    # for paramset in ParameterGrid(lgbm_param_grid):
    #     lightgbmvanilla.set_params(**paramset)
    #     # lgbm_vanilla = lightgbmvanilla.fit(genfeatarray, train_clarray)
    #     #evaluate the model with cross validation
    #     # crossvalid_results = cross_validate(lightgbmvanilla, genfeatarray, train_clarray, cv=5)
    #     # crossvalid_all_scores = crossvalid_results['test_score']
    #     cvsplit= sgkf.split(genfeatarray, train_clarray, groups=patientids)
    #     crossvalid_results = cross_val_score(lightgbmvanilla, 
    #                                          genfeatarray, 
    #                                          train_clarray,  
    #                                          groups=patientids,
    #                                          cv=cvsplit,  
    #                                          scoring='balanced_accuracy')
    #     crossvalid_meanscore = np.mean(crossvalid_results)
    #     crossvalid_maxscore = np.max(crossvalid_results)
    #     # Keep best trainings score and parameters set
    #     if crossvalid_maxscore > cv_bestsplit:
    #         cv_bestsplit = crossvalid_maxscore
    #         lgbm_vanilla_bestmaxset = paramset
    #         cv_bestmax_scorevect = crossvalid_results
    #     if crossvalid_meanscore > cv_bestmean:
    #         cv_bestmean = crossvalid_meanscore 
    #         lgbm_vanilla_bestmeanset = paramset
    #         cv_bestmean_scorevect = crossvalid_results   

    # # If saving:
    # # if saveclassifier_ridge:
    # #     joblib.dump(ridge_vanilla, pathridge_vanilla) print('\n\n ** lgbm__vanilla **')
    # print('\n\n ** lightgbm_vanilla **')
    # print('The best split average accuracy is:',cv_bestsplit)  
    # print('Corresponding set of parameters for lgbm_vanilla_bestmaxset is:',
    #         lgbm_vanilla_bestmaxset)
    # print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
    # print('The best mean average accuracy is:',cv_bestmean)
    # print('Corresponding set of parameters for lgbm_vanilla_bestmeanset is:',
    #         lgbm_vanilla_bestmeanset)
    # print('Corresponding scores for all splits are:', cv_bestmean_scorevect)









# #### Parse the featarray to the class SelectedFeaturesMatrix 

# SelectedFeaturesMatrix = SelectedFeaturesMatrix(train_featarray)


# #### Classification training with the features kept by mrmr

# if os.path.exists(pathselfeat_mrmr):
#     # Generate the matrix with selected feature for mrmr
#     featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr(selfeat_mrmr)

#     #Shuffle feature arrays using the permutation index 
#     featarray_mrmr = featarray_mrmr[permutation_index,:]
  

#     ##### RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier and fit (train) it to the data
#     ridgemrmr = ridge
#     #ridge_mrmr = ridgemrmr.fit(featarray_mrmr, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(ridge_param_grd):
#         ridgemrmr.set_params(**paramset)
#         # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
#         #evaluate the model with cross validation
#         cvsplit= sgkf.split(featarray_mrmr, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(ridgemrmr, 
#                                              featarray_mrmr, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             ridge_mrmr_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             ridge_mrmr_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** ridge_mrmr **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for ridge_mrmr_bestmaxset is:',
#             ridge_mrmr_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for ridge_mrmr_bestmeanset is:',
#             ridge_mrmr_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### LOGISTIC REGRESSION
#     # Initialize the Logistic Regression and fit (train) it to the data
#     lrmrmr = lr
#     #lr_mrmr = lrmrmr.fit(featarray_mrmr, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(lregression_param_grid):
#         lrmrmr.set_params(**paramset)
#         # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
#         #evaluate the model with cross validation
#         cvsplit= sgkf.split(featarray_mrmr, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(lrmrmr, 
#                                              featarray_mrmr, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             lr_mrmr_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             lr_mrmr_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** lr_mrmr **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for lr_mrmr_bestmaxset is:',
#             lr_mrmr_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for lr_mrmr_bestmeanset is:',
#             lr_mrmr_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### RANDOM FOREST
#     # Initialize the Random Forest and fit (train) it to the data
#     forestmrmr = forest
#     # forest_mrmr = forestmrmr.fit(featarray_mrmr, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(forest_param_grid):
#         forestmrmr.set_params(**paramset)
#         # forest_mrmr = forestmrmr.fit(featarray_mrmr, train_clarray)
#         #evaluate the model with cross validation
#         cvsplit= sgkf.split(featarray_mrmr, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(forestmrmr, 
#                                              featarray_mrmr, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             forest_mrmr_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             forest_mrmr_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** forest_mrmr **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for forest_mrmr_bestmaxset is:',
#             forest_mrmr_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for forest_mrmr_bestmeanset is:',
#             forest_mrmr_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)
  

#     ##### XGBOOST
#     xgboostmrmr = xgboost
#     # xgboost_mrmr = xgboostmrmr.fit(featarray_mrmr, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(xgboost_param_grid):
#         xgboostmrmr.set_params(**paramset)
#         # xgboost_mrmr = xgboostmrmr.fit(featarray_mrmr, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(xgboostmrmr, featarray_mrmr, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         cvsplit= sgkf.split(featarray_mrmr, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(xgboostmrmr, 
#                                              featarray_mrmr, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             xgboost_mrmr_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             xgboost_mrmr_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** xgboost_mrmr **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for xgboost_mrmr_bestmaxset is:',
#             xgboost_mrmr_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for xgboost_mrmr_bestmeanset is:',
#             xgboost_mrmr_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### LIGHT GBM
#     lightgbmmrmr = lightgbm
#     # lgbm_mrmr = lightgbmmrmr.fit(featarray_mrmr, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(lgbm_param_grid):
#         lightgbmmrmr.set_params(**paramset)
#         # lgbm_mrmr = lightgbmmrmr.fit(featarray_mrmr, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(lightgbmmrmr, featarray_mrmr, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         cvsplit= sgkf.split(featarray_mrmr, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(lightgbmmrmr, 
#                                              featarray_mrmr, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             lgbm_mrmr_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             lgbm_mrmr_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** lgbm_mrmr **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for lgbm_mrmr_bestmaxset is:',
#             lgbm_mrmr_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for lgbm_mrmr_bestmeanset is:',
#             lgbm_mrmr_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)










# #### Classification training with the features kept by boruta

# if os.path.exists(pathselfeat_boruta):
#     # Generate the matrix with selected feature for boruta
#     featarray_boruta = SelectedFeaturesMatrix.boruta_matr(selfeat_boruta)

#     #Shuffle feature arrays using the permutation index 
#     featarray_boruta = featarray_boruta[permutation_index,:]
 

#     ##### RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier and fit (train) it to the data
#     ridgeboruta = ridge
#     # ridge_boruta = ridgeboruta.fit(featarray_boruta, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(ridge_param_grd):
#         ridgeboruta.set_params(**paramset)
#         # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
#         #evaluate the model with cross validation
#         cvsplit= sgkf.split(featarray_boruta, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(ridgeboruta, 
#                                              featarray_boruta, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             ridge_boruta_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             ridge_boruta_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** ridge_boruta **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for ridge_boruta_bestmaxset is:',
#             ridge_boruta_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for ridge_boruta_bestmeanset is:',
#             ridge_boruta_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### LOGISTIC REGRESSION
#     # Initialize the Logistic Regression and fit (train) it to the data
#     lrboruta = lr
#     # lr_boruta = lrboruta.fit(featarray_boruta, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(lregression_param_grid):
#         lrboruta.set_params(**paramset)
#         # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
#         #evaluate the model with cross validation
#         cvsplit= sgkf.split(featarray_boruta, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(lrboruta, 
#                                              featarray_boruta, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             lr_boruta_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             lr_boruta_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** lr_boruta **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for lr_boruta_bestmaxset is:',
#             lr_boruta_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for lr_boruta_bestmeanset is:',
#             lr_boruta_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)
 

#     ##### RANDOM FOREST
#     # Initialize the Random Forest and fit (train) it to the data
#     forestboruta = forest
#     # forest_boruta = forestboruta.fit(featarray_boruta, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(forest_param_grid):
#         forestboruta.set_params(**paramset)
#         # forest_boruta = forestboruta.fit(featarray_boruta, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(forestboruta, featarray_boruta, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         cvsplit= sgkf.split(featarray_boruta, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(forestboruta, 
#                                              featarray_boruta, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             forest_boruta_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             forest_boruta_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** forest_boruta **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for forest_boruta_bestmaxset is:',
#             forest_boruta_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for forest_boruta_bestmeanset is:',
#             forest_boruta_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)
  

#     ##### XGBOOST
#     xgboostboruta = xgboost
#     # xgboost_boruta = xgboostboruta.fit(featarray_boruta, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(xgboost_param_grid):
#         xgboostboruta.set_params(**paramset)
#         cvsplit= sgkf.split(featarray_boruta, train_clarray, groups=patientids)
#         # xgboost_boruta = xgboostboruta.fit(featarray_boruta, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(xgboostboruta, featarray_boruta, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         crossvalid_results = cross_val_score(xgboostboruta, 
#                                              featarray_boruta, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             xgboost_boruta_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             xgboost_boruta_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** xgboost_boruta **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for xgboost_boruta_bestmaxset is:',
#             xgboost_boruta_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for xgboost_boruta_bestmeanset is:',
#             xgboost_boruta_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### LIGHT GBM
#     lightgbmboruta = lightgbm
#     # lgbm_boruta = lightgbmboruta.fit(featarray_boruta, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(lgbm_param_grid):
#         lightgbmboruta.set_params(**paramset)
#         # lgbm_boruta = lightgbmboruta.fit(featarray_boruta, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(lightgbmboruta, featarray_boruta, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         cvsplit= sgkf.split(featarray_boruta, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(lightgbmboruta, 
#                                              featarray_boruta, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             lgbm_boruta_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             lgbm_boruta_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** lgbm_boruta **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for lgbm_boruta_bestmaxset is:',
#             lgbm_boruta_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for lgbm_boruta_bestmeanset is:',
#             lgbm_boruta_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)










# #### Classification training with the features kept by mannwhitneyu

# if os.path.exists(pathselfeat_mannwhitneyu):
#     # Generate the matrix with selected feature for mannwhitney
#     featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(selfeat_mannwhitneyu)
    
#     #Shuffle feature arrays using the permutation index 
#     featarray_mannwhitney = featarray_mannwhitney[permutation_index,:]


#     ##### RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier and fit (train) it to the data
#     ridgemannwhitney = ridge
#     # ridge_mannwhitney = ridgemannwhitney.fit(featarray_mannwhitney, train_clarray)
#     # use Grid Search to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(ridge_param_grd):
#         ridgemannwhitney.set_params(**paramset)
#         # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
#         #evaluate the model with cross validation
#         cvsplit= sgkf.split(featarray_mannwhitney, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(ridgemannwhitney, 
#                                              featarray_mannwhitney, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             ridge_mannwhitney_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             ridge_mannwhitney_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** ridge_mannwhitney **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for ridge_mannwhitney_bestmaxset is:',
#             ridge_mannwhitney_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for ridge_mannwhitney_bestmeanset is:',
#             ridge_mannwhitney_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### LOGISTIC REGRESSION
#     # Initialize the Logistic Regression and fit (train) it to the data
#     lrmannwhitney = lr
#     # lr_mannwhitney = lrmannwhitney.fit(featarray_mannwhitney, train_clarray)
#     # use Gridsearch to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(lregression_param_grid):
#         lrmannwhitney.set_params(**paramset)
#         # forest_vanilla = forestvanilla.fit(genfeatarray, train_clarray)
#         #evaluate the model with cross validation
#         cvsplit= sgkf.split(featarray_mannwhitney, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(lrmannwhitney, 
#                                              featarray_mannwhitney, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             lr_mannwhitney_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             lr_mannwhitney_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** lr_mannwhitney **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for lr_mannwhitney_bestmaxset is:',
#             lr_mannwhitney_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for lr_mannwhitney_bestmeanset is:',
#             lr_mannwhitney_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### RANDOM FOREST
#     # Initialize the Random Forest and fit (train) it to the data
#     forestmannwhitney = forest
#     # forest_mannwhitney = forestmannwhitney.fit(featarray_mannwhitney, train_clarray)
#     # use Gridsearch to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(forest_param_grid):
#         forestmannwhitney.set_params(**paramset)
#         cvsplit= sgkf.split(featarray_mannwhitney, train_clarray, groups=patientids)
#         # forest_mannwhitney = forestmannwhitney.fit(featarray_mannwhitney, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(forestmannwhitney, featarray_mannwhitney, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         crossvalid_results = cross_val_score(forestmannwhitney, 
#                                              featarray_mannwhitney, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             forest_mannwhitney_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             forest_mannwhitney_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** forest_mannwhitney **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for forest_mannwhitney_bestmaxset is:',
#             forest_mannwhitney_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for forest_mannwhitney_bestmeanset is:',
#             forest_mannwhitney_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#     ##### XGBOOST
#     xgboostmannwhitney = xgboost
#     # xgboost_mannwhitney = xgboostmannwhitney.fit(featarray_mannwhitney, train_clarray)
#     # use Gridsearch to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0
#     for paramset in ParameterGrid(xgboost_param_grid):
#         xgboostmannwhitney.set_params(**paramset)
#         # xgboost_mannwhitney = xgboostmannwhitney.fit(featarray_mannwhitney, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(xgboostmannwhitney, featarray_mannwhitney, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         cvsplit= sgkf.split(featarray_mannwhitney, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(xgboostmannwhitney, 
#                                              featarray_mannwhitney, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             xgboost_mannwhitney_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             xgboost_mannwhitney_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** xgboost_mannwhitney **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for xgboost_mannwhitney_bestmaxset is:',
#             xgboost_mannwhitney_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for xgboost_mannwhitney_bestmeanset is:',
#             xgboost_mannwhitney_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)
  

#     ##### LIGHT GBM
#     lightgbmmannwhitney = lightgbm
#     # lgbm_mannwhitney = lightgbmmannwhitney.fit(featarray_mannwhitney, train_clarray)
#     # use Gridsearch to find the best set of HPs 
#     # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
#     cv_bestmean = 0 #cv stands for cross-validation 
#     cv_bestsplit = 0 
#     for paramset in ParameterGrid(lgbm_param_grid):
#         lightgbmmannwhitney.set_params(**paramset)
#         # lgbm_mannwhitney = lightgbmmannwhitney.fit(featarray_mannwhitney, train_clarray)
#         #evaluate the model with cross validation
#         # crossvalid_results = cross_validate(lightgbmmannwhitney, featarray_mannwhitney, train_clarray, cv=5)
#         # crossvalid_all_scores = crossvalid_results['test_score']
#         cvsplit= sgkf.split(featarray_mannwhitney, train_clarray, groups=patientids)
#         crossvalid_results = cross_val_score(lightgbmmannwhitney, 
#                                              featarray_mannwhitney, 
#                                              train_clarray,  
#                                              groups=patientids,
#                                              cv=cvsplit,  
#                                              scoring='balanced_accuracy')
#         crossvalid_meanscore = np.mean(crossvalid_results)
#         crossvalid_maxscore = np.max(crossvalid_results)
#         # Keep best trainings score and parameters set
#         if crossvalid_maxscore > cv_bestsplit:
#             cv_bestsplit = crossvalid_maxscore
#             lgbm_mannwhitney_bestmaxset = paramset
#             cv_bestmax_scorevect = crossvalid_results
#         if crossvalid_meanscore > cv_bestmean:
#             cv_bestmean = crossvalid_meanscore 
#             lgbm_mannwhitney_bestmeanset = paramset
#             cv_bestmean_scorevect = crossvalid_results   

#     # If saving:
#     # if saveclassifier_ridge:
#     #     joblib.dump(ridge_vanilla, pathridge_vanilla)
#     print('\n\n ** lgbm_mannwhitney **')
#     print('The best split average accuracy is:',cv_bestsplit)  
#     print('Corresponding set of parameters for lgbm_mannwhitney_bestmaxset is:',
#             lgbm_mannwhitney_bestmaxset)
#     print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
#     print('The best mean average accuracy is:',cv_bestmean)
#     print('Corresponding set of parameters for lgbm_mannwhitney_bestmeanset is:',
#             lgbm_mannwhitney_bestmeanset)
#     print('Corresponding scores for all splits are:', cv_bestmean_scorevect)



np.save(pathfeatselect + 'random_permutation_index_11_28_bestmean.npy', bestmean_permutation_index)
np.save(pathfeatselect + 'random_permutation_index_11_28_bestsplit.npy', bestsplit_permutation_index)

print('\nAll classifiers trained.')
print('Classifiers saved here: ', modelfolder)
print('Classifiers saved are the ones that have saving_classifiers set as True in the config.')

print('\nCross validation of all classifiers done.')