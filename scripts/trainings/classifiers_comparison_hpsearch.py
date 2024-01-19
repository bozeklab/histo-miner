#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn import linear_model, ensemble
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedGroupKFold

import src.histo_miner.utils.misc as utils_misc

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
classification_eval_folder = confighm.paths.folders.classification_evaluation
patientid_avail = confighm.parameters.bool.patientid_avail

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
save_evaluations = config.parameters.bool.save_evaluations

ridge_param_grid_random_state = list(config.classifierparam.ridge.grid_dict.random_state)
ridge_param_grid_alpha = list(config.classifierparam.ridge.grid_dict.alpha)

forest_param_grid_random_state = list(config.classifierparam.random_forest.grid_dict.random_state)
forest_param_grid_n_estimators = list(config.classifierparam.random_forest.grid_dict.n_estimators)
forest_param_grid_class_weight = list(config.classifierparam.random_forest.grid_dict.class_weight)

xgboost_param_grid_random_state = list(config.classifierparam.xgboost.grid_dict.random_state)
xgboost_param_grid_n_estimators = list(config.classifierparam.xgboost.grid_dict.n_estimators)
xgboost_param_grid_learning_rate = list(config.classifierparam.xgboost.grid_dict.learning_rate)
xgboost_param_grid_objective = list(config.classifierparam.xgboost.grid_dict.objective)

lgbm_param_grid_random_state = list(config.classifierparam.light_gbm.grid_dict.random_state)
lgbm_param_grid_n_estimators = list(config.classifierparam.light_gbm.grid_dict.n_estimators)
lgbm_param_grid_learning_rate = list(config.classifierparam.light_gbm.grid_dict.learning_rate)
lgbm_param_grid_objective = list(config.classifierparam.light_gbm.grid_dict.objective)
lgbm_param_grid_num_leaves = list(config.classifierparam.light_gbm.grid_dict.num_leaves)

lregression_param_grid_random_state = list(
    config.classifierparam.logistic_regression.grid_dict.random_state)
lregression_param_grid_penalty = list(
    config.classifierparam.logistic_regression.grid_dict.penalty)
lregression_param_grid_solver = list(
    config.classifierparam.logistic_regression.grid_dict.solver)
lregression_param_grid_multi_class = list(
    config.classifierparam.logistic_regression.grid_dict.multi_class)
lregression_param_grid_class_weight = list(
    config.classifierparam.logistic_regression.grid_dict.class_weight)



################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################


pathfeatselect = pathtofolder + '' + '/feature_selection/'
ext = '.npy'

path_featarray = pathfeatselect + 'featarray' + ext
path_clarray = pathfeatselect + 'clarray' + ext
path_patientids_array = pathfeatselect + 'patientids' + ext

train_featarray = np.load(path_featarray)
train_clarray = np.load(path_clarray)
train_clarray = np.transpose(train_clarray)
print('Feature feature arrays and class arrays loaded')

if patientid_avail:
    patientids_load = np.load(path_patientids_array, allow_pickle=True)
    patientids_list = list(patientids_load)
    # patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
    # patientids = np.asarray(patientids_convert)
    print('Patient IDs loaded')

path_save_results = classification_eval_folder + 'classifiers_comparison_hpsearch.txt'



##############################################################
## Hyperparemeter Search and Cross-Validation of Classifiers
##############################################################


# Define the classifiers
# More information here: #https://scikit-learn.org/stable/modules/linear_model.html
##### RIDGE CLASSIFIER
ridge = linear_model.RidgeClassifier()

##### LOGISTIC REGRESSION
lr = linear_model.LogisticRegression()

##### RANDOM FOREST
forest = ensemble.RandomForestClassifier()

##### XGBOOST
xgboost = xgboost.XGBClassifier(verbosity=0)

##### LIGHT GBM setting
lightgbm = lightgbm.LGBMClassifier(verbosity=-1)


#RMQ: Verbosity is set to 0 for XGBOOST to avoid printing WARNINGS (not wanted here for sake of
#simplicity)/ In Light GBM, to avoid showing WARNINGS, the verbosity as to be set to -1.
# See parameters documentation to learn about the other verbosity available. 


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
permutation_index = np.load(pathfeatselect + 
                            '/bestperm/' +
                            'random_permutation_index_11_28_xgboost_bestmean.npy')

### Shuffle classification arrays using the permutation index
train_clarray = train_clarray[permutation_index]

### Shuffle patient IDs arrays using the permutation index 
if patientid_avail:
    patientids = patientids_list[permutation_index]
    # Create a mapping of unique elements to positive integers
    patientids_ordered = utils_misc.convert_names_to_orderedint(patientids)


### Create Stratified Group  instance for the cross validation 
stratgroupkf = StratifiedGroupKFold(n_splits=10, shuffle=False)


#### Classification training with all features kept 

# Use all the feature (no selection) as input
genfeatarray = np.transpose(train_featarray)

#Shuffle feature arrays using the permutation index 
genfeatarray = genfeatarray[permutation_index,:]

##### RIDGE CLASSIFIER
# Initialize the RidgeClassifier and fit (train) it to the data
ridgevanilla = ridge
# use Grid Search to find the best set of HPs 
# (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
cv_bestmean = 0 #cv stands for cross-validation 
cv_bestsplit = 0
for paramset in ParameterGrid(ridge_param_grd):
    ridgevanilla.set_params(**paramset)
    # Evaluate the model with cross validation
    crossvalid_results = cross_val_score(ridgevanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         groups=patientids_ordered,
                                         cv=stratgroupkf,  
                                         scoring='balanced_accuracy')
    crossvalid_meanscore = np.mean(crossvalid_results)
    crossvalid_maxscore = np.max(crossvalid_results)
    # Keep best trainings score and parameters set
    if crossvalid_maxscore > cv_bestsplit:
        cv_bestsplit = crossvalid_maxscore
        ridge_vanilla_bestmaxset = paramset
        cv_bestmax_scorevect = crossvalid_results
    if crossvalid_meanscore > cv_bestmean:
        cv_bestmean = crossvalid_meanscore 
        ridge_vanilla_bestmeanset = paramset
        cv_bestmean_scorevect = crossvalid_results   

print('\n\n ** ridge_vanilla **')
print('The best split average accuracy is:',cv_bestsplit)  
print('Corresponding set of parameters for ridge_vanilla_bestmaxset is:',
        ridge_vanilla_bestmaxset)
print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
print('The best mean average accuracy is:',cv_bestmean)
print('Corresponding set of parameters for ridge_vanilla_bestmeanset is:',
        ridge_vanilla_bestmeanset)
print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


##### LOGISTIC REGRESSION
# Initialize the Logistic Regression and fit (train) it to the data
lrvanilla  = lr
# use Grid Search to find the best set of HPs 
# (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
cv_bestmean = 0 #cv stands for cross-validation 
cv_bestsplit = 0
for paramset in ParameterGrid(lregression_param_grid):
    lrvanilla.set_params(**paramset)
    # Evaluate the model with cross validation
    crossvalid_results = cross_val_score(lrvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         groups=patientids_ordered,
                                         cv=stratgroupkf,  
                                         scoring='balanced_accuracy')
    crossvalid_meanscore = np.mean(crossvalid_results)
    crossvalid_maxscore = np.max(crossvalid_results)
    # Keep best trainings score and parameters set
    if crossvalid_maxscore > cv_bestsplit:
        cv_bestsplit = crossvalid_maxscore
        lr_vanilla_bestmaxset = paramset
        cv_bestmax_scorevect = crossvalid_results
    if crossvalid_meanscore > cv_bestmean:
        cv_bestmean = crossvalid_meanscore 
        lr_vanilla_bestmeanset = paramset
        cv_bestmean_scorevect = crossvalid_results   

print('\n\n ** lr_vanilla **')
print('The best split average accuracy is:',cv_bestsplit)  
print('Corresponding set of parameters for lr_vanilla_bestmaxset is:',
        lr_vanilla_bestmaxset)
print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
print('The best mean average accuracy is:',cv_bestmean)
print('Corresponding set of parameters for lr_vanilla_bestmeanset is:',
        lr_vanilla_bestmeanset)
print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


# #### RANDOM FOREST
# Initialize the Random Forest and fit (train) it to the data
forestvanilla = forest
# use Grid Search to find the best set of HPs 
# (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
cv_bestmean = 0 #cv stands for cross-validation 
cv_bestsplit = 0
for paramset in ParameterGrid(forest_param_grid):
    forestvanilla.set_params(**paramset)
    # Evaluate the model with cross validation
    crossvalid_results = cross_val_score(forestvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         groups=patientids_ordered,
                                         cv=stratgroupkf,  
                                         scoring='balanced_accuracy')
    crossvalid_meanscore = np.mean(crossvalid_results)
    crossvalid_maxscore = np.max(crossvalid_results)
    # Keep best trainings score and parameters set
    if crossvalid_maxscore > cv_bestsplit:
        cv_bestsplit = crossvalid_maxscore
        forest_vanilla_bestmaxset = paramset
        cv_bestmax_scorevect = crossvalid_results
    if crossvalid_meanscore > cv_bestmean:
        cv_bestmean = crossvalid_meanscore 
        forest_vanilla_bestmeanset = paramset
        cv_bestmean_scorevect = crossvalid_results   

print('\n\n ** forest_vanilla **')
print('The best split average accuracy is:',cv_bestsplit)  
print('Corresponding set of parameters for forest_vanilla_bestmaxset is:',
        forest_vanilla_bestmaxset)
print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
print('The best mean average accuracy is:',cv_bestmean)
print('Corresponding set of parameters for forest_vanilla_bestmeanset is:',
        forest_vanilla_bestmeanset)
print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#### XGBOOST
xgboostvanilla = xgboost
# use Grid Search to find the best set of HPs 
# (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
cv_bestmean = 0 #cv stands for cross-validation 
cv_bestsplit = 0
for paramset in ParameterGrid(xgboost_param_grid):
    xgboostvanilla.set_params(**paramset)
    # Evaluate the model with cross validation
    crossvalid_results = cross_val_score(xgboostvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         groups=patientids_ordered,
                                         cv=stratgroupkf,  
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
# lgbm_traindata_vanilla = lightgbm.Dataset(genfeatarray, label=train_clarray) 
# #lgbm_valdata_vanilla = lgbm_traindata_vanilla.create_valid()
# lgbm_vanilla = lightgbm.train(lightgbm_paramters, 
#                               lgbm_traindata_vanilla, 
#                               lgbm_n_estimators)
lightgbmvanilla = lightgbm
# lgbm_vanilla = lightgbmvanilla.fit(genfeatarray, train_clarray)
# use Grid Search to find the best set of HPs 
# (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
cv_bestmean = 0 #cv stands for cross-validation 
cv_bestsplit = 0 
for paramset in ParameterGrid(lgbm_param_grid):
    lightgbmvanilla.set_params(**paramset)
    # Evaluate the model with cross validation
    crossvalid_results = cross_val_score(lightgbmvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         groups=patientids_ordered,
                                         cv=stratgroupkf,  
                                         scoring='balanced_accuracy')
    crossvalid_meanscore = np.mean(crossvalid_results)
    crossvalid_maxscore = np.max(crossvalid_results)
    # Keep best trainings score and parameters set
    if crossvalid_maxscore > cv_bestsplit:
        cv_bestsplit = crossvalid_maxscore
        lgbm_vanilla_bestmaxset = paramset
        cv_bestmax_scorevect = crossvalid_results
    if crossvalid_meanscore > cv_bestmean:
        cv_bestmean = crossvalid_meanscore 
        lgbm_vanilla_bestmeanset = paramset
        cv_bestmean_scorevect = crossvalid_results   

print('\n\n ** lightgbm_vanilla **')
print('The best split average accuracy is:',cv_bestsplit)  
print('Corresponding set of parameters for lgbm_vanilla_bestmaxset is:',
        lgbm_vanilla_bestmaxset)
print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
print('The best mean average accuracy is:',cv_bestmean)
print('Corresponding set of parameters for lgbm_vanilla_bestmeanset is:',
        lgbm_vanilla_bestmeanset)
print('Corresponding scores for all splits are:', cv_bestmean_scorevect)



##############################################################
## Optional: save evaluations into a file
##############################################################

if save_evaluations:
    with open(path_save_results, 'w') as file:
        file.write('\n\n ** ridge_vanilla **')
        file.write('The best split average accuracy is:' + str(cv_bestsplit))  
        file.write('Corresponding set of parameters for ridge_vanilla_bestmaxset is:'+ str(
                        ridge_vanilla_bestmaxset))
        file.write('Corresponding scores for all splits are:' + str(cv_bestmax_scorevect))
        file.write('The best mean average accuracy is:' + str(cv_bestmean))
        file.write('Corresponding set of parameters for ridge_vanilla_bestmeanset is:' 
            + str(ridge_vanilla_bestmeanset))
        file.write('Corresponding scores for all splits are:' + str(cv_bestmean_scorevect))
        file.write('\n\n ** lr_vanilla **')
        file.write('The best split average accuracy is:' + str(cv_bestsplit))
        file.write('Corresponding set of parameters for lr_vanilla_bestmaxset is:' + str(lr_vanilla_bestmaxset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmax_scorevect))
        file.write('The best mean average accuracy is:' + str(cv_bestmean))
        file.write('Corresponding set of parameters for lr_vanilla_bestmeanset is:' 
            + str(lr_vanilla_bestmeanset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmean_scorevect))
        file.write('\n\n ** forest_vanilla **')
        file.write('The best split average accuracy is:' + str(cv_bestsplit))  
        file.write('Corresponding set of parameters for forest_vanilla_bestmaxset is:'
           + str(forest_vanilla_bestmaxset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmax_scorevect))
        file.write('The best mean average accuracy is:' + str(cv_bestmean))
        file.write('Corresponding set of parameters for forest_vanilla_bestmeanset is:' 
            + str(forest_vanilla_bestmeanset))
        file.write('Corresponding scores for all splits are:' + str(cv_bestmean_scorevect))
        file.write('\n\n ** xgboost_vanilla **')
        file.write('The best split average accuracy is:' + str(cv_bestsplit))  
        file.write('Corresponding set of parameters for xgboost_vanilla_bestmaxset is:' 
            + str(xgboost_vanilla_bestmaxset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmax_scorevect))
        file.write('The best mean average accuracy is:' + str(cv_bestmean))
        file.write('Corresponding set of parameters for xgboost_vanilla_bestmeanset is:' 
            + str(xgboost_vanilla_bestmeanset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmean_scorevect))
        file.write('\n\n ** lightgbm_vanilla **')
        file.write('The best split average accuracy is:' + str(cv_bestsplit))
        file.write('Corresponding set of parameters for lgbm_vanilla_bestmaxset is:' 
            + str(lgbm_vanilla_bestmaxset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmax_scorevect))
        file.write('The best mean average accuracy is:' + str(cv_bestmean))
        file.write('Corresponding set of parameters for lgbm_vanilla_bestmeanset is:' 
            + str(lgbm_vanilla_bestmeanset))
        file.write('Corresponding scores for all splits are:' + str(cv_bestmean_scorevect))

    print('Evaluations saved at location:', path_save_results)
