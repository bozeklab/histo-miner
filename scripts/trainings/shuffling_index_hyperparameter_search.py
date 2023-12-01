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

xgboost_param_grid_random_state = list(config.classifierparam.xgboost.grid_dict.random_state)
xgboost_param_grid_n_estimators = list(config.classifierparam.xgboost.grid_dict.n_estimators)
xgboost_param_grid_learning_rate = list(config.classifierparam.xgboost.grid_dict.learning_rate)
xgboost_param_grid_objective = list(config.classifierparam.xgboost.grid_dict.objective)

lgbm_param_grid_random_state = list(config.classifierparam.light_gbm.grid_dict.random_state)
lgbm_param_grid_n_estimators = list(config.classifierparam.light_gbm.grid_dict.n_estimators)
lgbm_param_grid_learning_rate = list(config.classifierparam.light_gbm.grid_dict.learning_rate)
lgbm_param_grid_objective = list(config.classifierparam.light_gbm.grid_dict.objective)
lgbm_param_grid_num_leaves = list(config.classifierparam.light_gbm.grid_dict.num_leaves)




################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################


pathfeatselect = pathtofolder + '/feature_selection/'
ext = '.npy'


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
##### XGBOOST
xgboost = xgboost.XGBClassifier(verbosity=0)

##### LIGHT GBM setting
# The use of light GBM classifier is not following the convention of the other one
# Here we will save parameters needed for training, but there are no .fit method
lightgbm = lightgbm.LGBMClassifier(verbosity=-1) 


###### Load all paramters into a dictionnary for Grid Search
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


### Create a new permutation and save it
# permutation_index = np.random.permutation(train_clarray.size)
# np.save(pathfeatselect + 'random_permutation_index_new2.npy', permutation_index)

### Create Stratified Group  instance for the cross validation 
stratgroupkf = StratifiedGroupKFold(n_splits=10, shuffle=False)

#Initialisation of index permutation parrameters 

xgboost_cv_bestmean_perm = 0
xgboost_bestmean_permutation_index = []
xgboost_cv_bestsplit_perm = 0
xgboost_bestsplit_permutation_index = []

lgbm_cv_bestmean_perm = 0
lgbm_bestmean_permutation_index = []
lgbm_cv_bestsplit_perm = 0
lgbm_bestsplit_permutation_index = []


for index in tqdm(range(0,200)):

    #We reload everything at every round because we don't know how it affects
    #Variable values
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
    genfeatarray = np.transpose(train_featarray) 
    genfeatarray = genfeatarray[permutation_index,:]

        ### Shuffle patient IDs arrays using the permutation index 
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


    ##### XGBOOST
    xgboostvanilla = xgboost
    # xgboost_vanilla = xgboostvanilla.fit(genfeatarray, train_clarray)
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
    
    # Keep the best permutation index list for xgboost 
    if cv_bestmean > xgboost_cv_bestmean_perm:
        xgboost_cv_bestmean_perm = cv_bestmean
        xgboost_bestmean_permutation_index = permutation_index
    if cv_bestsplit > xgboost_cv_bestsplit_perm:
        xgboost_cv_bestsplit_perm = cv_bestsplit
        xgboost_bestsplit_permutation_index = permutation_index

    print('\n\n ** xgboost_vanilla **')
    print('The best split average accuracy is:',cv_bestsplit)  
    print('Corresponding set of parameters for xgboost_vanilla_bestmaxset is:',
            xgboost_vanilla_bestmaxset)
    print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
    print('The best mean average accuracy is:',cv_bestmean)
    print('Corresponding set of parameters for xgboost_vanilla_bestmeanset is:',
            xgboost_vanilla_bestmeanset)
    print('Corresponding scores for all splits are:', cv_bestmean_scorevect)
    print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


    #### LIGHT GBM
    # lgbm_traindata_vanilla = lightgbm.Dataset(genfeatarray, label=train_clarray) 
    # #lgbm_valdata_vanilla = lgbm_traindata_vanilla.create_valid()
    # lgbm_vanilla = lightgbm.train(lightgbm_paramters, 
    #                               lgbm_traindata_vanilla, 
    #                               lgbm_n_estimators)
    lightgbmvanilla = lightgbm
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

    # Keep the best permutation index list for xgboost 
    if cv_bestmean > xgboost_cv_bestmean_perm:
        lgbm_cv_bestmean_perm = cv_bestmean
        lgbm_bestmean_permutation_index = permutation_index
    if cv_bestsplit > xgboost_cv_bestsplit_perm:
        lgbm_cv_bestsplit_perm = cv_bestsplit
        lgbm_bestsplit_permutation_index = permutation_index

    print('\n\n ** lightgbm_vanilla **')
    print('The best split average accuracy is:',cv_bestsplit)  
    print('Corresponding set of parameters for lgbm_vanilla_bestmaxset is:',
            lgbm_vanilla_bestmaxset)
    print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
    print('The best mean average accuracy is:',cv_bestmean)
    print('Corresponding set of parameters for lgbm_vanilla_bestmeanset is:',
            lgbm_vanilla_bestmeanset)
    print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


np.save(pathfeatselect + 'random_permutation_index_11_28_xgboost_bestmean.npy', xgboost_bestmean_permutation_index)
np.save(pathfeatselect + 'random_permutation_index_11_28_xgboost_bestsplit.npy', xgboost_bestsplit_permutation_index)

np.save(pathfeatselect + 'random_permutation_index_11_28_lgbm_bestmean.npy', lgbm_bestmean_permutation_index)
np.save(pathfeatselect + 'random_permutation_index_11_28_lgbm_bestsplit.npy',lgbm_bestsplit_permutation_index)


print('Shuffling parameter search done.')