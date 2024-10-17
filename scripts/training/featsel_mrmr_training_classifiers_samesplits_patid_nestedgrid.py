#Lucas Sancéré 

import os
import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import random

import math
from tqdm import tqdm
import random
import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter, defaultdict

from src.histo_miner.feature_selection import SelectedFeaturesMatrix, FeatureSelector
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
pathfeatselect = confighm.paths.folders.feature_selection_main
classification_eval_folder = confighm.paths.folders.classification_evaluation
use_permutations = confighm.parameters.bool.permutation

eval_folder_name = confighm.names.eval_folder


with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
run_name = config.names.run_name
nbr_of_splits = config.parameters.int.nbr_of_splits
nbr_of_inner_splits = config.parameters.int.nestedcross_inner_splits

# Could be simplified maybe if only one classifier is kept later 
run_xgboost = config.parameters.bool.run_classifiers.xgboost
run_lgbm = config.parameters.bool.run_classifiers.light_gbm 
# Like following:
# if run_xgboost and not run_lgbm:
# elif run_lgbm and not run_xgboost:
# else: RAISE error

# For the nested corss validation with HP search 
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

ext = '.npy'

featarray_name = 'perwsi_featarray'
classarray_name = 'perwsi_clarray'
pathfeatnames = pathfeatselect + 'featnames' + ext

train_featarray = np.load(pathfeatselect + featarray_name + ext)
train_clarray = np.load(pathfeatselect + classarray_name + ext)
featnames = np.load(pathfeatnames)
featnameslist = list(featnames)


# Load patient ids
path_patientids_array = pathfeatselect + 'patientids' + ext
patientids_load = np.load(path_patientids_array, allow_pickle=True)
patientids_list = list(patientids_load)
patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
patientids = np.asarray(patientids_convert)

#Calculate number of different patients:
unique_elements_set = set(patientids_list)
num_unique_elements = len(unique_elements_set)
print('Number of patient is:', num_unique_elements)



##############################################################
## Load Classifiers
##############################################################


# Define the classifiers
##### XGBOOST
xgboost = xgboost.XGBClassifier(verbosity=0)


#RMQ: Verbosity is set to 0 for XGBOOST to avoid printing WARNINGS (not wanted here for sake of
#simplicity)/ In Light GBM, to avoid showing WARNINGS, the verbosity as to be set to -1.
# See parameters documentation to learn about the other verbosity available. 


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



##############################################################
## Traininig Classifiers to obtain instance prediction score
##############################################################



if run_xgboost and not run_lgbm:
    classifier_name = 'xgboost'
if run_lgbm and not run_xgboost:
    classifier_name = 'lgbm'


# Need print summary (before and after) if the config was not saved before running!
print('Classfier {} used, feature selection {} used.'.format(classifier_name, 'mrmr'))
print('Outer splits: {} , Inner splits: {}.'.format(nbr_of_splits, nbr_of_inner_splits))
print('Folder analysed is {}'.format(pathfeatselect))


print('Start Classifiers trainings...')


train_featarray = np.transpose(train_featarray)

# Initialize a StandardScaler 
# scaler = StandardScaler() 
# scaler.fit(train_featarray) 
# train_featarray = scaler.transform(train_featarray) 

# Create a mapping of unique elements to positive integers
mapping = {}
current_integer = 1
patientids_ordered = []

for num in patientids:
    if num not in mapping:
        mapping[num] = current_integer
        current_integer += 1
    patientids_ordered.append(mapping[num])

patientids_ordered = np.asarray(patientids_ordered)


### Create Stratified Group to further split the dataset into n_splits 
stratgroupkf = StratifiedKFold(n_splits=nbr_of_splits, shuffle=False)


# Create a list of splits with all features 
splits_nested_list = list()
# Create a list of patient IDs corresponding of the splits:
splits_patientID_list = list()
for i, (train_index, test_index) in enumerate(stratgroupkf.split(train_featarray, 
                                                                 train_clarray, 
                                                                 groups=patientids_ordered
                                                                 )):
    # Generate training and test data from the indexes
    X_train = train_featarray[train_index]
    X_test = train_featarray[test_index]
    y_train = train_clarray[train_index]
    y_test = train_clarray[test_index]

    splits_nested_list.append([X_train, y_train, X_test, y_test])

    # Generate the corresponding list for patient ids
    X_train_patID = patientids_ordered[train_index]
    X_test_patID = patientids_ordered[test_index]

    splits_patientID_list.append([X_train_patID, X_test_patID])


# Initialization of parameters
nbr_feat = len(X_train[1])
print('nbr_feat is:',nbr_feat)



##############################################################
##  Classifying with XGBOOST 
##############################################################


if run_xgboost and not run_lgbm:

    balanced_accuracies = {"balanced_accuracies_mrmr": {"initialization": True}}

    length_selfeatmrmr = dict()
    selfeat_mrmr_names_allsplits = []
    selfeat_mrmr_id_allsplits = []

    all_features_balanced_accuracy = list()

    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]

        X_train_patID_split = splits_patientID_list[i][0]
        
        # The array is then transpose to feat FeatureSelector requirements
        X_train_tr = np.transpose(X_train)

        ########### SELECTION OF FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        if i == 0:
            feature_selector = FeatureSelector(X_train_tr, y_train)
        else: 
            feature_selector.reset_attributes(X_train_tr, y_train)

        ## mr.MR calculations
        print('Selection of features with mrmr method...')
        selfeat_mrmr = feature_selector.run_mrmr(nbr_feat)
        selfeat_mrmr_index = selfeat_mrmr[0]
        # Now associate the index of selected features (selfeat_mrmr_index) to the list of names:
        selfeat_mrmr_names = [featnameslist[index] for index in selfeat_mrmr_index] 
        selfeat_mrmr_names_allsplits.append(selfeat_mrmr_names)
        selfeat_mrmr_id_allsplits.append(selfeat_mrmr_index)


        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train

        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION
        balanced_accuracies_mrmr = list()

        ### With mrmr and mannwhitneyu selected features
        for nbr_keptfeat_idx in tqdm(range(nbr_feat, 0, -1)):

            
            # First we train with all features 
            if nbr_keptfeat_idx == nbr_feat:

                # ENTER NESTED GRID SEARCH HERE 
                best_inner_balanced_accuracy = 0

                print('\n\nNested gridsearch of split {} starting...'.format(i))

                for paramset in tqdm(ParameterGrid(xgboost_param_grid)):

                    # set the parameter set choosen in the grid
                    xgboost.set_params(**paramset)

                    # Create Stratified Group to further split the dataset into n_splits 
                    innerstratgroupkf = StratifiedKFold(n_splits=nbr_of_inner_splits, shuffle=False)

                    # create empty lists for initialization
                    inner_splits_nested_list = list()
                    inner_splits_patientID_list = list()

                    inner_balanced_accuracy = list()

                    for k, (inner_train_index, inner_val_index) in enumerate(innerstratgroupkf.split(X_train, 
                                                                             y_train, 
                                                                             groups=X_train_patID_split
                                                                             )):
                        # Generate training and test data from the indexes
                        inner_X_train = X_train[inner_train_index]
                        inner_X_val = X_train[inner_val_index]
                        inner_y_train = train_clarray[inner_train_index]
                        inner_y_val = train_clarray[inner_val_index]

                        inner_splits_nested_list.append([X_train, y_train, X_test, y_test])

                        # Generate the corresponding list for patient ids
                        inner_X_train_patID = patientids_ordered[inner_train_index]
                        inner_X_test_patID = patientids_ordered[inner_val_index]

                        inner_splits_patientID_list.append([inner_X_train_patID, inner_X_test_patID])


                    for l in range(nbr_of_inner_splits):  

                        inner_X_train = splits_nested_list[l][0]
                        inner_y_train = splits_nested_list[l][1]
                        inner_X_val = splits_nested_list[l][2]
                        inner_y_val = splits_nested_list[l][3]

                        #Training
                        xgboost_inner_training_allfeat = xgboost.fit(inner_X_train, inner_y_train)

                        # Predictions on the test split
                        y_inner_pred_allfeat = xgboost_inner_training_allfeat.predict(inner_X_val)

                        # Calculate balanced accuracy for the current split
                        inner_balanced_accuracy_allfeat = balanced_accuracy_score(inner_y_val, 
                                                                                  y_inner_pred_allfeat)

                        # Update mrmr list with the all feature evaluation
                        inner_balanced_accuracy.append(inner_balanced_accuracy_allfeat)


                    inner_balanced_accuracy_npy = np.asarray(inner_balanced_accuracy)
                    mean_inner_balanced_accuracy = np.mean(inner_balanced_accuracy_npy)

                    if mean_inner_balanced_accuracy > best_inner_balanced_accuracy:
                        # we exchange the parameters and store the ba
                        best_inner_balanced_accuracy = mean_inner_balanced_accuracy
                        best_paramset = paramset


                # Update mrmr list with the all feature evaluation
                all_features_balanced_accuracy.append(best_inner_balanced_accuracy)


                print('\nNested gridsearch of split {} finised'.format(i))

                print('Calculate balanced_accuracies for decreasing number of features kept')


            # Then we decrease number of feature kept during training + evaluation

            else:

                # We have to take into account the mrmr is removing 0s from the feature list
                # as it cannot work with 0s
                if nbr_keptfeat_idx <=  len(selfeat_mrmr_index):

                    # Kept the selected features
                    selfeat_mrmr_index_reduced =  selfeat_mrmr_index[0:nbr_keptfeat_idx]
                    selfeat_mrmr_index_reduced = sorted(selfeat_mrmr_index_reduced)

                    # Generate matrix of features
                    featarray_mrmr = feature_array[:, selfeat_mrmr_index_reduced]

                    #Training
                    # needs to be re initialized each time!!!! Very important
                    xgboost_mrmr_training = xgboost

                    # we keep the same found parameters (we are on the same outer split ! just different nbr of feat) 
                    # it is then not fully optimized but at least with no biaises
                    xgboost_mrmr_training.set_params(**best_paramset)

                    # actual training
                    xgboost_mrmr_training_inst = xgboost_mrmr_training.fit(featarray_mrmr, 
                                                                           y_train)

                    # Predictions on the test split
                    y_pred_mrmr = xgboost_mrmr_training_inst.predict(
                        X_test[:, selfeat_mrmr_index_reduced]
                        )
                    # Calculate balanced accuracy for the current split
                    balanced_accuracy_mrmr = balanced_accuracy_score(y_test, 
                                                                     y_pred_mrmr)
                    balanced_accuracies_mrmr.append(balanced_accuracy_mrmr)


        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies_mrmr

        # update changing size of kept feature for mrmr
        length_selfeatmrmr[currentsplit] = len(selfeat_mrmr_index)



##############################################################
##  Classifying with LGBM 
##############################################################


elif run_lgbm and not run_xgboost:

    balanced_accuracies = {"balanced_accuracies_mrmr": {"initialization": True}}

    length_selfeatmrmr = dict()
    selfeat_mrmr_names_allsplits = []
    selfeat_mrmr_id_allsplits = []

    all_features_balanced_accuracy = list()

    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]

        X_train_patID_split = splits_patientID_list[i][0]
        
        # The array is then transpose to feat FeatureSelector requirements
        X_train_tr = np.transpose(X_train)

        ########### SELECTION OF FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        if i == 0:
            feature_selector = FeatureSelector(X_train_tr, y_train)
        else: 
            feature_selector.reset_attributes(X_train_tr, y_train)

        ## mr.MR calculations
        print('Selection of features with mrmr method...')
        selfeat_mrmr = feature_selector.run_mrmr(nbr_feat)
        selfeat_mrmr_index = selfeat_mrmr[0]
        # Now associate the index of selected features (selfeat_mrmr_index) to the list of names:
        selfeat_mrmr_names = [featnameslist[index] for index in selfeat_mrmr_index] 
        selfeat_mrmr_names_allsplits.append(selfeat_mrmr_names)
        selfeat_mrmr_id_allsplits.append(selfeat_mrmr_index)
        

        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train


        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION
        balanced_accuracies_mrmr = list()

        ### With mrmr and mannwhitneyu selected features
        for nbr_keptfeat_idx in tqdm(range(nbr_feat, 0, -1)):

            # ENTER NESTED GRID SEARCH HERE 
            best_inner_balanced_accuracy = 0

            # First we train with all features 
            if nbr_keptfeat_idx == nbr_feat:

                # ENTER NESTED GRID SEARCH HERE 
                best_inner_balanced_accuracy = 0

                print('\n\nNested gridsearch of split {} starting...'.format(i))

                for paramset in tqdm(ParameterGrid(lgbm_param_grid)):

                    # set the parameter set choosen in the grid
                    # lightgbm.set_params(**paramset)

                    # Create Stratified Group to further split the dataset into n_splits 
                    innerstratgroupkf = StratifiedKFold(n_splits=nbr_of_inner_splits, shuffle=False)

                    # create empty lists for initialization
                    inner_splits_nested_list = list()
                    inner_splits_patientID_list = list()

                    inner_balanced_accuracy = list()

                    for k, (inner_train_index, inner_val_index) in enumerate(innerstratgroupkf.split(X_train, 
                                                                             y_train, 
                                                                             groups=X_train_patID_split
                                                                             )):
                       
                        # Generate training and test data from the indexes
                        inner_X_train = X_train[inner_train_index]
                        inner_X_val = X_train[inner_val_index]
                        inner_y_train = train_clarray[inner_train_index]
                        inner_y_val = train_clarray[inner_val_index]

                        inner_splits_nested_list.append([X_train, y_train, X_test, y_test])

                        # Generate the corresponding list for patient ids
                        inner_X_train_patID = patientids_ordered[inner_train_index]
                        inner_X_test_patID = patientids_ordered[inner_val_index]

                        inner_splits_patientID_list.append([inner_X_train_patID, inner_X_test_patID])

                    for l in range(nbr_of_inner_splits):  

                        inner_X_train = splits_nested_list[l][0]
                        inner_y_train = splits_nested_list[l][1]
                        inner_X_val = splits_nested_list[l][2]
                        inner_y_val = splits_nested_list[l][3]

                        #Training
                        train_data = lightgbm.Dataset(inner_X_train, label=inner_y_train)
                        lightgbm_training_allfeat = lightgbm.train(
                            paramset,
                            train_data
                        )

                        # Predictions on the test split
                        y_inner_pred_allfeat_prob = lightgbm_training_allfeat.predict(
                                                     inner_X_val,
                                                     num_iteration=lightgbm_training_allfeat.best_iteration)
                        y_inner_pred_allfeat = (y_inner_pred_allfeat_prob > 0.5).astype(int)

                        # Calculate balanced accuracy for the current split
                        inner_balanced_accuracy_allfeat = balanced_accuracy_score(inner_y_val, 
                                                                                  y_inner_pred_allfeat)

                         # Update mrmr list with the all feature evaluation
                        inner_balanced_accuracy.append(inner_balanced_accuracy_allfeat)


                    inner_balanced_accuracy_npy = np.asarray(inner_balanced_accuracy)
                    mean_inner_balanced_accuracy = np.mean(inner_balanced_accuracy_npy)

                    if mean_inner_balanced_accuracy > best_inner_balanced_accuracy:
                        # we exchange the parameters and store the ba
                        best_inner_balanced_accuracy = mean_inner_balanced_accuracy
                        best_paramset = paramset


                # Update mrmr list with the all feature evaluation
                all_features_balanced_accuracy.append(best_inner_balanced_accuracy)


                print('\nNested gridsearch of split {} finised'.format(i))

                print('Calculate balanced_accuracies for decreasing number of features kept')


            # Then we decrease number of feature kept during training + evaluation
            else:

                # We have to take into account the mrmr is removing 0s from the feature list
                # as it cannot work with 0s
                if nbr_keptfeat_idx <= len(selfeat_mrmr_index):

                    # Kept the selected features
                    selfeat_mrmr_index_reduced =  selfeat_mrmr_index[0:nbr_keptfeat_idx]
                    selfeat_mrmr_index_reduced = sorted(selfeat_mrmr_index_reduced)

                    # Generate matrix of features
                    featarray_mrmr = feature_array[:, selfeat_mrmr_index_reduced]

                    #Training
                    train_data = lightgbm.Dataset(featarray_mrmr, label=y_train)
                    lightgbm_mrmr_training_inst = lightgbm.train(
                        best_paramset,
                        train_data,
                        )

                    # Predictions on the test split
                    y_pred_mrmr_prob = lightgbm_mrmr_training_inst.predict(
                        X_test[:, selfeat_mrmr_index_reduced],
                        )
                    y_pred_mrmr  = (y_pred_mrmr_prob > 0.5).astype(int)

                    # Calculate balanced accuracy for the current split
                    balanced_accuracy_mrmr = balanced_accuracy_score(y_test, 
                                                                     y_pred_mrmr)
                    balanced_accuracies_mrmr.append(balanced_accuracy_mrmr)


        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies_mrmr

        # update changing size of kept feature for mrmr
        length_selfeatmrmr[currentsplit] = len(selfeat_mrmr_index)


else:
    raise ValueError('run_xgboost and run_lgbm cannot be both True or both False for'
                  'the script to run')



####################################################################
## Extract mean,min,max of balanced accuracy and kept feature names 
####################################################################


## calculate and write the saving of the mean balanced accuracies
## Do for mrmr
# Calculate the mean accuracies 

mean_balanced_accuracies_mrmr = list()
min_balanced_accuracies_mrmr = list()
max_balanced_accuracies_mrmr = list()
std_balanced_accuracies_mrmr = list()

best_mean_mrmr = 0


# we want to delete the first elements of the lists when there is not the same number of features kept by mrmr.
# but we keep the first element as it is with all features

listoflengths = list()

for i in range(nbr_of_splits):

    currentsplit =  f"split_{i}"
    listoflengths.append(length_selfeatmrmr[currentsplit])

nbrkept_max_allsplits = min(listoflengths)
print(nbrkept_max_allsplits)

# remove firsts elements to be comparable in termes of number of feat kept
for i in range(nbr_of_splits):

    currentsplit =  f"split_{i}"
    # We use sclicing to keep the nbr_feat-nbrkept_max_allsplits last elements of the list 
    balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies['balanced_accuracies_mrmr'][currentsplit][-nbrkept_max_allsplits:]

    selfeat_mrmr_names_allsplits[i] = selfeat_mrmr_names_allsplits[i][-nbrkept_max_allsplits:]


for index in range(0, nbrkept_max_allsplits):
    
    ba_featsel_mrmr = list()

    for i in range(nbr_of_splits):

        currentsplit =  f"split_{i}"

        balanced_accuracy_mrmr = np.asarray(
            balanced_accuracies['balanced_accuracies_mrmr'][currentsplit][index]
            ) 

        ba_featsel_mrmr.append(balanced_accuracy_mrmr)


    ba_featsel_mrmr = np.asarray(ba_featsel_mrmr)

    mean_balanced_accuracies_mrmr.append(np.mean(ba_featsel_mrmr))
    min_balanced_accuracies_mrmr.append(np.min(ba_featsel_mrmr))
    max_balanced_accuracies_mrmr.append(np.max(ba_featsel_mrmr))
    std_balanced_accuracies_mrmr.append(np.std(ba_featsel_mrmr))

    ###### Find name of selected features that leads to the best prediction
    if np.mean(ba_featsel_mrmr) > best_mean_mrmr:
        nbr_kept_features_mrmr = nbrkept_max_allsplits - index
        kept_features_mrmr = [split[0: nbr_kept_features_mrmr] for split in selfeat_mrmr_names_allsplits]
        best_mean_mrmr = np.mean(ba_featsel_mrmr)  


mean_ba_mrmr_npy = np.asarray(mean_balanced_accuracies_mrmr)
min_ba_mrmr_npy = np.asarray(min_balanced_accuracies_mrmr)
max_ba_mrmr_npy = np.asarray(max_balanced_accuracies_mrmr)
std_ba_mrmr_npy = np.asarray(std_balanced_accuracies_mrmr)


# for all features

all_features_balanced_accuracy_npy = np.asarray(all_features_balanced_accuracy)
mean_ba_allfeat = np.mean(all_features_balanced_accuracy_npy)



####################################################################
## Select best features after cross-validation
####################################################################

# We will just keep the features that are most kept on the cross
# If there is a tie, we calculate their scores, if there are most 1s 
# than we keep them


# We set the number of features kept for each splits again
nbr_kept_feat = nbr_kept_features_mrmr


# Create a score for feature selection
prescores = [10**(index) for index in range(nbr_kept_feat,0,-1)]

featscores = prescores * nbr_of_splits


##### Best slected features by MRMR 

# We start by counting number of occurance in top 5
bestsel_mrmr_index = [
    featindex 
    for splitindex in range(0, nbr_of_splits)
    for featindex in selfeat_mrmr_id_allsplits[splitindex][0:nbr_kept_feat]   
    ]

# Count occurrences of each featindex
featindex_counts_mrmr = Counter(bestsel_mrmr_index)

# Calculate score of each featindex (in case of draw we advantage feat arriving first more often)
score_sums = defaultdict(int)
for idx, featindex in enumerate(bestsel_mrmr_index):
    score_sums[featindex] += featscores[idx]

# Take log10 for all values
score_sums = {key: math.log10(value) for key, value in score_sums.items()}

# Sort featindex by occurrence count and then by  scores
sorted_bestfeatindex_mrmr = sorted(
        featindex_counts_mrmr.keys(), 
        key=lambda x: (-featindex_counts_mrmr[x], -score_sums[x])
        )

# Retrieve names of best selected features
mrmr_finalselfeat_names = list(featnames[sorted_bestfeatindex_mrmr[0:len(bestsel_mrmr_index)]])


# create a list with  featindex count and score 
# Create a dictionary to store the best features with their count and score
best_features_info = {
    featnames[featindex]: {
        'count': featindex_counts_mrmr[featindex],
        'score': score_sums[featindex]
    }
    for featindex in sorted_bestfeatindex_mrmr[0:len(bestsel_mrmr_index)]
}







##############################################################
## Save numpy files
##############################################################

# save the mean balanced accuracies for visualization
save_results_path = classification_eval_folder + eval_folder_name + '/'
save_ext = '.npy' 

if not os.path.exists(classification_eval_folder):
    os.mkdir(classification_eval_folder)

if not os.path.exists(save_results_path):
    os.mkdir(save_results_path)


# Need print summary (before and after) if the config was not saved before running!
print('Classfier {} used, feature selection {} used.'.format(classifier_name, 'mrmr'))
print('Outer splits: {} , Inner splits: {}.'.format(nbr_of_splits, nbr_of_inner_splits))
print('Start saving numpy in folder: ', save_results_path)


name_mrmr_output = '_ba_mrmr_' + str(nbr_of_splits) + 'splits_' + run_name
np.save(save_results_path + 'mean_' + classifier_name + name_mrmr_output + save_ext, 
    mean_ba_mrmr_npy)
np.save(save_results_path + 'max_' + classifier_name + name_mrmr_output + save_ext, 
    min_ba_mrmr_npy)
np.save(save_results_path + 'min_' + classifier_name + name_mrmr_output + save_ext, 
    max_ba_mrmr_npy)
np.save(save_results_path + 'std_' + classifier_name + name_mrmr_output + save_ext, 
    std_ba_mrmr_npy)
np.save(save_results_path + 'topselfeatid_' + classifier_name  + name_mrmr_output + save_ext, 
    sorted_bestfeatindex_mrmr)


print('Numpy saved.')



##############################################################
## Save text file
##############################################################


txtfilename = (
    classifier_name +  '_' +
    'mrmr' +  '_' +
    str(nbr_of_splits) + 'splits_' +
    str(nbr_of_inner_splits) + 'innersplits_' +
    run_name + '_info'
)

save_txt_ext = '.txt'
save_text_folder = save_results_path + '/infofiles/' 

if not os.path.exists(save_text_folder):
    os.mkdir(save_text_folder)

save_text_path = save_text_folder + txtfilename + save_txt_ext


## ADD NAME OF CLASSIFER THAT WAS RUNNING

print('Start saving name and number of feature kept in best case')

with open(save_text_path, 'w') as file:
    file.write('** With {} classifier **'.format(classifier_name))

    file.write('\n\n\n\n ** mrmr **')
    file.write('\n\nBest mean balanced accuracy is:' +  
        str(max(best_mean_mrmr, mean_ba_allfeat))) 
    file.write('\n\nAll feat mean balanced accuracy is:' +  
        str(mean_ba_allfeat))  
    file.write('\n\nThe number of kept features in the best scenario is:' + 
        str(nbr_kept_features_mrmr))  
    # file.write('\n\nThese features are:' +
    #     str(kept_features_mrmr)) 
    file.write('\n\nThe best features overall are:' +  
        str([best_features_info]))
    #file.write('\n\nThe best 5 features are:' +  
    # str([kept_features[0:4] for kept_features in kept_features_mrmr])) 


print('Text file name:',txtfilename )
print('Text file saved.')


#### Save all splits balanced accuracies values 

#### Save roc curve information 



