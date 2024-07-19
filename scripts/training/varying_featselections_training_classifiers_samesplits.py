#Lucas Sancéré 

import os
import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import random

from tqdm import tqdm
import random
import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

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
boruta_max_depth = confighm.parameters.int.boruta_max_depth
boruta_random_state = confighm.parameters.int.boruta_random_state



with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures

xgboost_random_state = config.classifierparam.xgboost.random_state
xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
xgboost_lr = config.classifierparam.xgboost.learning_rate
xgboost_objective = config.classifierparam.xgboost.objective
lgbm_random_state = config.classifierparam.light_gbm.random_state
lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
lgbm_lr = config.classifierparam.light_gbm.learning_rate
lgbm_objective = config.classifierparam.light_gbm.objective
lgbm_numleaves = config.classifierparam.light_gbm.num_leaves

# Could be simplified maybe if only one classifier is kept later 
run_xgboost = config.parameters.bool.run_classifiers.xgboost
run_lgbm = config.parameters.bool.run_classifiers.light_gbm 
# Like following:
# if run_xgboost and not run_lgbm:
# elif run_lgbm and not run_xgboost:
# else: RAISE error

              

############################################################
## Load feature selection numpy files
############################################################

# ext = '.npy'

# print('Load feature selection numpy files...')

# # Load feature selection numpy files
# pathselfeat_boruta = pathfeatselect + '/selfeat_boruta_idx_depth18' + ext
# selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
# print('Loading feature selected indexes done.')



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


##############################################################
## Traininig Classifiers to obtain instance prediction score
##############################################################


print('Start Classifiers trainings...')


### Create a new permutation and save it
# permutation_index = np.random.permutation(train_clarray.size)
# np.save(pathfeatselect + 'random_permutation_index_new2.npy', permutation_index)
### Load permutation index not to have 0 and 1s not mixed
# permutation_index = np.load(pathfeatselect + 
#                             '/bestperm/' +
#                             'random_permutation_index_11_28_xgboost_bestmean.npy') 
# nbr_slides = len(train_clarray)

# permutation_index = list(range(nbr_slides))
# random.shuffle(permutation_index)
# permutation_index = np.asarray(permutation_index)



# nbrindeces = len(permutation_index)

### Shuffle classification arrays using the permutation index
# train_clarray = train_clarray[permutation_index]


# Shuffle the all features arrays using the permutation index

train_featarray = np.transpose(train_featarray)
# train_featarray = train_featarray[permutation_index, :]


# Create a mapping of unique elements to positive integers
mapping = {}
current_integer = 1
patientids_ordered = []

for num in patientids:
    if num not in mapping:
        mapping[num] = current_integer
        current_integer += 1
    patientids_ordered.append(mapping[num])

### Shuffle patient IDs arrays using the permutation index 
patientids_ordered = np.asarray(patientids_ordered)
# patientids_ordered = patientids_ordered[permutation_index]


# number of splits for cross validation
nbr_of_splits = 10


### Create Stratified Group to further split the dataset into n_splits 
stratgroupkf = StratifiedGroupKFold(n_splits=nbr_of_splits, shuffle=False)


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


# -- XGBOOST --

if run_xgboost and not run_lgbm:


    balanced_accuracies = {"balanced_accuracies_mannwhitneyu": {"initialization": True},
                           "balanced_accuracies_mrmr": {"initialization": True},
                           "balanced_accuracies_boruta": {"initialization": True}}

    # NOT SURE TO KEEP IT THOUGH -------
    # selected_features = {"best_features_mannwhitneyu": {"initialization": True},
    #                      "best_features_mrmr": {"initialization": True},
    #                      "best_features_boruta": {"initialization": True}}
    # NOT SURE TO KEEP IT THOUGH -------
    
    selfeat_mannwhitneyu_names_allsplits = [] 
    selfeat_mrmr_names_allsplits = []
    selfeat_boruta_allsplits = []
    number_feat_kept_boruta = []

    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        # The array is then transpose to feat FeatureSelector requirements
        X_train_tr = np.transpose(X_train)

        ########### SELECTION OF FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        if i == 0:
            feature_selector = FeatureSelector(X_train_tr, y_train)
        else: 
            feature_selector.reset_attributes(X_train_tr, y_train)
        ## Mann Whitney U calculations
        print('Selection of features with mannwhitneyu method...')
        selfeat_mannwhitneyu_index, orderedp_mannwhitneyu = feature_selector.run_mannwhitney(nbr_feat)
        # Now associate the index of selected features (selfeat_mannwhitneyu_index) to the list of names:
        selfeat_mannwhitneyu_names = [featnameslist[index] for index in selfeat_mannwhitneyu_index]
        selfeat_mannwhitneyu_names_allsplits.append(selfeat_mannwhitneyu_names)

        ## mr.MR calculations
        print('Selection of features with mrmr method...')
        selfeat_mrmr = feature_selector.run_mrmr(nbr_feat)
        selfeat_mrmr_index = selfeat_mrmr[0]
        # Now associate the index of selected features (selfeat_mrmr_index) to the list of names:
        selfeat_mrmr_names = [featnameslist[index] for index in selfeat_mrmr_index] 
        selfeat_mrmr_names_allsplits.append(selfeat_mrmr_names)

        # Boruta calculations (for one specific depth)
        print('Selection of features with Boruta method...')
        selfeat_boruta_index = feature_selector.run_boruta(max_depth=boruta_max_depth, 
                                                           random_state=boruta_random_state)
        nbrfeatsel_boruta = len(selfeat_boruta_index)
        number_feat_kept_boruta.append(nbrfeatsel_boruta)
        # Now associate the index of selected features (selfeat_boruta_index) to the list of names:
        selfeat_boruta_names = [featnameslist[index] for index in selfeat_boruta_index]
        selfeat_boruta_allsplits.append(selfeat_boruta_names)


        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train
        ## For Boruta calculations
        featarray_boruta = feature_array[:, np.transpose(selfeat_boruta_index)]

        ##  Mann Whitney U calculations & mr.MR calculations are done later 


        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION
        balanced_accuracies_mrmr = list()
        balanced_accuracies_mannwhitneyu = list()


        print('Calculate balanced_accuracies for decreasing number of features kept')
        ### With mrmr and mannwhitneyu selected features
        for nbr_keptfeat_idx in tqdm(range(nbr_feat, 0, -1)):

            # First we train with all features 
            if nbr_keptfeat_idx == nbr_feat:
                #Training
                xgboost_training_allfeat = xgboost.fit(X_train, y_train)

                # Predictions on the test split
                y_pred_allfeat = xgboost_training_allfeat.predict(X_test)

                # Calculate balanced accuracy for the current split
                balanced_accuracy_allfeat = balanced_accuracy_score(y_test, 
                                                                    y_pred_allfeat)

                # Update mrmr and mannwhitney list with the all feature evaluation
                balanced_accuracies_mrmr.append(balanced_accuracy_allfeat)
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_allfeat)


            # Then we decrease number of feature kept during training + evaluation
            else:
                # Kept the selected features
                selfeat_mrmr_index_reduced =  selfeat_mrmr_index[0:nbr_keptfeat_idx]
                selfeat_mannwhitneyu_index_reduced = selfeat_mannwhitneyu_index[0:nbr_keptfeat_idx]
                selfeat_mrmr_index_reduced = sorted(selfeat_mrmr_index_reduced)
                selfeat_mannwhitneyu_index_reduced = sorted(selfeat_mannwhitneyu_index_reduced)

                # Generate matrix of features
                featarray_mrmr = feature_array[:, selfeat_mrmr_index_reduced]
                featarray_mannwhitneyu = feature_array[:, selfeat_mannwhitneyu_index_reduced]

                #Training
                # needs to be re initilaized each time!!!! Vry important
                xgboost_mrmr_training = xgboost
                xgboost_mannwhitneyu_training = xgboost 
                # actual training
                xgboost_mrmr_training_inst = xgboost_mrmr_training.fit(featarray_mrmr, 
                                                                       y_train)
                xgboost_mannwhitneyu_training_inst = xgboost_mannwhitneyu_training.fit(
                                                                       featarray_mannwhitneyu, 
                                                                       y_train
                                                                       )

                # Predictions on the test split
                y_pred_mrmr = xgboost_mrmr_training_inst.predict(
                    X_test[:, selfeat_mrmr_index_reduced]
                    )
                y_pred_mannwhitneyu = xgboost_mannwhitneyu_training_inst.predict(
                    X_test[:, selfeat_mannwhitneyu_index_reduced]
                    )


                # Calculate balanced accuracy for the current split
                balanced_accuracy_mrmr = balanced_accuracy_score(y_test, 
                                                                 y_pred_mrmr)
                balanced_accuracy_mannwhitneyu = balanced_accuracy_score(y_test, 
                                                                         y_pred_mannwhitneyu)

                balanced_accuracies_mrmr.append(balanced_accuracy_mrmr)
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_mannwhitneyu)


        ### With boruta selected features
        #Training
        # sometimes boruta is not keeping any features, so need to check if there are some
        if np.size(featarray_boruta) == 0:      
            balanced_accuracy_boruta = None

        else:
            xgboost_boruta_training = xgboost
            xgboost_boruta_training = xgboost_boruta_training.fit(featarray_boruta, 
                                                                  y_train)
            # Predictions on the test split
            y_pred_boruta = xgboost_boruta_training.predict(
                X_test[:, np.transpose(selfeat_boruta_index)]
                )

            # Calculate balanced accuracy for the current split
            balanced_accuracy_boruta = balanced_accuracy_score(y_test,
                                                               y_pred_boruta)



        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_mannwhitneyu'][currentsplit] = balanced_accuracies_mannwhitneyu
        balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies_mrmr
        balanced_accuracies['balanced_accuracies_boruta'][currentsplit] = balanced_accuracy_boruta



# -- LGBM --

elif run_lgbm and not run_xgboost:

    balanced_accuracies = {"balanced_accuracies_mannwhitneyu": {"initialization": True},
                           "balanced_accuracies_mrmr": {"initialization": True},
                           "balanced_accuracies_boruta": {"initialization": True}}
    
    selfeat_mannwhitneyu_names_allsplits = [] 
    selfeat_mrmr_names_allsplits = []
    selfeat_boruta_allsplits = []
    number_feat_kept_boruta = []
    
    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        # The array is then transpose to feat FeatureSelector requirements
        X_train_tr = np.transpose(X_train)

        ########### SELECTION OF FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        if i == 0:
            feature_selector = FeatureSelector(X_train_tr, y_train)
        else: 
            feature_selector.reset_attributes(X_train_tr, y_train)
        ## Mann Whitney U calculations
        print('Selection of features with mannwhitneyu method...')
        selfeat_mannwhitneyu_index, orderedp_mannwhitneyu = feature_selector.run_mannwhitney(nbr_feat)
        # Now associate the index of selected features (selfeat_mannwhitneyu_index) to the list of names:
        selfeat_mannwhitneyu_names = [featnameslist[index] for index in selfeat_mannwhitneyu_index]
        selfeat_mannwhitneyu_names_allsplits.append(selfeat_mannwhitneyu_names)

        ## mr.MR calculations
        print('Selection of features with mrmr method...')
        selfeat_mrmr = feature_selector.run_mrmr(nbr_feat)
        selfeat_mrmr_index = selfeat_mrmr[0]
        # Now associate the index of selected features (selfeat_mrmr_index) to the list of names:
        selfeat_mrmr_names = [featnameslist[index] for index in selfeat_mrmr_index] 
        selfeat_mrmr_names_allsplits.append(selfeat_mrmr_names)

        ## Boruta calculations (for one specific depth)
        print('Selection of features with Boruta method...')
        selfeat_boruta_index = feature_selector.run_boruta(max_depth=boruta_max_depth, 
                                                           random_state=boruta_random_state)
        nbrfeatsel_boruta = len(selfeat_boruta_index)
        number_feat_kept_boruta.append(nbrfeatsel_boruta)
        # Now associate the index of selected features (selfeat_boruta_index) to the list of names:
        selfeat_boruta_names = [featnameslist[index] for index in selfeat_boruta_index]
        selfeat_boruta_allsplits.append(selfeat_boruta_names)

        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train
        ## For Boruta calculations
        featarray_boruta = feature_array[:, np.transpose(selfeat_boruta_index)]
  
        ##  Mann Whitney U calculations & mr.MR calculations are done later 


        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION
        balanced_accuracies_mrmr = list()
        balanced_accuracies_mannwhitneyu = list()


        print('Calculate balanced_accuracies for decreasing number of features kept')
        ### With mrmr and mannwhitneyu selected features
        for nbr_keptfeat_idx in tqdm(range(nbr_feat, 0, -1)):

            # First we train with all features 
            if nbr_keptfeat_idx == nbr_feat:
                #Training
                lightgbm_training_allfeat = lightgbm.fit(X_train, y_train)

                # Predictions on the test split
                y_pred_allfeat = lightgbm_training_allfeat.predict(X_test)

                # Calculate balanced accuracy for the current split
                balanced_accuracy_allfeat = balanced_accuracy_score(y_test, 
                                                                    y_pred_allfeat)

                # Update mrmr and mannwhitney list with the all feature evaluation
                balanced_accuracies_mrmr.append(balanced_accuracy_allfeat)
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_allfeat)


            # Then we decrease number of feature kept during training + evaluation
            else:
                # Kept the selected features
                selfeat_mrmr_index_reduced =  selfeat_mrmr_index[0:nbr_keptfeat_idx]
                selfeat_mannwhitneyu_index_reduced = selfeat_mannwhitneyu_index[0:nbr_keptfeat_idx]
                selfeat_mrmr_index_reduced = sorted(selfeat_mrmr_index_reduced)
                selfeat_mannwhitneyu_index_reduced = sorted(selfeat_mannwhitneyu_index_reduced)


                # Generate matrix of features
                featarray_mrmr = feature_array[:, selfeat_mrmr_index_reduced]
                featarray_mannwhitneyu = feature_array[:, selfeat_mannwhitneyu_index_reduced]


                #Training
                # needs to be re initialized each time!!!! Very important
                lightgbm_mrmr_training = lightgbm
                lightgbm_mannwhitneyu_training = lightgbm
                # actual training
                lightgbm_mrmr_training_inst = lightgbm_mrmr_training.fit(featarray_mrmr, 
                                                                         y_train)
                lightgbm_mannwhitneyu_training_inst = lightgbm_mannwhitneyu_training.fit(
                                                                       featarray_mannwhitneyu, 
                                                                       y_train
                                                                       )

                # Predictions on the test split
                y_pred_mrmr = lightgbm_mrmr_training_inst.predict(
                    X_test[:, selfeat_mrmr_index_reduced]
                    )
                y_pred_mannwhitneyu = lightgbm_mannwhitneyu_training_inst.predict(
                    X_test[:, selfeat_mannwhitneyu_index_reduced]
                    )

                # Calculate balanced accuracy for the current split
                balanced_accuracy_mrmr = balanced_accuracy_score(y_test, 
                                                                 y_pred_mrmr)
                balanced_accuracy_mannwhitneyu = balanced_accuracy_score(y_test, 
                                                                         y_pred_mannwhitneyu)

                balanced_accuracies_mrmr.append(balanced_accuracy_mrmr)
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_mannwhitneyu)


        ### With boruta selected features
        #Training
        # sometimes boruta is not keeping any features, so need to check if there are some
        if np.size(featarray_boruta) == 0:      
            balanced_accuracy_boruta = None
            # balanced_accuracy_boruta_2 = None
            # balanced_accuracy_boruta_3 = None
        else:
            lightgbm_boruta_training = lightgbm
            lightgbm_boruta_training = lightgbm_boruta_training.fit(featarray_boruta, 
                                                                  y_train)
            # Predictions on the test split
            y_pred_boruta = lightgbm_boruta_training.predict(
                X_test[:, np.transpose(selfeat_boruta_index)]
                )

            # Calculate balanced accuracy for the current split
            balanced_accuracy_boruta = balanced_accuracy_score(y_test,
                                                               y_pred_boruta)

        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_mannwhitneyu'][currentsplit] = balanced_accuracies_mannwhitneyu
        balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies_mrmr
        balanced_accuracies['balanced_accuracies_boruta'][currentsplit] = balanced_accuracy_boruta

else:
    raise ValueError('run_xgboost and run_lgbm cannot be both True or both False for'
                      'the script to run')



####################################################################
## Extract mean,min,max of balanced accuracy and kept feature names 
####################################################################


### calculate and write the saving of the mean balanced accuracies
# Calculate the mean accuracies 

mean_balanced_accuracies_mannwhitneyu = list()
min_balanced_accuracies_mannwhitneyu = list()
max_balanced_accuracies_mannwhitneyu = list()
std_balanced_accuracies_mannwhitneyu = list()

mean_balanced_accuracies_mrmr = list()
min_balanced_accuracies_mrmr = list()
max_balanced_accuracies_mrmr = list()
std_balanced_accuracies_mrmr = list()

best_mean_mannwhitneyu = 0
best_mean_mrmr = 0

# do the list of means for mannwhitneyu and mrmr
for index in range(0, nbr_feat):
    
    ba_featsel_mannwhitneyu = list()
    ba_featsel_mrmr = list()

    for i in range(nbr_of_splits):

        currentsplit =  f"split_{i}"

        balanced_accuracy_mannwhitneyu = float(
            balanced_accuracies['balanced_accuracies_mannwhitneyu'][currentsplit][index]
            ) 
        ba_featsel_mannwhitneyu.append(balanced_accuracy_mannwhitneyu)

        balanced_accuracy_mrmr = np.asarray(
            balanced_accuracies['balanced_accuracies_mrmr'][currentsplit][index]
            ) 
        ba_featsel_mrmr.append(balanced_accuracy_mrmr)

    ba_featsel_mannwhitneyu = np.asarray(ba_featsel_mannwhitneyu)

    mean_balanced_accuracies_mannwhitneyu.append(np.mean(ba_featsel_mannwhitneyu))
    min_balanced_accuracies_mannwhitneyu.append(np.min(ba_featsel_mannwhitneyu))
    max_balanced_accuracies_mannwhitneyu.append(np.max(ba_featsel_mannwhitneyu))
    std_balanced_accuracies_mannwhitneyu.append(np.std(ba_featsel_mannwhitneyu))


    ba_featsel_mrmr = np.asarray(ba_featsel_mrmr)

    mean_balanced_accuracies_mrmr.append(np.mean(ba_featsel_mrmr))
    min_balanced_accuracies_mrmr.append(np.min(ba_featsel_mrmr))
    max_balanced_accuracies_mrmr.append(np.max(ba_featsel_mrmr))
    std_balanced_accuracies_mrmr.append(np.std(ba_featsel_mrmr))

    # Try to find name of selected features that leads to the best prediction
    if np.mean(ba_featsel_mannwhitneyu) > best_mean_mannwhitneyu:
        nbr_kept_features_mannwhitneyu = index + 1
        kept_features_mannwhitneyu = [selfeat[0:index] for selfeat in selfeat_mannwhitneyu_names_allsplits]
        best_mean_mannwhitneyu = np.mean(ba_featsel_mannwhitneyu)

    if np.mean(ba_featsel_mrmr) > best_mean_mrmr:
        nbr_kept_features_mrmr = index + 1
        kept_features_mrmr = [selfeat[0:index] for selfeat in selfeat_mrmr_names_allsplits]
        best_mean_mrmr = np.mean(ba_featsel_mrmr)





balanced_accuracies_boruta = list()


for i in range(nbr_of_splits): 
    currentsplit =  f"split_{i}"
    ba_boruta = (
        [balanced_accuracies['balanced_accuracies_boruta'][currentsplit]]
        )
    balanced_accuracies_boruta.append(ba_boruta)


# Transform a list of list into a list?
balanced_accuracies_boruta = [value[0] for value in balanced_accuracies_boruta]
# Then only keep mean value
ba_boruta_npy = np.asarray(balanced_accuracies_boruta)
ba_boruta_npy = ba_boruta_npy[ba_boruta_npy != None]

mean_ba_boruta_npy = np.mean(ba_boruta_npy)
min_ba_boruta_npy = np.min(ba_boruta_npy)
max_ba_boruta_npy = np.max(ba_boruta_npy)
std_ba_boruta_npy = np.std(ba_boruta_npy)



#mean_ba_boruta_npy = np.asarray(mean_ba_boruta_npy)

# Create a numpy array of max and min number of feat kept by boruta

# check if the number of feat kept is the same for all splits or not 
# if it is the case create another value for visualization purposes
if len(set(number_feat_kept_boruta))==1:
    boruta_visu_xcoord = [number_feat_kept_boruta[0] + 1, number_feat_kept_boruta[0]]
# if there are diff values take min and max of nbr of feature kept
else:
    boruta_visu_xcoord = [min(number_feat_kept_boruta), max(number_feat_kept_boruta)]


boruta_visu_xcoord_npy = np.asarray(boruta_visu_xcoord)
 
mean_ba_mannwhitneyu_npy = np.asarray(mean_balanced_accuracies_mannwhitneyu)
min_ba_mannwhitneyu_npy = np.asarray(min_balanced_accuracies_mannwhitneyu)
max_ba_mannwhitneyu_npy = np.asarray(max_balanced_accuracies_mannwhitneyu)
std_ba_mannwhitneyu_npy = np.asarray(std_balanced_accuracies_mannwhitneyu)

mean_ba_mrmr_npy = np.asarray(mean_balanced_accuracies_mrmr)
min_ba_mrmr_npy = np.asarray(min_balanced_accuracies_mrmr)
max_ba_mrmr_npy = np.asarray(max_balanced_accuracies_mrmr)
std_ba_mrmr_npy = np.asarray(std_balanced_accuracies_mrmr)



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


if run_xgboost and not run_lgbm:
    classifier_name = 'xgboost'
if run_lgbm and not run_xgboost:
    classifier_name = 'lgbm'


print('Start saving numpy in folder: ', save_results_path)

name_mannwhitneyu_output = 'ba_mannwhitneyut_10splits2'
np.save(save_results_path + 'mean_' + classifier_name + name_mannwhitneyu_output + save_ext, 
    mean_ba_mannwhitneyu_npy)
np.save(save_results_path + 'min_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
    min_ba_mannwhitneyu_npy)
np.save(save_results_path + 'max_' + classifier_name + name_mannwhitneyu_output + save_ext, 
    max_ba_mannwhitneyu_npy)
np.save(save_results_path + 'std_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
    std_ba_mannwhitneyu_npy)

name_mrmr_output = 'ba_mrmr_10splits2'
np.save(save_results_path + 'mean_' + classifier_name + name_mrmr_output + save_ext, 
    mean_ba_mrmr_npy)
np.save(save_results_path + 'max_' + classifier_name + name_mrmr_output + save_ext, 
    min_ba_mrmr_npy)
np.save(save_results_path + 'min_' + classifier_name + name_mrmr_output + save_ext, 
    max_ba_mrmr_npy)
np.save(save_results_path + 'std_' + classifier_name + name_mrmr_output + save_ext, 
    std_ba_mrmr_npy)

name_boruta_output = 'ba_boruta_10splits2'
np.save(save_results_path + 'mean_' + classifier_name + name_boruta_output + save_ext, 
    mean_ba_boruta_npy)
np.save(save_results_path + 'max_' + classifier_name + name_boruta_output + save_ext, 
    min_ba_boruta_npy)
np.save(save_results_path + 'min_' + classifier_name + name_boruta_output + save_ext, 
    max_ba_boruta_npy)
np.save(save_results_path + 'std_' + classifier_name + name_boruta_output + save_ext, 
    std_ba_boruta_npy)

np.save(save_results_path + classifier_name  + 'nbr_feat_kept_boruta_10splits2' + save_ext, boruta_visu_xcoord_npy)

print('Numpy saved.')



##############################################################
## Save text file
##############################################################


txtfilename = classifier_name + '_run1_info' 

save_txt_ext = '.txt'
save_text_path = save_results_path + txtfilename + save_txt_ext


## ADD NAME OF CLASSIFER THAT WAS RUNNING

print('Start saving name and number of feature kept in best case')

with open(save_text_path, 'w') as file:
    file.write('** With {} classifier **'.format(classifier_name))

    file.write('\n\n\n\n ** mannwhitneyu **')
    file.write('\n\nBest mean balanced accuracy is:' + str(best_mean_mannwhitneyu))  
    file.write('\n\nThe number of kept features in the best scenario is:' + str(nbr_kept_features_mannwhitneyu))  
    file.write('\n\nThese features are:' +  str(kept_features_mannwhitneyu)) 
    file.write('\n\nThe best 5 features are:' +  str([kept_features[0:4] for kept_features in kept_features_mannwhitneyu])

    file.write('\n\n\n\n ** mrmr **')
    file.write('\n\nBest mean balanced accuracy is:' +  str(best_mean_mrmr))  
    file.write('\n\nThe number of kept features in the best scenario is:' + str(nbr_kept_features_mrmr))  
    file.write('\n\nThese features are:' + str(kept_features_mrmr)) 
    file.write('\n\nThe best 5 features are:' +  str([kept_features[0:4] for kept_features in kept_features_mrmr]) 

    file.write('\n\n\n\n ** boruta **')
    file.write('\n\nMean balanced accuracy is:' +  str(mean_ba_boruta_npy)) 
    file.write('\n\nhe numbers of kept features are:' + str(number_feat_kept_boruta))  
    file.write('\n\nThese features are:' + str(selfeat_boruta_allsplits)) 
    


print('Text files saved.')





