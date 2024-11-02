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
boruta_max_depth = confighm.parameters.int.boruta_max_depth
boruta_random_state = confighm.parameters.int.boruta_random_state



with open("./../../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
nbr_of_splits = config.parameters.int.nbr_of_splits
run_name = config.names.run_name

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
# lightgbm = lightgbm.LGBMClassifier(random_state= lgbm_random_state,
#                                    n_estimators=lgbm_n_estimators,
#                                    learning_rate=lgbm_lr,
#                                    objective=lgbm_objective,
#                                    num_leaves=lgbm_numleaves,
#                                    verbosity=-1)
param_lightgbm = {
                  "random_state": lgbm_random_state,
                  "n_estimators": lgbm_n_estimators,
                  "learning_rate": lgbm_lr,
                  "objective":lgbm_objective,
                  "num_leaves":lgbm_numleaves,
                  "verbosity":-1
                  }

#RMQ: Verbosity is set to 0 for XGBOOST to avoid printing WARNINGS (not wanted here for sake of
#simplicity)/ In Light GBM, to avoid showing WARNINGS, the verbosity as to be set to -1.
# See parameters documentation to learn about the other verbosity available. 


##############################################################
## Traininig Classifiers to obtain instance prediction score
##############################################################


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

    balanced_accuracies = {"balanced_accuracies_boruta": {"initialization": True}}

    selfeat_boruta_names_allsplits = []
    selfeat_boruta_id_allsplits = []
    number_feat_kept_boruta = []

    all_features_balanced_accuracy = list()

    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        # The array is then transpose to feat FeatureSelector requirements
        X_train_tr = np.transpose(X_train)

        #Boruta cannot take any NANs
        nan_raws_indices = np.where(np.any(np.isnan(X_train_tr), axis=1))

        # Remove those columns from y_train and X_train
        X_train_tr = np.delete(X_train_tr, nan_raws_indices, axis=0)
        y_train = np.delete(y_train, nan_raws_indices)

        ########### SELECTION OF FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        if i == 0:
            feature_selector = FeatureSelector(X_train_tr, y_train)
        else: 
            feature_selector.reset_attributes(X_train_tr, y_train)

        # Boruta calculations (for one specific depth)
        print('Selection of features with Boruta method...')
        selfeat_boruta_index = feature_selector.run_boruta(max_depth=boruta_max_depth, 
                                                           random_state=boruta_random_state)
        nbrfeatsel_boruta = len(selfeat_boruta_index)
        number_feat_kept_boruta.append(nbrfeatsel_boruta)
        # Now associate the index of selected features (selfeat_boruta_index) to the list of names:
        selfeat_boruta_names = [featnameslist[index] for index in selfeat_boruta_index]
        selfeat_boruta_names_allsplits.append(selfeat_boruta_names)
        selfeat_boruta_id_allsplits.append(selfeat_boruta_index)


        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train
        ## For Boruta calculations
        if nbrfeatsel_boruta == 0:      
            featarray_boruta = np.array([])             
        else:
            featarray_boruta = feature_array[:, np.transpose(selfeat_boruta_index)]


        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION

        ### With boruta selected features
        # First we train with all features 
        #Training
        xgboost_training_allfeat = xgboost.fit(X_train, y_train)

        # Predictions on the test split
        y_pred_allfeat = xgboost_training_allfeat.predict(X_test)

        # Calculate balanced accuracy for the current split
        balanced_accuracy_allfeat = balanced_accuracy_score(y_test, 
                                                            y_pred_allfeat)

        # Update mrmr list with the all feature evaluation
        all_features_balanced_accuracy.append(balanced_accuracy_allfeat)


        # Then we do classification with selected features only 
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
        balanced_accuracies['balanced_accuracies_boruta'][currentsplit] = balanced_accuracy_boruta




##############################################################
##  Classifying with LGBM 
##############################################################


elif run_lgbm and not run_xgboost:

    balanced_accuracies = {"balanced_accuracies_boruta": {"initialization": True}}

    selfeat_boruta_names_allsplits = []
    selfeat_boruta_id_allsplits = []
    number_feat_kept_boruta = []

    all_features_balanced_accuracy = list()

    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        # The array is then transpose to feat FeatureSelector requirements
        X_train_tr = np.transpose(X_train)

        # #Boruta cannot take any NANs
        # nan_cols = np.any(np.isnan(X_train_tr), axis=0)
        # nan_col_indices = np.where(nan_cols)[0]
        
        # # Remove those columns from y_train and X_train
        # X_train_tr = np.delete(X_train_tr, nan_col_indices, axis=1)
        # y_train = np.delete(y_train, nan_col_indices, axis=1)


        ########### SELECTION OF FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        if i == 0:
            feature_selector = FeatureSelector(X_train_tr, y_train)
        else: 
            feature_selector.reset_attributes(X_train_tr, y_train)

        # Boruta calculations (for one specific depth)
        print('Selection of features with Boruta method...')
        selfeat_boruta_index = feature_selector.run_boruta(max_depth=boruta_max_depth, 
                                                           random_state=boruta_random_state)
        nbrfeatsel_boruta = len(selfeat_boruta_index)
        number_feat_kept_boruta.append(nbrfeatsel_boruta)
        # Now associate the index of selected features (selfeat_boruta_index) to the list of names:
        selfeat_boruta_names = [featnameslist[index] for index in selfeat_boruta_index]
        selfeat_boruta_names_allsplits.append(selfeat_boruta_names)
        selfeat_boruta_id_allsplits.append(selfeat_boruta_index)


        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train
        ## For Boruta calculations
        if nbrfeatsel_boruta == 0:      
            featarray_boruta = np.array([])             
        else:
            featarray_boruta = feature_array[:, np.transpose(selfeat_boruta_index)] 


        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION

        ### With boruta selected features
        # First we train with all features 
        #Training
        train_data = lightgbm.Dataset(X_train, label=y_train)
        lightgbm_training_allfeat = lightgbm.train(
            param_lightgbm,
            train_data
            )

        # Predictions on the test split
        y_pred_allfeat_prob = lightgbm_training_allfeat.predict(
            X_test,
            num_iteration=lightgbm_training_allfeat.best_iteration)
        y_pred_allfeat = (y_pred_allfeat_prob > 0.5).astype(int)

        # Calculate balanced accuracy for the current split
        balanced_accuracy_allfeat = balanced_accuracy_score(y_test, 
                                                            y_pred_allfeat)

        # Update mrmr and mannwhitney list with the all feature evaluation
        all_features_balanced_accuracy.append(balanced_accuracy_allfeat)


        # Then we do classification with selected features only 
        #Training
        # sometimes boruta is not keeping any features, so need to check if there are some
        if np.size(featarray_boruta) == 0:      
            balanced_accuracy_boruta = None

        else:
            train_data = lightgbm.Dataset(featarray_boruta, label=y_train)
            lightgbm_boruta_training = lightgbm.train(
                    param_lightgbm,
                    featarray_boruta
                    )

            # Predictions on the test split
            y_pred_boruta_prob = lightgbm_boruta_training.predict(
                X_test[:, np.transpose(selfeat_boruta_index)]
                )
            y_pred_boruta =  (y_pred_boruta_prob > 0.5).astype(int)

            # Calculate balanced accuracy for the current split
            balanced_accuracy_boruta = balanced_accuracy_score(y_test,
                                                               y_pred_boruta)

        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_boruta'][currentsplit] = balanced_accuracy_boruta


else:
    raise ValueError('run_xgboost and run_lgbm cannot be both True or both False for'
                  'the script to run')



####################################################################
## Extract mean,min,max of balanced accuracy and kept feature names 
####################################################################


## calculate and write the saving of the mean balanced accuracies

### Boruta 

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
 



####################################################################
## Select best features after cross-validation
####################################################################


###### RMQS TO REMOVE

## For Boruta we could try to find another strategy by keeping the group of feat selected by Boruta that 
# have the highest occurence of their feature (and no score here as feat should be seen as a group)

# But this is not working much neither....

#####


# nbr_kept_feat = number_feat_kept_boruta


# # If nbr_kept_feat = number_feat_kept_boruta this variable is useless but kept for dev purposes 
# sorted_bestfeatindex_boruta = utils_misc.find_closest_sublist(
#     selfeat_boruta_id_allsplits, 
#     nbr_kept_feat
#     )

# # Retrieve names of best selected features
# boruta_finalselfeat_names = featnames[sorted_bestfeatindex_boruta]



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


name_boruta_output = '_ba_boruta_' + str(nbr_of_splits) + 'splits_' + run_name
np.save(save_results_path + 'mean_' + classifier_name + name_boruta_output + save_ext, 
    mean_ba_boruta_npy)
np.save(save_results_path + 'max_' + classifier_name + name_boruta_output + save_ext, 
    min_ba_boruta_npy)
np.save(save_results_path + 'min_' + classifier_name + name_boruta_output + save_ext, 
    max_ba_boruta_npy)
np.save(save_results_path + 'std_' + classifier_name + name_boruta_output + save_ext, 
    std_ba_boruta_npy)
# np.save(save_results_path + 'topselfeatid_' + classifier_name  + name_boruta_output + save_ext, 
#     sorted_bestfeatindex_boruta)

np.save(
    save_results_path + classifier_name  + 'nbr_feat_kept_boruta_'  + 
    str(nbr_of_splits) + run_name + save_ext, 
    boruta_visu_xcoord_npy
    )

print('Numpy saved.')



##############################################################
## Save text file
##############################################################


txtfilename = (
    classifier_name +  '_' +
    'boruta' +  '_' +
    str(nbr_of_splits) + 'splits_' +
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

    file.write('\n\n\n\n ** boruta **')
    file.write('\n\nMean balanced accuracy is:' +  
        str(mean_ba_boruta_npy)) 
    file.write('\n\nhe numbers of kept features are:' + 
        str(number_feat_kept_boruta))  
    # file.write('\n\nThe best group of selected feature close of having 5 features is:' +  
    #     str([]))
    file.write('\n\nThese features are:' + 
        str(selfeat_boruta_names_allsplits)) 
    

print('Text files saved.')


#### Save all splits balanced accuracies values 

#### Save roc curve information 



