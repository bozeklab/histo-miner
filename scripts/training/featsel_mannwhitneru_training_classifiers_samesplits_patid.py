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

    balanced_accuracies = {"balanced_accuracies_mannwhitneyu": {"initialization": True}}

    selfeat_mannwhitneyu_names_allsplits = [] 
    selfeat_mannwhitneyu_id_allsplits = [] 


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
        selfeat_mannwhitneyu_id_allsplits.append(selfeat_mannwhitneyu_index)


        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train

        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION
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

                # Update  mannwhitney list with the all feature evaluation
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_allfeat)


            # Then we decrease number of feature kept during training + evaluation
            else:

                # Kept the selected features
                selfeat_mannwhitneyu_index_reduced = selfeat_mannwhitneyu_index[0:nbr_keptfeat_idx]
                selfeat_mannwhitneyu_index_reduced = sorted(selfeat_mannwhitneyu_index_reduced)

                # Generate matrix of features
                featarray_mannwhitneyu = feature_array[:, selfeat_mannwhitneyu_index_reduced]

                #Training
                # needs to be re initialized each time!!!! Very important
                xgboost_mannwhitneyu_training = xgboost 
                # actual training
                xgboost_mannwhitneyu_training_inst = xgboost_mannwhitneyu_training.fit(
                                                                       featarray_mannwhitneyu, 
                                                                       y_train
                                                                       )

                # Predictions on the test split
                y_pred_mannwhitneyu = xgboost_mannwhitneyu_training_inst.predict(
                    X_test[:, selfeat_mannwhitneyu_index_reduced]
                    )

                # Calculate balanced accuracy for the current split
                balanced_accuracy_mannwhitneyu = balanced_accuracy_score(y_test, 
                                                                         y_pred_mannwhitneyu)
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_mannwhitneyu)


        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_mannwhitneyu'][currentsplit] = balanced_accuracies_mannwhitneyu



##############################################################
##  Classifying with LGBM 
##############################################################


elif run_lgbm and not run_xgboost:

 
    balanced_accuracies = {"balanced_accuracies_mannwhitneyu": {"initialization": True}}

    selfeat_mannwhitneyu_names_allsplits = [] 
    selfeat_mannwhitneyu_id_allsplits = [] 


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
        selfeat_mannwhitneyu_id_allsplits.append(selfeat_mannwhitneyu_index)


        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train

        ########## TRAINING AND EVALUATION WITH FEATURE SELECTION
        balanced_accuracies_mannwhitneyu = list()

        print('Calculate balanced_accuracies for decreasing number of features kept')
        ### With mrmr and mannwhitneyu selected features
        for nbr_keptfeat_idx in tqdm(range(nbr_feat, 0, -1)):

            # First we train with all features 
            if nbr_keptfeat_idx == nbr_feat:
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
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_allfeat)


            # Then we decrease number of feature kept during training + evaluation
            else:

                # Kept the selected features
                selfeat_mannwhitneyu_index_reduced = selfeat_mannwhitneyu_index[0:nbr_keptfeat_idx]
                selfeat_mannwhitneyu_index_reduced = sorted(selfeat_mannwhitneyu_index_reduced)

                # Generate matrix of features
                featarray_mannwhitneyu = feature_array[:, selfeat_mannwhitneyu_index_reduced]

                #Training
                train_data = lightgbm.Dataset(featarray_mannwhitneyu, label=y_train)
                lightgbm_mannwhitneyu_training_inst = lightgbm.train(
                    param_lightgbm,
                    train_data,
                    num_round=10)

                # Predictions on the test split
                y_pred_mannwhitneyu_prob = lightgbm_mannwhitneyu_training_inst.predict(
                    X_test[:, selfeat_mannwhitneyu_index_reduced]
                    )
                y_pred_mannwhitneyu = (y_pred_mannwhitneyu_prob > 0.5).astype(int)

                # Calculate balanced accuracy for the current split
                balanced_accuracy_mannwhitneyu = balanced_accuracy_score(y_test, 
                                                                         y_pred_mannwhitneyu)
                balanced_accuracies_mannwhitneyu.append(balanced_accuracy_mannwhitneyu)


        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_mannwhitneyu'][currentsplit] = balanced_accuracies_mannwhitneyu


else:
    raise ValueError('run_xgboost and run_lgbm cannot be both True or both False for'
                  'the script to run')



####################################################################
## Extract mean,min,max of balanced accuracy and kept feature names 
####################################################################


## calculate and write the saving of the mean balanced accuracies
## Do for mannwhitneyu
# Calculate the mean accuracies 

mean_balanced_accuracies_mannwhitneyu = list()
min_balanced_accuracies_mannwhitneyu = list()
max_balanced_accuracies_mannwhitneyu = list()
std_balanced_accuracies_mannwhitneyu = list()

best_mean_mannwhitneyu = 0

# do the list of means for mannwhitneyu and mrmr
for index in range(0, nbr_feat):
    
    ba_featsel_mannwhitneyu = list()

    for i in range(nbr_of_splits):

        currentsplit =  f"split_{i}"

        balanced_accuracy_mannwhitneyu = np.asarray(
            balanced_accuracies['balanced_accuracies_mannwhitneyu'][currentsplit][index]
            ) 

        ba_featsel_mannwhitneyu.append(balanced_accuracy_mannwhitneyu)
    
    ba_featsel_mannwhitneyu = np.asarray(ba_featsel_mannwhitneyu)

    mean_balanced_accuracies_mannwhitneyu.append(np.mean(ba_featsel_mannwhitneyu))
    min_balanced_accuracies_mannwhitneyu.append(np.min(ba_featsel_mannwhitneyu))
    max_balanced_accuracies_mannwhitneyu.append(np.max(ba_featsel_mannwhitneyu))
    std_balanced_accuracies_mannwhitneyu.append(np.std(ba_featsel_mannwhitneyu))

    ###### Find name of selected features that leads to the best prediction
    # it is not optimised because some if condition could habe been written previously
    # but it is the easiest to read
   
    if np.mean(ba_featsel_mannwhitneyu) > best_mean_mannwhitneyu:
        nbr_kept_features_mannwhitneyu = nbr_feat - index
        kept_features_mannwhitneyu = [split[0: nbr_kept_features_mannwhitneyu] for split in selfeat_mannwhitneyu_names_allsplits]
        best_mean_mannwhitneyu = np.mean(ba_featsel_mannwhitneyu)


mean_ba_mannwhitneyu_npy = np.asarray(mean_balanced_accuracies_mannwhitneyu)
min_ba_mannwhitneyu_npy = np.asarray(min_balanced_accuracies_mannwhitneyu)
max_ba_mannwhitneyu_npy = np.asarray(max_balanced_accuracies_mannwhitneyu)
std_ba_mannwhitneyu_npy = np.asarray(std_balanced_accuracies_mannwhitneyu)




####################################################################
## Select best features after cross-validation
####################################################################

# We will just keep the features that are most kept on the cross
# If there is a tie, we calculate their scores, if there are most 1s 
# than we keep them


# We set the number of features kept for each splits again
nbr_kept_feat = nbr_kept_features_mannwhitneyu


# Create a score for feature selection
prescores = [10**(index) for index in range(nbr_kept_feat,0,-1)]

featscores = prescores * nbr_of_splits


##### Best slected features by Mannwhitney U 

# We start by counting number of occurance in top 5
bestsel_mannwhitneyu_index = [
    featindex 
    for splitindex in range(0, nbr_of_splits)
    for featindex in selfeat_mannwhitneyu_id_allsplits[splitindex][0:nbr_kept_feat]  
    ]

# Count occurrences of each featindex
featindex_counts_mannwhitneyu = Counter(bestsel_mannwhitneyu_index)

# Calculate score of each featindex (in case of draw we advantage feat arriving first more often)
score_sums = defaultdict(int)
for idx, featindex in enumerate(bestsel_mannwhitneyu_index):
    score_sums[featindex] += featscores[idx]

# Take log10 for all values
score_sums = {key: math.log10(value) for key, value in score_sums.items()}

# Sort featindex by occurrence count and then by  scores
sorted_bestfeatindex_mannwhitneyu = sorted(
        featindex_counts_mannwhitneyu.keys(), 
        key=lambda x: (-featindex_counts_mannwhitneyu[x], -score_sums[x])
        )

# Retrieve names of best selected features
mannwhitneyu_finalselfeat_names = list(featnames[sorted_bestfeatindex_mannwhitneyu[0:nbr_kept_feat]])


# create a list with  featindex count and score 
# Create a dictionary to store the best features with their count and score
best_features_info = {
    featnames[featindex]: {
        'count': featindex_counts_mannwhitneyu[featindex],
        'score': score_sums[featindex]
    }
    for featindex in sorted_bestfeatindex_mannwhitneyu[0:len(bestsel_mannwhitneyu_index)]
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


if run_xgboost and not run_lgbm:
    classifier_name = 'xgboost'
if run_lgbm and not run_xgboost:
    classifier_name = 'lgbm'


print('Start saving numpy in folder: ', save_results_path)

name_mannwhitneyu_output = '_ba_mannwhitneyut_' + str(nbr_of_splits) + 'splits_' + run_name
np.save(save_results_path + 'mean_' + classifier_name + name_mannwhitneyu_output + save_ext, 
    mean_ba_mannwhitneyu_npy)
np.save(save_results_path + 'min_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
    min_ba_mannwhitneyu_npy)
np.save(save_results_path + 'max_' + classifier_name + name_mannwhitneyu_output + save_ext, 
    max_ba_mannwhitneyu_npy)
np.save(save_results_path + 'std_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
    std_ba_mannwhitneyu_npy)
np.save(save_results_path + 'topselfeatid_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
    sorted_bestfeatindex_mannwhitneyu)


print('Numpy saved.')



##############################################################
## Save text file
##############################################################


txtfilename = (
    classifier_name +  '_' +
    'mannwhitneyu' +  '_' +
    str(nbr_of_splits) + 'splits_' +
    run_name + '_info'
)

save_txt_ext = '.txt'
save_text_path = save_results_path + txtfilename + save_txt_ext


## ADD NAME OF CLASSIFER THAT WAS RUNNING

print('Start saving name and number of feature kept in best case')

with open(save_text_path, 'w') as file:
    file.write('** With {} classifier **'.format(classifier_name))

    file.write('\n\n\n\n ** mannwhitneyu **')
    file.write('\n\nBest mean balanced accuracy is:' + 
        str(best_mean_mannwhitneyu))  
    file.write('\n\nThe number of kept features in the best scenario is:' + 
        str(nbr_kept_features_mannwhitneyu))  
    # file.write('\n\nThese features are:' +  
    #     str(kept_features_mannwhitneyu)) 
    file.write('\n\nThe best 5 features overall are:' +  
        str([best_features_info]))
    # file.write('\n\nThe best 5 features are:' +  
    # str([kept_features[0:4] for kept_features in kept_features_mannwhitneyu]))

print('Text files saved.')


#### Save all splits balanced accuracies values 

#### Save roc curve information 




