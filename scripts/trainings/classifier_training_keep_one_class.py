#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

from tqdm import tqdm
import random
import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
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
pathtomain = confighm.paths.folders.main
pathfeatselect = confighm.paths.folders.feature_selection_main

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

ext = '.npy'

print('Load feature selection numpy files...')

# Load feature selection numpy files
pathselfeat_boruta = pathfeatselect + '/selfeat_boruta_idx_depth18' + ext
selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
print('Loading feature selected indexes done.')



################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################

#This is to check but should be fine

featarray_name = 'perwsi_featarray'
classarray_name = 'perwsi_clarray'
ext = '.npy'

train_featarray = np.load(pathfeatselect + featarray_name + ext)
train_clarray = np.load(pathfeatselect + classarray_name + ext)


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
permutation_index = np.load(pathfeatselect + 
                            '/bestperm/' +
                            'random_permutation_index_11_28_xgboost_bestmean.npy')
nbrindeces = len(permutation_index)

### Shuffle classification arrays using the permutation index
train_clarray = train_clarray[permutation_index]


# Generate the matrix with selected feature for boruta
selected_features_matrix = SelectedFeaturesMatrix(train_featarray)
featarray_boruta = selected_features_matrix.boruta_matr(selfeat_boruta)


# Shuffle selected feature arrays using the permutation index 
featarray_boruta = featarray_boruta[permutation_index, :]

# Shuffle the all features arrays using the permutation index
train_featarray = np.transpose(train_featarray)
train_featarray = train_featarray[permutation_index, :]


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
patientids_ordered = patientids_ordered[permutation_index]

### Create Stratified Group to further split the dataset into 5 
stratgroupkf = StratifiedGroupKFold(n_splits=10, shuffle=False)


# Need another instance of the classifier
lgbm_slide_ranking = lightgbm
xgboost_slide_ranking = xgboost


if classification_from_allfeatures:
    # Create a list of splits with all features 
    splits_nested_list = list()
    # Create a list of patient IDs corresponding of the splits:
    splits_patientID_list = list()
    for i, (train_index, test_index) in enumerate(stratgroupkf.split(train_featarray, 
                                                                     train_clarray, 
                                                                     groups=patientids_ordered)):
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

else:
    # Create a list of splits indeces with the selected features 
    splits_nested_list = list()
    # Create a list of patient IDs corresponding of the splits:
    splits_patientID_list = list()
    for i, (train_index, test_index) in enumerate(stratgroupkf.split(featarray_boruta, 
                                                                     train_clarray, 
                                                                     groups=patientids_ordered)):
        # Generate training and test data from the indexes
        X_train = featarray_boruta[train_index]
        X_test = featarray_boruta[test_index]
        y_train = train_clarray[train_index]
        y_test = train_clarray[test_index]

        splits_nested_list.append([X_train, y_train, X_test, y_test])

        # Generate the corresponding list for patient ids
        X_train_patID = patientids_ordered[train_index]
        X_test_patID = patientids_ordered[test_index]

        splits_patientID_list.append([X_train_patID, X_test_patID])



# With the new version of slide selection, everything should be done in one loop,
# with beginning of the loop beeing the slide selection I think

# -- XGBOOST --

nbr_of_splits = 10 # Assuming 10 splits

if run_xgboost and not run_lgbm:

    balanced_accuracies = []
    list_proba_predictions_slideselect = []
    

    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        xgboost_training = xgboost
        xgboost_training = xgboost_training.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = xgboost_training.predict(X_test)

        # Now we should keep one prediction per patient for evaluation, the most represented prediction
        # load the corresponding ID lists
        X_train_patID = splits_patientID_list[i][0]
        X_test_patID = splits_patientID_list[i][1]
        
        # Dictionary to store the index of the highest probability score for each group
        y_pred_per_patient = []
        y_test_per_patient = []

        # Iterate through groups
        for patientid in np.unique(X_test_patID):
            # Get the indices of samples belonging to the current group
            slide_indices = np.where(X_test_patID == patientid)[0]

            # If the patient has more than one slide, we keep the one with most prediction
            if len(slide_indices) > 1:
                # Get the probability predictions of class 0 for each slides of the patient
                patient_slides_probas_class0 = [y_pred[index] for index in slide_indices]
                
                # HVE TO CHECK WHAT IS CLASS ONE AND WHAT IS CLASS 2
                count_norec = patient_slides_probas_class0.count(0)
                count_rec =  patient_slides_probas_class0.count(1)
                if count_norec > count_rec:
                    y_pred_per_patient.append(int(0))
                    y_test_per_patient.append(y_test[slide_indices][0])
                else:
                    y_pred_per_patient.append(int(1))
                    y_test_per_patient.append(y_test[slide_indices][0])

            else: 
                y_pred_per_patient.append(y_pred[slide_indices][0])
                y_test_per_patient.append(y_test[slide_indices][0])

        # Calculate balanced accuracy for the current split
        balanced_accuracy = balanced_accuracy_score(y_test_per_patient, y_pred_per_patient)
        balanced_accuracies.append(balanced_accuracy)

    # Have the mean of balanced accuracies
    balanced_accuracies_numpy = np.asarray(balanced_accuracies)


    mean_bacc = np.mean(balanced_accuracies_numpy)
    print('slpits balanced accuracies:', balanced_accuracies)
    print('mean balanced accuracy for xgboost: {}'.format(mean_bacc))
        


# -- LGBM --

elif run_lgbm and not run_xgboost:

    balanced_accuracies = []
    list_proba_predictions_slideselect = []

    for i in range(nbr_of_splits):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        lgbm_slide_ranking = lightgbm
        lgbm_slide_ranking = lgbm_slide_ranking.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = lgbm_slide_ranking.predict(X_test)

        # Now we should keep one prediction per patient for evaluation, the most represented prediction
        # load the corresponding ID lists
        X_train_patID = splits_patientID_list[i][0]
        X_test_patID = splits_patientID_list[i][1]
        
        # Dictionary to store the index of the highest probability score for each group
        y_pred_per_patient = []
        y_test_per_patient = []

        # Iterate through groups
        for patientid in np.unique(X_test_patID):
            # Get the indices of samples belonging to the current group
            slide_indices = np.where(X_test_patID == patientid)[0]

            # If the patient has more than one slide, we keep the one with most prediction
            if len(slide_indices) > 1:
                # Get the probability predictions of class 0 for each slides of the patient
                patient_slides_probas_class0 = [y_pred[index] for index in slide_indices]
                
                # HVE TO CHECK WHAT IS CLASS ONE AND WHAT IS CLASS 2
                count_norec = patient_slides_probas_class0.count(0)
                count_rec =  patient_slides_probas_class0.count(1)
                if count_norec > count_rec:
                    y_pred_per_patient.append(int(0))
                    y_test_per_patient.append(y_test[slide_indices][0])
                else:
                    y_pred_per_patient.append(int(1))
                    y_test_per_patient.append(y_test[slide_indices][0])

            else: 
                y_pred_per_patient.append(y_pred[slide_indices][0])
                y_test_per_patient.append(y_test[slide_indices][0])

        # Calculate balanced accuracy for the current split
        balanced_accuracy = balanced_accuracy_score(y_test_per_patient, y_pred_per_patient)
        balanced_accuracies.append(balanced_accuracy)

    # Have the mean of balanced accuracies
    balanced_accuracies_numpy = np.asarray(balanced_accuracies)


    mean_bacc = np.mean(balanced_accuracies_numpy)
    print('slpits balanced accuracies:', balanced_accuracies)
    print('mean balanced accuracy for lgbm: {}'.format(mean_bacc))



else:
    raise ValueError('run_xgboost and run_lgbm cannot be both True or both False for'
                      'the script to run')





