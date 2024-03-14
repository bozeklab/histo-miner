#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import os

from tqdm import tqdm
import random
import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedGroupKFold, train_test_split
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

      
################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################

#This is to check but should be fine

featarray_name = 'perwsi_featarray'
classarray_name = 'perwsi_clarray'
ext = '.npy'

train_featarray = np.load(pathfeatselect + featarray_name + ext)
train_clarray = np.load(pathfeatselect + classarray_name + ext)



##############################################################
## Load Classifiers
##############################################################


##### XGBOOST
xgboost = xgboost.XGBClassifier(random_state= xgboost_random_state,
                                n_estimators=xgboost_n_estimators, 
                                learning_rate=xgboost_lr, 
                                objective=xgboost_objective,
                                verbosity=0)


lightgbm = lightgbm.LGBMClassifier(random_state= lgbm_random_state,
                                   n_estimators=lgbm_n_estimators,
                                   learning_rate=lgbm_lr,
                                   objective=lgbm_objective,
                                   num_leaves=lgbm_numleaves,
                                   verbosity=-1)



# step to run 
step = 4



##############################################################
## 1 Basic - works fine
##############################################################

if step==1:

    print('Running step: ',step)

    print('Path used: ', pathfeatselect)

    print('Start Classifiers trainings...')

    train_featarray = np.transpose(train_featarray)


    X_train, X_test, y_train, y_test = train_test_split(train_featarray, 
                                                        train_clarray, 
                                                        test_size=0.2, 
                                                        random_state=0)


    classifier_training = xgboost
    classifier_training = classifier_training.fit(X_train, y_train) 

    # Make predictions on the test set
    y_pred = classifier_training.predict(X_test)
           

    # Calculate balanced accuracy for the current split
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print('balanced accuracy:', balanced_accuracy)





###############################################################################
## 2 Including cross validation with patient groupts - works
###############################################################################


if step==2:

    print('Running step: ',step)

    print('Path used: ', pathfeatselect)

    # Load patient ids
    path_patientids_array = pathfeatselect + 'patientids' + ext
    patientids_load = np.load(path_patientids_array, allow_pickle=True)
    patientids_list = list(patientids_load)
    patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
    patientids = np.asarray(patientids_convert)



    train_featarray = np.transpose(train_featarray)

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

    # number of splits for cross validation
    nbr_of_splits = 10 

    ### Create Stratified Group to further split the dataset into nbr_of_splits 
    stratgroupkf = StratifiedGroupKFold(n_splits=nbr_of_splits, shuffle=False)

    print('Create splits...')

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


    print('Do cross validation...')

    balanced_accuracies = []

    for i in tqdm(range(nbr_of_splits)):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        classifier_training = xgboost
        classifier_training = classifier_training.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = classifier_training.predict(X_test)

        # Calculate balanced accuracy for the current split
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(balanced_accuracy)


    # Have the mean of balanced accuracies
    balanced_accuracies_numpy = np.asarray(balanced_accuracies)

    mean_bacc = np.mean(balanced_accuracies_numpy)
    print('xgboost slpits balanced accuracies:', balanced_accuracies)
    print('xgboost mean balanced accuracy: {}'.format(mean_bacc))


    # # not obtimized but for test


    balanced_accuracies = []

    for i in tqdm(range(nbr_of_splits)):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        classifier_training = lightgbm
        classifier_training = classifier_training.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = classifier_training.predict(X_test)

        # Calculate balanced accuracy for the current split
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(balanced_accuracy)


    # Have the mean of balanced accuracies
    balanced_accuracies_numpy = np.asarray(balanced_accuracies)

    mean_bacc = np.mean(balanced_accuracies_numpy)
    print('lightgbm slpits balanced accuracies:', balanced_accuracies)
    print('lightgbm mean balanced accuracy: {}'.format(mean_bacc))





###############################################################################
## 3 Including cross validation with patient groupts 
##   and permutations - works
###############################################################################


if step==3:

    print('Running step: ',step)

    print('Path used: ', pathfeatselect)

    # Load patient ids
    path_patientids_array = pathfeatselect + 'patientids' + ext
    patientids_load = np.load(path_patientids_array, allow_pickle=True)
    patientids_list = list(patientids_load)
    patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
    patientids = np.asarray(patientids_convert)

    # permutation_index = np.load(pathtomain + 
    #                         '/bestperm/' +
    #                         'random_permutation_index_11_28_xgboost_bestmean.npy')
    # nbrindeces = len(permutation_index)

    permutation_index = np.random.permutation(train_clarray.size)

    ### Shuffle classification arrays using the permutation index
    train_clarray = train_clarray[permutation_index]

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

    # number of splits for cross validation
    nbr_of_splits = 10 

    ### Create Stratified Group to further split the dataset into 5 
    stratgroupkf = StratifiedGroupKFold(n_splits=nbr_of_splits, shuffle=False)

    print('Create splits...')

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


    print('Do cross validation...')

    balanced_accuracies = []

    for i in tqdm(range(nbr_of_splits)):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        classifier_training = xgboost
        classifier_training = classifier_training.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = classifier_training.predict(X_test)

        # Calculate balanced accuracy for the current split
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(balanced_accuracy)


    # Have the mean of balanced accuracies
    balanced_accuracies_numpy = np.asarray(balanced_accuracies)

    mean_bacc = np.mean(balanced_accuracies_numpy)
    print('xgboost slpits balanced accuracies:', balanced_accuracies)
    print('xgboost mean balanced accuracy: {}'.format(mean_bacc))


    # # not obtimized but for test


    balanced_accuracies = []

    for i in tqdm(range(nbr_of_splits)):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        classifier_training = lightgbm
        classifier_training = classifier_training.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = classifier_training.predict(X_test)

        # Calculate balanced accuracy for the current split
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracies.append(balanced_accuracy)


    # Have the mean of balanced accuracies
    balanced_accuracies_numpy = np.asarray(balanced_accuracies)

    mean_bacc = np.mean(balanced_accuracies_numpy)
    print('lightgbm balanced accuracies:', balanced_accuracies)
    print('lightgbm balanced accuracy: {}'.format(mean_bacc))





###############################################################################
## 4 training keep one class (done here again)- worked
###############################################################################

if step==4:

    print('Running step: ',step)

    print('Path used: ', pathfeatselect)

    # Load patient ids
    path_patientids_array = pathfeatselect + 'patientids' + ext
    patientids_load = np.load(path_patientids_array, allow_pickle=True)
    patientids_list = list(patientids_load)
    patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
    patientids = np.asarray(patientids_convert)

    # permutation_index = np.load(pathtomain + 
    #                         '/bestperm/' +
    #                         'random_permutation_index_11_28_xgboost_bestmean.npy')
    # nbrindeces = len(permutation_index)

    permutation_index = np.random.permutation(train_clarray.size)


    ### Shuffle classification arrays using the permutation index
    train_clarray = train_clarray[permutation_index]

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

    # number of splits for cross validation
    nbr_of_splits = 10 

    ### Create Stratified Group to further split the dataset into 5 
    stratgroupkf = StratifiedGroupKFold(n_splits=nbr_of_splits, shuffle=False)

    print('Create splits...')

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


    print('Do cross validation...')

    balanced_accuracies = []

    for i in tqdm(range(nbr_of_splits)):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        classifier_training = xgboost
        classifier_training = classifier_training.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = classifier_training.predict(X_test)

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
    print('xgboost slpits balanced accuracies:', balanced_accuracies)
    print('xgboost mean balanced accuracy: {}'.format(mean_bacc))

    

    balanced_accuracies = []

    for i in tqdm(range(nbr_of_splits)):  

        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        classifier_training = lightgbm
        classifier_training = classifier_training.fit(X_train, y_train) 

        # Make predictions on the test set
        y_pred = classifier_training.predict(X_test)

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
    print('lightgbm slpits balanced accuracies:', balanced_accuracies)
    print('lightgbm mean balanced accuracy: {}'.format(mean_bacc))





##### Stop working after feature selection it seems. So looks like feature selection is messing 
# Could be the feature selection itslef or when keeping number of features. Maybe I can try other strategies
# before feature selection to see how it evolves



#######################################################################
## 5 model based ranking - on going progress 
########################################################################


if step==5:

    print('Running step: ',step)

    print('Path used: ', pathfeatselect)








































#######################################################################
## 6 classic training - on going progress 
########################################################################