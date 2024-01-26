#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

from tqdm import tqdm
import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedGroupKFold

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
pathselfeat_boruta = pathfeatselect + 'selfeat_boruta_idx_depth18' + ext
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
## Traininig Classifiers to obtain instance prediction score
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
SelectedFeaturesMatrix = SelectedFeaturesMatrix(train_featarray)
featarray_boruta = SelectedFeaturesMatrix.boruta_matr(selfeat_boruta)


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

### Create list of 5 model names
### It is needed as very scikit learn model NEEDS a different name  
modelnames = []
for i in range(5):
    model_name = f"model_{i}"
    modelnames.append(model_name)

# Initialize the list of trained_models:
trained_models = []

# Need another instance of the classifier
lgbm_slide_ranking = lightgbm
xgboost_slide_ranking = xgboost


if classification_from_allfeatures:
    # Create a list of splits indeces with all features 
    splits_nested_list = list()
    for i, (train_index, test_index) in enumerate(stratgroupkf.split(train_featarray, 
                                                                     train_clarray, 
                                                                     groups=patientids_ordered)):
        # Generate training and test data from the indexes
        X_train, X_test = train_featarray[train_index], train_featarray[test_index]
        y_train, y_test = train_clarray[train_index], train_clarray[test_index]

        splits_nested_list.append([X_train, y_train, X_test, y_test])

else:
    # Create a list of splits indeces with the selected features 
    splits_nested_list = list()
    for i, (train_index, test_index) in enumerate(stratgroupkf.split(featarray_boruta, 
                                                                     train_clarray, 
                                                                     groups=patientids_ordered)):
        # Generate training and test data from the indexes
        X_train, X_test = featarray_boruta[train_index], featarray_boruta[test_index]
        y_train, y_test = train_clarray[train_index], train_clarray[test_index]

        splits_nested_list.append([X_train, y_train, X_test, y_test])


# -- XGBOOST --
if run_xgboost and not run_lgbm:

    list_proba_predictions = []

    for i in range(10):  # Assuming you have 10 splits
        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        xgboost_slide_ranking = xgboost
        xgboost_slide_ranking = xgboost_slide_ranking.fit(X_train, 
                                                          y_train, 
                                                          eval_set=[(X_test, y_test)])
        
        if classification_from_allfeatures:
            proba_predictions = xgboost_slide_ranking.predict_proba(train_featarray)
        else:
            proba_predictions = xgboost_slide_ranking.predict_proba(featarray_boruta)
        
        # Keep only the highest of the 2 probabilities 
        highest_proba_prediction = np.max(proba_predictions, axis=1)
        list_proba_predictions.append(highest_proba_prediction)


# -- LGBM --
elif run_lgbm and not run_xgboost:

    list_proba_predictions = []

    for i in range(10):  # Assuming you have 10 splits
        X_train = splits_nested_list[i][0]
        y_train = splits_nested_list[i][1]
        X_test = splits_nested_list[i][2]
        y_test = splits_nested_list[i][3]
        
        lgbm_slide_ranking = lightgbm
        lgbm_slide_ranking = lgbm_slide_ranking.fit(X_train, 
                                                    y_train, 
                                                    eval_set=[(X_test, y_test)])
        
        if classification_from_allfeatures:
            proba_predictions = lgbm_slide_ranking.predict_proba(train_featarray)
        else:
            proba_predictions = lgbm_slide_ranking.predict_proba(featarray_boruta)
        
        # Keep only the highest of the 2 probabilities 
        highest_proba_prediction = np.max(proba_predictions, axis=1)
        list_proba_predictions.append(highest_proba_prediction)

else:
    raise ValueError('run_xgboost and run_lgbm cannot be both True or both False for'
                      'the script to run')



# Calculate the mean along axis 0
mean_proba_predictions = np.mean(list_proba_predictions, axis=0)



##############################################################
## Keep highest score for each patient
##############################################################


# Dictionary to store the index of the highest probability score for each group
idx_most_representative_slide_per_patient = []
nbrpatient = max(patientids_ordered)

# Iterate through groups
for patientid in np.unique(patientids_ordered):
    # Get the indices of samples belonging to the current group
    slide_indices = np.where(patientids_ordered == patientid)[0]

    # Get the mean probability predictions for the current group
    patient_mean_probas = mean_proba_predictions[slide_indices]

    # Find the index of the maximum probability score
    max_proba_index = np.argmax(patient_mean_probas)

    # Store the index in the dictionary
    idx_most_representative_slide_per_patient.append(slide_indices[max_proba_index])


idx_most_representative_slide_per_patient = [val - 1 for val 
                                             in idx_most_representative_slide_per_patient]


# Generate new feature matrix and new classification array
# Generate classification array of only representative slides
train_clarray_refined = train_clarray[idx_most_representative_slide_per_patient]

# Generate the features of representatative slides with all features
feat_representative_slides = train_featarray[idx_most_representative_slide_per_patient,:]
# Generate the features of representatative slides with selected features
feat_representative_slides_boruta = featarray_boruta[idx_most_representative_slide_per_patient,:]



##############################################################
## Save idx and feat array of most representative slides 
##############################################################

# Save ID of the most representative slides! (becareful to permute again) 
# We need to permute again to retreive original order
# The - 1 is needed to find back the original indexes starting to 0 and not 1
idx_most_representative_slides_export = [
    permutation_index[idx_most_representative_slide_per_patient]
    ][0]
# idx_most_representative_slides_export = [len(permutation_index) - index - 1 for index 
#                                          in idx_most_representative_slide_per_patient]
idx_most_representative_slides_export = np.asarray(idx_most_representative_slides_export)

# this export whould be tested back by loading the indeces here and check 
# if the result are similar*                         

# saving
if run_xgboost and not run_lgbm:
    
    if classification_from_allfeatures:
        np.save(pathfeatselect + 
                'most_representative_slides_all_features_xgboost_idx.npy', 
                 idx_most_representative_slides_export) 
    else:    
        np.save(pathfeatselect + 
                'most_representative_slides_selfeatures_xgboost_idx.npy', 
                idx_most_representative_slides_export) 

elif run_lgbm and not run_xgboost:

    if classification_from_allfeatures:
        np.save(pathfeatselect + 
                'most_representative_slides_all_features_lgbm_idx.npy', 
                 idx_most_representative_slides_export) 
    else:    
        np.save(pathfeatselect + 
                'most_representative_slides_selfeatures_lgbm_idx.npy', 
                idx_most_representative_slides_export) 


### HERE IS THE WAY TO LOAD THE INDEXES AND MAKE THEM COMPTAIBLE WITH THE PERMUTATIONS (IN OTHER CODES)
# idx_mostrepr_test = np.load(pathfeatselect + 
#                 'most_representative_slides_all_features_xgboost_idx.npy')
# # idx_mostrepr_test2 = [i for i in range(len(permutation_index)) if permutation_index[i] in idx_mostrepr_test ]

# idx_mostrepr_test2 = list()
# for value in idx_mostrepr_test:
#     if value in permutation_index:
#         new_index = np.where(permutation_index == value)[0]
#         idx_mostrepr_test2.append(new_index)

# idx_mostrepr_test2 = np.asarray(idx_mostrepr_test2)
# idx_mostrepr_test2 = idx_mostrepr_test2[:,0]


# Save new arrays (feature and classification)
if run_xgboost and not run_lgbm:
    np.save(pathfeatselect + 'repslidesx_clarray.npy', 
                train_clarray_refined)

    if classification_from_allfeatures:
        feat_representative_slides_export = np.transpose(
            feat_representative_slides
            )
        np.save(pathfeatselect + 'repslidesx_featarray.npy', 
                feat_representative_slides_export)
    else:
        feat_representative_slides_boruta_export = np.transpose(
            feat_representative_slides_boruta
            )
        np.save(pathfeatselect + 'repslidesx_selectfeat.npy', 
                feat_representative_slides_boruta_export)


elif run_lgbm and not run_xgboost:
    np.save(pathfeatselect + 'repslidesl_clarray.npy', 
                train_clarray_refined)

    if classification_from_allfeatures:
        feat_representative_slides_export = np.transpose(
            feat_representative_slides
            )
        np.save(pathfeatselect + 'repslidesl_featarray.npy', 
                feat_representative_slides_export) 
    else:
        feat_representative_slides_boruta_export = np.transpose(
            feat_representative_slides_boruta
            )
        np.save(pathfeatselect + 'repslidesl_selectfeat.npy', 
                feat_representative_slides_boruta_export) 



### SEPERATE THIS PART IN A NEAR FUTURE ?


##############################################################
## Cross validation of classifiers with selected instances
##############################################################

# I have now  to either do the learning here with the kept sample 

# Need another instance of the classifier
lgbm_training = lightgbm
xgboost_training = xgboost

# -- XGBOOST --
### Evaluate with cross validation for xgboost
if run_xgboost and not run_lgbm:
    if classification_from_allfeatures:
        # Evaluate with cross validation for lgbm with selected features (representative slides)
        crossvalid_results_refined = cross_val_score(xgboost_training, 
                                                     feat_representative_slides, 
                                                     train_clarray_refined,  
                                                     cv=10,  
                                                     scoring='balanced_accuracy')

        crossvalidrep_meanscore = np.mean(crossvalid_results_refined)
        crossvalidrep_maxscore = np.max(crossvalid_results_refined)

        # Evaluate with cross validation for lgbm with selected features (all slides)
        crossvalid_results_original = cross_val_score(xgboost_training, 
                                                      train_featarray, 
                                                      train_clarray,  
                                                      cv=10,  
                                                      scoring='balanced_accuracy')

        crossvalidor_meanscore = np.mean(crossvalid_results_original)
        crossvalidor_maxscore = np.max(crossvalid_results_original)



    else:
        # Evaluate with cross validation for lgbm with selected features (representative slides)
        crossvalid_results_refined = cross_val_score(xgboost_training, 
                                                     feat_representative_slides_boruta, 
                                                     train_clarray_refined,  
                                                     cv=10,  
                                                     scoring='balanced_accuracy')

        crossvalidrep_meanscore = np.mean(crossvalid_results_refined)
        crossvalidrep_maxscore = np.max(crossvalid_results_refined)

        # Evaluate with cross validation for lgbm with selected features (all slides)
        crossvalid_results_original = cross_val_score(xgboost_training, 
                                                      featarray_boruta, 
                                                      train_clarray,  
                                                      cv=10,  
                                                      scoring='balanced_accuracy')

        crossvalidor_meanscore = np.mean(crossvalid_results_original)
        crossvalidor_maxscore = np.max(crossvalid_results_original)


# -- LGBM --
### Evaluate with cross validation for lgbm
elif run_lgbm and not run_xgboost:
    if classification_from_allfeatures:
        # Evaluate with cross validation for lgbm with selected features (representative slides)
        crossvalid_results_refined = cross_val_score(lgbm_training, 
                                                     feat_representative_slides, 
                                                     train_clarray_refined,  
                                                     cv=10,  
                                                     scoring='balanced_accuracy')

        crossvalidrep_meanscore = np.mean(crossvalid_results_refined)
        crossvalidrep_maxscore = np.max(crossvalid_results_refined)

        # Evaluate with cross validation for lgbm with selected features (all slides)
        crossvalid_results_original = cross_val_score(lgbm_training, 
                                                      train_featarray, 
                                                      train_clarray,  
                                                      cv=10,  
                                                      scoring='balanced_accuracy')

        crossvalidor_meanscore = np.mean(crossvalid_results_original)
        crossvalidor_maxscore = np.max(crossvalid_results_original)



    else:
        # Evaluate with cross validation for lgbm with selected features (representative slides)
        crossvalid_results_refined = cross_val_score(lgbm_training, 
                                                     feat_representative_slides_boruta, 
                                                     train_clarray_refined,  
                                                     cv=10,  
                                                     scoring='balanced_accuracy')

        crossvalidrep_meanscore = np.mean(crossvalid_results_refined)
        crossvalidrep_maxscore = np.max(crossvalid_results_refined)

        # Evaluate with cross validation for lgbm with selected features (all slides)
        crossvalid_results_original = cross_val_score(lgbm_training, 
                                                      featarray_boruta, 
                                                      train_clarray,  
                                                      cv=10,  
                                                      scoring='balanced_accuracy')

        crossvalidor_meanscore = np.mean(crossvalid_results_original)
        crossvalidor_maxscore = np.max(crossvalid_results_original)



##############################################################
## New search of best HPs
##############################################################

# Load grid of parameters for both classifiers trainings

xgboost_param_grid_random_state = list(config.classifierparam.xgboost.grid_dict.random_state)
xgboost_param_grid_n_estimators = list(config.classifierparam.xgboost.grid_dict.n_estimators)
xgboost_param_grid_learning_rate = list(config.classifierparam.xgboost.grid_dict.learning_rate)
xgboost_param_grid_objective = list(config.classifierparam.xgboost.grid_dict.objective)

lgbm_param_grid_random_state = list(config.classifierparam.light_gbm.grid_dict.random_state)
lgbm_param_grid_n_estimators = list(config.classifierparam.light_gbm.grid_dict.n_estimators)
lgbm_param_grid_learning_rate = list(config.classifierparam.light_gbm.grid_dict.learning_rate)
lgbm_param_grid_objective = list(config.classifierparam.light_gbm.grid_dict.objective)
lgbm_param_grid_num_leaves = list(config.classifierparam.light_gbm.grid_dict.num_leaves)


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


 
# Start the HP search to maximize the mean balanced accuracy over splits
# We only focus on the mean so far

# -- XGBOOST --
if run_xgboost and not run_lgbm:
    if classification_from_allfeatures:
        cv_bestmean_xgboost = 0 
        for paramset in tqdm(ParameterGrid(xgboost_param_grid)):
            xgboost_training.set_params(**paramset)
            # Evaluate the model with cross validation
            crossvalid_results_xgboost = cross_val_score(xgboost_training, 
                                                        feat_representative_slides, 
                                                        train_clarray_refined,  
                                                        cv=10,  
                                                        scoring='balanced_accuracy')

            crossvalid_meanscore_xgboost = np.mean(crossvalid_results_xgboost)
            if crossvalid_meanscore_xgboost > cv_bestmean_xgboost:
                cv_bestmean_xgboost = crossvalid_meanscore_xgboost 
                bestmeanset_xgboost = paramset
                cv_bestmean_scorevect_xgboost = crossvalid_results_xgboost
        print('\n\n ** xgboost (all features) **')
        print('The best mean average accuracy is:',cv_bestmean_xgboost)
        print('Corresponding set of parameters for xgboost on all features is:',
               bestmeanset_xgboost)
        print('Edit the configuration file consequently')            

    else:
        cv_bestmean_xgboost = 0 
        for paramset in tqdm(ParameterGrid(xgboost_param_grid)):
            xgboost_training.set_params(**paramset)
            # Evaluate the model with cross validation
            crossvalid_results_xgboost = cross_val_score(xgboost_training, 
                                                        feat_representative_slides_boruta, 
                                                        train_clarray_refined,  
                                                        cv=10,  
                                                        scoring='balanced_accuracy')

            crossvalid_meanscore_xgboost = np.mean(crossvalid_results_xgboost)
            if crossvalid_meanscore_xgboost > cv_bestmean_xgboost:
                cv_bestmean_xgboost = crossvalid_meanscore_xgboost 
                bestmeanset_xgboost = paramset
                cv_bestmean_scorevect_xgboost = crossvalid_results_xgboost
        print('\n\n ** xgboost (selected features) **')
        print('The best mean average accuracy is:',cv_bestmean_xgboost)
        print('Corresponding set of parameters for xgboost on selected features is:',
               bestmeanset_xgboost)
        print('Edit the configuration file consequently')            



# -- LGBM --
elif run_lgbm and not run_xgboost:
    if classification_from_allfeatures:
        cv_bestmean_lgbm = 0 
        for paramset in tqdm(ParameterGrid(lgbm_param_grid)):
            lgbm_training.set_params(**paramset)
            # Evaluate the model with cross validation
            crossvalid_results_lgbm = cross_val_score(lgbm_training, 
                                                       feat_representative_slides, 
                                                       train_clarray_refined,  
                                                       cv=10,  
                                                       scoring='balanced_accuracy')

            crossvalid_meanscore_lgbm = np.mean(crossvalid_results_lgbm)
            if crossvalid_meanscore_lgbm > cv_bestmean_lgbm:
                cv_bestmean_lgbm = crossvalid_meanscore_lgbm 
                bestmeanset_lgbm = paramset
                cv_bestmean_scorevect_lgbm = crossvalid_results_lgbm
        print('\n\n ** lgbm (all features) **')
        print('The best mean average accuracy is:',cv_bestmean_lgbm)
        print('Corresponding set of parameters for xgboost on all features is:',
               bestmeanset_lgbm)
        print('Edit the configuration file consequently')     

    else:
        cv_bestmean_lgbm = 0 
        for paramset in tqdm(ParameterGrid(lgbm_param_grid)):
            lgbm_training.set_params(**paramset)
            # Evaluate the model with cross validation
            crossvalid_results_lgbm = cross_val_score(lgbm_training, 
                                                       feat_representative_slides_boruta, 
                                                       train_clarray_refined,  
                                                       cv=10,  
                                                       scoring='balanced_accuracy')

            crossvalid_meanscore_lgbm = np.mean(crossvalid_results_lgbm)
            if crossvalid_meanscore_lgbm > cv_bestmean_lgbm:
                cv_bestmean_lgbm = crossvalid_meanscore_lgbm 
                bestmeanset_lgbm = paramset
                cv_bestmean_scorevect_lgbm = crossvalid_results_lgbm
        print('\n\n ** lgbm (selected features) **')
        print('The best mean average accuracy is:',cv_bestmean_lgbm)
        print('Corresponding set of parameters for xgboost on selected features is:',
               bestmeanset_lgbm)
        print('Edit the configuration file consequently')    

# dev ink for dev
devink = 0