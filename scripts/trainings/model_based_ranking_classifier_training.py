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



# To reproduce the results of fig. opf the paper

# Important to notice that with bolean search_bestsplit in config files
# set as True, the code will not produce and save the same results as expected

# The Hyperparameter set in the config also needs to be updates

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
patientid_csv = confighm.paths.files.patientid_csv
patientid_avail = confighm.parameters.bool.patientid_avail
nbr_keptfeat = confighm.parameters.int.nbr_keptfeat

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_eval_folder = config.paths.folders.classification_eval_folder
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
search_bestsplit = config.parameters.bool.search_bestsplit
perm_bestsplit = config.names.permutation_idx.perm_bestsplit
perm_cvmean = config.names.permutation_idx.perm_cvmean

xgboost_random_state = config.classifierparam.xgboost.random_state
xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
xgboost_lr = config.classifierparam.xgboost.learning_rate
xgboost_objective = config.classifierparam.xgboost.objective

lgbm_random_state = config.classifierparam.light_gbm.random_state
lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
lgbm_lr = config.classifierparam.light_gbm.learning_rate
lgbm_objective = config.classifierparam.light_gbm.objective
lgbm_numleaves = config.classifierparam.light_gbm.num_leaves

saveclassifier_xgboost = config.parameters.bool.saving_classifiers.xgboost
saveclassifier_lgbm = config.parameters.bool.saving_classifiers.light_gbm 



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
print('Number of patient is:',num_unique_elements)



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




#!!!!!!!!!!!!!!!!!!!!!DEV
# TRY TO HARD CODE WITHOUT THE LOOP TO SEE IF IT IT BETTER

# XGBOOST - STILL TO UPDATE WITHOUT BOOLEANS NOT TO HAVE A TOO LONG CODE (dev)

# list_proba_predictions = []


# X_train_1 = splits_nested_list[0][0]
# y_train_1 = splits_nested_list[0][1]
# X_test_1 = splits_nested_list[0][2]
# y_test_1 = splits_nested_list[0][3]
# xgboost_slide_ranking_1 = xgboost
# xgboost_slide_ranking_1 = xgboost_slide_ranking_1.fit(X_train_1, 
#                                                       y_train_1, 
#                                                       eval_set=[(X_test_1, y_test_1)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_1.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_1.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_2 = splits_nested_list[1][0]
# y_train_2 = splits_nested_list[1][1]
# X_test_2 = splits_nested_list[1][2]
# y_test_2 = splits_nested_list[1][3]
# xgboost_slide_ranking_2 = xgboost
# xgboost_slide_ranking_2 = xgboost_slide_ranking_2.fit(X_train_2, 
#                                                       y_train_2, 
#                                                       eval_set=[(X_test_2, y_test_2)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_2.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_2.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_3 = splits_nested_list[2][0]
# y_train_3 = splits_nested_list[2][1]
# X_test_3 = splits_nested_list[2][2]
# y_test_3 = splits_nested_list[2][3]
# xgboost_slide_ranking_3 = xgboost
# xgboost_slide_ranking_3 = xgboost_slide_ranking_3.fit(X_train_3, 
#                                                       y_train_3, 
#                                                       eval_set=[(X_test_3, y_test_3)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_3.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_3.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_4 = splits_nested_list[3][0]
# y_train_4 = splits_nested_list[3][1]
# X_test_4 = splits_nested_list[3][2]
# y_test_4 = splits_nested_list[3][3]
# xgboost_slide_ranking_4 = xgboost
# xgboost_slide_ranking_4 = xgboost_slide_ranking_4.fit(X_train_4, 
#                                                       y_train_4, 
#                                                       eval_set=[(X_test_4, y_test_4)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_4.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_4.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_5 = splits_nested_list[4][0]
# y_train_5 = splits_nested_list[4][1]
# X_test_5 = splits_nested_list[4][2]
# y_test_5 = splits_nested_list[4][3]
# xgboost_slide_ranking_5 = xgboost
# xgboost_slide_ranking_5 = xgboost_slide_ranking_5.fit(X_train_5, 
#                                                       y_train_5, 
#                                                       eval_set=[(X_test_5, y_test_5)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_5.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_5.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_6 = splits_nested_list[5][0]
# y_train_6 = splits_nested_list[5][1]
# X_test_6 = splits_nested_list[5][2]
# y_test_6 = splits_nested_list[5][3]
# xgboost_slide_ranking_6 = xgboost
# xgboost_slide_ranking_6 = xgboost_slide_ranking_6.fit(X_train_6, 
#                                                       y_train_6, 
#                                                       eval_set=[(X_test_6, y_test_6)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_6.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_6.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_7 = splits_nested_list[6][0]
# y_train_7 = splits_nested_list[6][1]
# X_test_7 = splits_nested_list[6][2]
# y_test_7 = splits_nested_list[6][3]
# xgboost_slide_ranking_7 = xgboost
# xgboost_slide_ranking_7 = xgboost_slide_ranking_7.fit(X_train_7, 
#                                                       y_train_7, 
#                                                       eval_set=[(X_test_7, y_test_7)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_7.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_7.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_8 = splits_nested_list[7][0]
# y_train_8 = splits_nested_list[7][1]
# X_test_8 = splits_nested_list[7][2]
# y_test_8 = splits_nested_list[7][3]
# xgboost_slide_ranking_8 = xgboost
# xgboost_slide_ranking_8 = xgboost_slide_ranking_8.fit(X_train_8, 
#                                                       y_train_8, 
#                                                       eval_set=[(X_test_8, y_test_8)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_8.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_8.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_9 = splits_nested_list[8][0]
# y_train_9 = splits_nested_list[8][1]
# X_test_9 = splits_nested_list[8][2]
# y_test_9 = splits_nested_list[8][3]
# xgboost_slide_ranking_9 = xgboost
# xgboost_slide_ranking_9 = xgboost_slide_ranking_9.fit(X_train_9, 
#                                                       y_train_9, 
#                                                       eval_set=[(X_test_9, y_test_9)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_9.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_9.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))


# X_train_10 = splits_nested_list[9][0]
# y_train_10 = splits_nested_list[9][1]
# X_test_10 = splits_nested_list[9][2]
# y_test_10 = splits_nested_list[9][3]
# xgboost_slide_ranking_10 = xgboost
# xgboost_slide_ranking_10 = xgboost_slide_ranking_10.fit(X_train_10, 
#                                                       y_train_10, 
#                                                       eval_set=[(X_test_10, y_test_10)])
# if classification_from_allfeatures:
#     proba_predictions = xgboost_slide_ranking_10.predict_proba(train_featarray)
# else:
#     proba_predictions = xgboost_slide_ranking_10.predict_proba(featarray_boruta)
# # We keep only he highest of the 2 probabilities 
# highest_proba_prediction = np.max(proba_predictions, axis=1)
# list_proba_predictions.append((highest_proba_prediction))

#!!!!!!!!!!!!!!!!!!!!!DEV


#!!!!!!!!!!!!!!!!!!!!!DEV
# TRY TO HARD CODE WITHOUT THE LOOP TO SEE IF IT IT BETTER

# LGBM - STILL TO UPDATE WITHOUT BOOLEANS NOT TO HAVE A TOO LONG CODE (dev)

list_proba_predictions = []


X_train_1 = splits_nested_list[0][0]
y_train_1 = splits_nested_list[0][1]
X_test_1 = splits_nested_list[0][2]
y_test_1 = splits_nested_list[0][3]
lgbm_slide_ranking_1 = lightgbm
lgbm_slide_ranking_1 = lgbm_slide_ranking_1.fit(X_train_1, 
                                                y_train_1, 
                                                eval_set=[(X_test_1, y_test_1)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_1.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_1.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_2 = splits_nested_list[1][0]
y_train_2 = splits_nested_list[1][1]
X_test_2 = splits_nested_list[1][2]
y_test_2 = splits_nested_list[1][3]
lgbm_slide_ranking_2 = lightgbm
lgbm_slide_ranking_2 = lgbm_slide_ranking_2.fit(X_train_2, 
                                                y_train_2, 
                                                eval_set=[(X_test_2, y_test_2)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_2.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_2.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_3 = splits_nested_list[2][0]
y_train_3 = splits_nested_list[2][1]
X_test_3 = splits_nested_list[2][2]
y_test_3 = splits_nested_list[2][3]
lgbm_slide_ranking_3 = lightgbm
lgbm_slide_ranking_3 = lgbm_slide_ranking_3.fit(X_train_3, 
                                                y_train_3, 
                                                eval_set=[(X_test_3, y_test_3)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_3.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_3.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_4 = splits_nested_list[3][0]
y_train_4 = splits_nested_list[3][1]
X_test_4 = splits_nested_list[3][2]
y_test_4 = splits_nested_list[3][3]
lgbm_slide_ranking_4 = lightgbm
lgbm_slide_ranking_4 = lgbm_slide_ranking_4.fit(X_train_4, 
                                                y_train_4, 
                                                eval_set=[(X_test_4, y_test_4)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_4.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_4.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_5 = splits_nested_list[4][0]
y_train_5 = splits_nested_list[4][1]
X_test_5 = splits_nested_list[4][2]
y_test_5 = splits_nested_list[4][3]
lgbm_slide_ranking_5 = lightgbm
lgbm_slide_ranking_5 = lgbm_slide_ranking_5.fit(X_train_5, 
                                                y_train_5, 
                                                eval_set=[(X_test_5, y_test_5)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_5.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_5.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))

X_train_6 = splits_nested_list[5][0]
y_train_6 = splits_nested_list[5][1]
X_test_6 = splits_nested_list[5][2]
y_test_6 = splits_nested_list[5][3]
lgbm_slide_ranking_6 = lightgbm
lgbm_slide_ranking_6 = lgbm_slide_ranking_6.fit(X_train_6, 
                                                      y_train_6, 
                                                      eval_set=[(X_test_6, y_test_6)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_6.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_6.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_7 = splits_nested_list[6][0]
y_train_7 = splits_nested_list[6][1]
X_test_7 = splits_nested_list[6][2]
y_test_7 = splits_nested_list[6][3]
lgbm_slide_ranking_7 = lightgbm
lgbm_slide_ranking_7 = lgbm_slide_ranking_7.fit(X_train_7, 
                                                      y_train_7, 
                                                      eval_set=[(X_test_7, y_test_7)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_7.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_7.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_8 = splits_nested_list[7][0]
y_train_8 = splits_nested_list[7][1]
X_test_8 = splits_nested_list[7][2]
y_test_8 = splits_nested_list[7][3]
lgbm_slide_ranking_8 = lightgbm
lgbm_slide_ranking_8 = lgbm_slide_ranking_8.fit(X_train_8, 
                                                      y_train_8, 
                                                      eval_set=[(X_test_8, y_test_8)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_8.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_8.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_9 = splits_nested_list[8][0]
y_train_9 = splits_nested_list[8][1]
X_test_9 = splits_nested_list[8][2]
y_test_9 = splits_nested_list[8][3]
lgbm_slide_ranking_9 = lightgbm
lgbm_slide_ranking_9 = lgbm_slide_ranking_9.fit(X_train_9, 
                                                      y_train_9, 
                                                      eval_set=[(X_test_9, y_test_9)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_9.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_9.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


X_train_10 = splits_nested_list[9][0]
y_train_10 = splits_nested_list[9][1]
X_test_10 = splits_nested_list[9][2]
y_test_10 = splits_nested_list[9][3]
lgbm_slide_ranking_10 = lightgbm
lgbm_slide_ranking_10 = lgbm_slide_ranking_10.fit(X_train_10, 
                                                      y_train_10, 
                                                      eval_set=[(X_test_10, y_test_10)])
if classification_from_allfeatures:
    proba_predictions = lgbm_slide_ranking_10.predict_proba(train_featarray)
else:
    proba_predictions = lgbm_slide_ranking_10.predict_proba(featarray_boruta)
# We keep only he highest of the 2 probabilities 
highest_proba_prediction = np.max(proba_predictions, axis=1)
list_proba_predictions.append((highest_proba_prediction))


#!!!!!!!!!!!!!!!!!!!!!DEV


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
    devstop = 0 

# FOR DEV
idx_most_representative_slide_per_patient = [val - 1 for val 
                                             in idx_most_representative_slide_per_patient]


# Save ID of the most representative slides! (becareful to permute again) 
# We need to permute again to retreive original order
idx_most_representative_slides_export = [len(permutation_index) - index for index 
                                         in idx_most_representative_slide_per_patient]
idx_most_representative_slides_export = np.asarray(idx_most_representative_slides_export)                         
np.save(pathtomain + 'most_representative_slides_lgbm_idx.npy', idx_most_representative_slides_export)
# np.save(pathtomain + 'most_representative_slides_xgboost_idx.npy', idx_most_representative_slides_export) 

# Generate new feature matrix and new classification array
# Generate classification array of only representative slides
train_clarray_refined = train_clarray[idx_most_representative_slide_per_patient]


# Generate the features of representatative slides with selected features
feat_representative_slides = train_featarray[idx_most_representative_slide_per_patient,:]
# Generate the features of representatative slides with selected features
feat_representative_slides_boruta = featarray_boruta[idx_most_representative_slide_per_patient,:]








### SEPERATE THIS PART IN A NEAR FUTURE


##############################################################
## Cross validation of classifiers with selected instances
##############################################################

# I have now  to either do the learning here with the kept sample 

# Need another instance of the classifier
lgbm_training = lightgbm
xgboost_training = xgboost

### Evaluate with cross validation for xgboost
# if classification_from_allfeatures:
#     # Evaluate with cross validation for lgbm with selected features (representative slides)
#     crossvalid_results_refined = cross_val_score(xgboost_training, 
#                                                  feat_representative_slides, 
#                                                  train_clarray_refined,  
#                                                  cv=10,  
#                                                  scoring='balanced_accuracy')

#     crossvalidref_meanscore = np.mean(crossvalid_results_refined)
#     crossvalidref_maxscore = np.max(crossvalid_results_refined)

#     # Evaluate with cross validation for lgbm with selected features (all slides)
#     crossvalid_results_original = cross_val_score(xgboost_training, 
#                                                   train_featarray, 
#                                                   train_clarray,  
#                                                   cv=10,  
#                                                   scoring='balanced_accuracy')

#     crossvalidor_meanscore = np.mean(crossvalid_results_original)
#     crossvalidor_maxscore = np.max(crossvalid_results_original)



# else:
#     # Evaluate with cross validation for lgbm with selected features (representative slides)
#     crossvalid_results_refined = cross_val_score(xgboost_training, 
#                                                  feat_representative_slides_boruta, 
#                                                  train_clarray_refined,  
#                                                  cv=10,  
#                                                  scoring='balanced_accuracy')

#     crossvalidref_meanscore = np.mean(crossvalid_results_refined)
#     crossvalidref_maxscore = np.max(crossvalid_results_refined)

#     # Evaluate with cross validation for lgbm with selected features (all slides)
#     crossvalid_results_original = cross_val_score(xgboost_training, 
#                                                   featarray_boruta, 
#                                                   train_clarray,  
#                                                   cv=10,  
#                                                   scoring='balanced_accuracy')

#     crossvalidor_meanscore = np.mean(crossvalid_results_original)
#     crossvalidor_maxscore = np.max(crossvalid_results_original)


### Evaluate with cross validation for lgbm
if classification_from_allfeatures:
    # Evaluate with cross validation for lgbm with selected features (representative slides)
    crossvalid_results_refined = cross_val_score(lgbm_training, 
                                                 feat_representative_slides, 
                                                 train_clarray_refined,  
                                                 cv=10,  
                                                 scoring='balanced_accuracy')

    crossvalidref_meanscore = np.mean(crossvalid_results_refined)
    crossvalidref_maxscore = np.max(crossvalid_results_refined)

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

    crossvalidref_meanscore = np.mean(crossvalid_results_refined)
    crossvalidref_maxscore = np.max(crossvalid_results_refined)

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

### With xgboost 
# if classification_from_allfeatures:
#     cv_bestmean_xgboost = 0 
#     for paramset in tqdm(ParameterGrid(xgboost_param_grid)):
#         xgboost_training.set_params(**paramset)
#         # Evaluate the model with cross validation
#         crossvalid_results_xgboost = cross_val_score(xgboost_training, 
#                                                     feat_representative_slides, 
#                                                     train_clarray_refined,  
#                                                     cv=10,  
#                                                     scoring='balanced_accuracy')

#         crossvalid_meanscore_xgboost = np.mean(crossvalid_results_xgboost)
#         if crossvalid_meanscore_xgboost > cv_bestmean_xgboost:
#             cv_bestmean_xgboost = crossvalid_meanscore_xgboost 
#             bestmeanset_xgboost = paramset
#             cv_bestmean_scorevect_xgboost = crossvalid_results_xgboost

# else:
#     cv_bestmean_xgboost = 0 
#     for paramset in tqdm(ParameterGrid(xgboost_param_grid)):
#         xgboost_training.set_params(**paramset)
#         # Evaluate the model with cross validation
#         crossvalid_results_xgboost = cross_val_score(xgboost_training, 
#                                                     feat_representative_slides_boruta, 
#                                                     train_clarray_refined,  
#                                                     cv=10,  
#                                                     scoring='balanced_accuracy')

#         crossvalid_meanscore_xgboost = np.mean(crossvalid_results_xgboost)
#         if crossvalid_meanscore_xgboost > cv_bestmean_xgboost:
#             cv_bestmean_xgboost = crossvalid_meanscore_xgboost 
#             bestmeanset_xgboost = paramset
#             cv_bestmean_scorevect_xgboost = crossvalid_results_xgboost

### With lgbm
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


# No need of printing as we debug uing pass
pass