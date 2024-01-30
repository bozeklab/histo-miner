#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import os

import numpy as np
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn import linear_model, ensemble
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold

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
pathtofolder = confighm.paths.folders.feature_selection_main
classification_eval_folder = confighm.paths.folders.classification_evaluation
patientid_avail = confighm.parameters.bool.patientid_avail
wsi_selection = confighm.parameters.bool.wsi_selection

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/classification_training.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
search_bestsplit = config.parameters.bool.search_bestsplit
perm_bestsplit = config.names.permutation_idx.perm_bestsplit
perm_cvmean = config.names.permutation_idx.perm_cvmean

ridge_random_state = config.classifierparam.ridge.random_state
ridge_alpha = config.classifierparam.ridge.alpha

lregression_random_state = config.classifierparam.logistic_regression.random_state
lregression_penalty = config.classifierparam.logistic_regression.penalty
lregression_solver = config.classifierparam.logistic_regression.solver
lregression_multi_class = config.classifierparam.logistic_regression.multi_class
lregression_class_weight = config.classifierparam.logistic_regression.class_weight

forest_random_state = config.classifierparam.random_forest.random_state
forest_n_estimators = config.classifierparam.random_forest.n_estimators
forest_class_weight = config.classifierparam.random_forest.class_weight

xgboost_random_state = config.classifierparam.xgboost.random_state
xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
xgboost_lr = config.classifierparam.xgboost.learning_rate
xgboost_objective = config.classifierparam.xgboost.objective

lgbm_random_state = config.classifierparam.light_gbm.random_state
lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
lgbm_lr = config.classifierparam.light_gbm.learning_rate
lgbm_objective = config.classifierparam.light_gbm.objective
lgbm_numleaves = config.classifierparam.light_gbm.num_leaves



############################################################
## Load feature selection numpy files
############################################################


pathfeatselect = pathtofolder 
ext = '.npy'

print('Load feature selection numpy files...')

# Load feature selection numpy files

# COULD ADD RAISE ERRROR IF IT IS NOT FIND!
# Each time we check if the file exist because all selections are not forced to run
path_selfeat_mrmr_idx = pathfeatselect + 'selfeat_mrmr_idx' + ext
if os.path.exists(path_selfeat_mrmr_idx):
     selfeat_mrmr_idx = np.load(path_selfeat_mrmr_idx, allow_pickle=True)
path_selfeat_mannwhitneyu_idx = pathfeatselect + 'selfeat_mannwhitneyu_idx' + ext
if os.path.exists(path_selfeat_mannwhitneyu_idx):
    selfeat_mannwhitneyu_idx = np.load(path_selfeat_mannwhitneyu_idx, allow_pickle=True)
print('Loading feature selected indexes done.')



################################################################
## Load feat array, class arrays and IDs arrays (if applicable)
################################################################

#This is to check but should be fine
path_featarray = pathfeatselect + 'repslidesx_featarray' + ext
path_clarray = pathfeatselect + 'repslidesx_clarray' + ext

train_featarray = np.load(path_featarray)
train_clarray = np.load(path_clarray)
train_clarray = np.transpose(train_clarray)
if not wsi_selection:
    if patientid_avail:
        path_patientids_array = pathfeatselect + 'patientids' + ext
        patientids_load = np.load(path_patientids_array, allow_pickle=True)
        patientids_list = list(patientids_load)
        patientids_convert = utils_misc.convert_names_to_integers(patientids_list)
        patientids = np.asarray(patientids_convert)



##############################################################
## Traininig Classifiers
##############################################################


# Define the classifiers
# More information here: #https://scikit-learn.org/stable/modules/linear_model.html
##### RIDGE CLASSIFIER
ridge = linear_model.RidgeClassifier(random_state= ridge_random_state,
                                     alpha=ridge_alpha)
##### LOGISTIC REGRESSION
lr = linear_model.LogisticRegression(random_state=lregression_random_state,
                                     penalty=lregression_penalty,
                                     solver=lregression_solver,
                                     multi_class=lregression_multi_class,
                                     class_weight=lregression_class_weight)
##### RANDOM FOREST
forest = ensemble.RandomForestClassifier(random_state= forest_random_state,
                                         n_estimators=forest_n_estimators,
                                         class_weight=forest_class_weight)
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


### Load permutation index not to have 0 and 1s not mixed if the slide are not 
### pre-selected
if not wsi_selection:
    if search_bestsplit:
        permutation_index = np.load(pathfeatselect + perm_bestsplit + '.npy')
    else: 
        permutation_index = np.load(pathfeatselect + perm_cvmean + '.npy')

### Shuffle classification arrays using the permutation index
if not wsi_selection:
    train_clarray = train_clarray[permutation_index]

    if patientid_avail:

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
        patientids_ordered = patientids_ordered[permutation_index]

### Create Stratified Group  instance for the cross validation 
stratgroupkf = StratifiedGroupKFold(n_splits=10, shuffle=False)


# Initialize the lists:
if search_bestsplit:
    xgbbestsplit_aAcc_mrmr = list()
    xgbbestsplit_aAcc_mannwhitneyu = list()
    xgbbestsplit_aAcc_boruta = list() 
    lgbmbestsplit_aAcc_mrmr = list()
    lgbmbestsplit_aAcc_mannwhitneyu = list()
    lgbmbestsplit_aAcc_boruta = list() 
else: 
    xgbmean_aAcc_mrmr = list()
    xgbmean_aAcc_mannwhitneyu = list()
    xgbmean_aAcc_boruta = list()
    lgbmmean_aAcc_mrmr = list()
    lgbmmean_aAcc_mannwhitneyu = list()
    lgbmmean_aAcc_boruta = list()


#### Parse the featarray to the class SelectedFeaturesMatrix 
SelectedFeaturesMatrix = SelectedFeaturesMatrix(train_featarray)


#### Classification training (decreasing number of features left, starting with all)


# Use all the feature (no selection) as input
genfeatarray = np.transpose(train_featarray)

#Shuffle feature arrays using the permutation index
if not wsi_selection:
    genfeatarray = genfeatarray[permutation_index,:]

#### XGBOOST
xgboostvanilla = xgboost
if wsi_selection:
    crossvalid_results = cross_val_score(xgboostvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         cv=10,  
                                         scoring='balanced_accuracy')
else:
    crossvalid_results = cross_val_score(xgboostvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         groups=patientids_ordered,
                                         cv=stratgroupkf,  
                                         scoring='balanced_accuracy')
crossvalid_meanscore = np.mean(crossvalid_results)
crossvalid_maxscore = np.max(crossvalid_results)

# Insert results in the corresponding lists
if search_bestsplit:
    xgbbestsplit_aAcc_mrmr.append(crossvalid_maxscore)
    xgbbestsplit_aAcc_mannwhitneyu.append(crossvalid_maxscore)
    xgbbestsplit_aAcc_boruta.append(crossvalid_maxscore)
else:
    xgbmean_aAcc_mrmr.append(crossvalid_meanscore)
    xgbmean_aAcc_mannwhitneyu.append(crossvalid_meanscore)
    xgbmean_aAcc_boruta.append(crossvalid_meanscore) 

#### LIGHT GBM
lightgbmvanilla = lightgbm
if wsi_selection:
    crossvalid_results = cross_val_score(lightgbmvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         cv=10,  
                                         scoring='balanced_accuracy')
else:
    crossvalid_results = cross_val_score(lightgbmvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         groups=patientids_ordered,
                                         cv=stratgroupkf,  
                                         scoring='balanced_accuracy')
crossvalid_meanscore = np.mean(crossvalid_results)
crossvalid_maxscore = np.max(crossvalid_results)

# Insert results in the corresponding lists
if search_bestsplit:
    lgbmbestsplit_aAcc_mrmr.append(crossvalid_maxscore)
    lgbmbestsplit_aAcc_mannwhitneyu.append(crossvalid_maxscore)
    lgbmbestsplit_aAcc_boruta.append(crossvalid_maxscore)
else:
    lgbmmean_aAcc_mrmr.append(crossvalid_meanscore)
    lgbmmean_aAcc_mannwhitneyu.append(crossvalid_meanscore)
    lgbmmean_aAcc_boruta.append(crossvalid_meanscore) 


for nbr_keptfeat_idx in range(55, 0, -1):
# previous dev:
# kept the selected features but in inverse order (for figures)
# Uncomment only for reproducing secondary figures results
# for nbr_keptfeat in range(56, 1, -1):

    # Kept the selected features
    selfeat_mrmr_idx =  selfeat_mrmr_idx[0:nbr_keptfeat_idx]
    print('\n', selfeat_mrmr_idx)
    selfeat_mannwhitneyu_idx = selfeat_mannwhitneyu_idx[0:nbr_keptfeat_idx]
    print(selfeat_mannwhitneyu_idx)

    # previous dev:
    # kept the selected features but in inverse order (for figures)
    # Uncomment only for reproducing secondary figures results
    # selfeat_mrmr_idx =  selfeat_mrmr_idx[1:nbr_keptfeat]
    # print('\n', selfeat_mrmr_idx)
    # selfeat_mannwhitneyu_idx = selfeat_mannwhitneyu_idx[1:nbr_keptfeat]
    # print(selfeat_mannwhitneyu_idx)


    #### Recall numberr of efatures kept:
    print('{} features kept.'.format(nbr_keptfeat_idx))


    #### Classification training with the features kept by mrmr
    if os.path.exists(path_selfeat_mrmr_idx):
        # Generate the matrix with selected feature for mrmr
        featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr( selfeat_mrmr_idx)

        #Shuffle feature arrays using the permutation index 
        if not wsi_selection:
            featarray_mrmr = featarray_mrmr[permutation_index,:]

        ##### XGBOOST
        xgboostmrmr = xgboost
        if wsi_selection:
            crossvalid_results = cross_val_score(xgboostmrmr, 
                                                 featarray_mrmr, 
                                                 train_clarray,  
                                                 cv=10,  
                                                 scoring='balanced_accuracy')
        else:
            crossvalid_results = cross_val_score(xgboostmrmr, 
                                                featarray_mrmr, 
                                                train_clarray,  
                                                groups=patientids_ordered,
                                                cv=stratgroupkf,  
                                                scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        crossvalid_maxscore = np.max(crossvalid_results)
        
        # Insert results in the corresponding lists
        if search_bestsplit:
            xgbbestsplit_aAcc_mrmr.append(crossvalid_maxscore)
        else: 
            xgbmean_aAcc_mrmr.append(crossvalid_meanscore)

        ##### LIGHT GBM
        lightgbmmrmr = lightgbm
        if wsi_selection:
            crossvalid_results = cross_val_score(lightgbmmrmr, 
                                                 featarray_mrmr, 
                                                 train_clarray,  
                                                 cv=10,  
                                                 scoring='balanced_accuracy')
        else:
            crossvalid_results = cross_val_score(lightgbmmrmr, 
                                                 featarray_mrmr, 
                                                 train_clarray,  
                                                 groups=patientids_ordered,
                                                 cv=stratgroupkf,  
                                                 scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        crossvalid_maxscore = np.max(crossvalid_results)

        # Insert results in the corresponding lists
        if search_bestsplit:
            lgbmbestsplit_aAcc_mrmr.append(crossvalid_maxscore)
        else: 
            lgbmmean_aAcc_mrmr.append(crossvalid_meanscore)




    #### Classification training with the features kept by mannwhitneyu

    if os.path.exists(path_selfeat_mannwhitneyu_idx):
        # Generate the matrix with selected feature for mannwhitney
        featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(selfeat_mannwhitneyu_idx)
        
        #Shuffle feature arrays using the permutation index 
        if not wsi_selection:
            featarray_mannwhitney = featarray_mannwhitney[permutation_index,:]

        ##### XGBOOST
        xgboostmannwhitney = xgboost
        if wsi_selection:
            crossvalid_results = cross_val_score(xgboostmannwhitney, 
                                                 featarray_mannwhitney, 
                                                 train_clarray,  
                                                 cv=10,  
                                                 scoring='balanced_accuracy')
        else:
            crossvalid_results = cross_val_score(xgboostmannwhitney, 
                                                 featarray_mannwhitney, 
                                                 train_clarray,  
                                                 groups=patientids_ordered,
                                                 cv=stratgroupkf,  
                                                 scoring='balanced_accuracy')
        crossvalid_meanscore = np.mean(crossvalid_results)
        crossvalid_maxscore = np.max(crossvalid_results)

        # Insert results in the corresponding lists
        if search_bestsplit:
            xgbbestsplit_aAcc_mannwhitneyu.append(crossvalid_maxscore)
        else: 
            xgbmean_aAcc_mannwhitneyu.append(crossvalid_meanscore)

        ##### LIGHT GBM
        lightgbmmannwhitney = lightgbm
        if wsi_selection:
            crossvalid_results = cross_val_score(lightgbmmannwhitney, 
                                                 featarray_mannwhitney, 
                                                 train_clarray,  
                                                 cv=10,  
                                                 scoring='balanced_accuracy')
        else:
            crossvalid_results = cross_val_score(lightgbmmannwhitney, 
                                                 featarray_mannwhitney, 
                                                 train_clarray,  
                                                 groups=patientids_ordered,
                                                 cv=stratgroupkf,  
                                                 scoring='balanced_accuracy')        
        crossvalid_meanscore = np.mean(crossvalid_results)
        crossvalid_maxscore = np.max(crossvalid_results)

        # Insert results in the corresponding lists
        if search_bestsplit:
            lgbmbestsplit_aAcc_mannwhitneyu.append(crossvalid_maxscore)
        else: 
            lgbmmean_aAcc_mannwhitneyu.append(crossvalid_meanscore)


# Conversion to numpy  
if search_bestsplit:
    # print('xgbbestsplit_aAcc_mrmr is:', xgbbestsplit_aAcc_mrmr)
    print('lgbmbestsplit_aAcc_mrmr is:', lgbmbestsplit_aAcc_mrmr)
    # print('xgbbestsplit_aAcc_mannwhitneyu is:', xgbbestsplit_aAcc_mannwhitneyu)
    print('lgbmbestsplit_aAcc_mannwhitneyu is:', lgbmbestsplit_aAcc_mannwhitneyu)
    # xgbbestsplit_aAcc_mrmr = np.asarray(xgbbestsplit_aAcc_mrmr)
    lgbmbestsplit_aAcc_mrmr = np.asarray(lgbmbestsplit_aAcc_mrmr)
    # xgbbestsplit_aAcc_mannwhitneyu = np.asarray(xgbbestsplit_aAcc_mannwhitneyu)
    lgbmbestsplit_aAcc_mannwhitneyu = np.asarray(lgbmbestsplit_aAcc_mannwhitneyu)
else: 
    # print('xgbmean_aAcc_mrmr is', xgbmean_aAcc_mrmr)
    print('lgbmmean_aAcc_mrmr is', lgbmmean_aAcc_mrmr)
    # print('xgbmean_aAcc_mannwhitneyu is', xgbmean_aAcc_mannwhitneyu)
    print('lgbmmean_aAcc_mannwhitneyu is', lgbmmean_aAcc_mannwhitneyu)
    # xgbmean_aAcc_mrmr = np.asarray(xgbmean_aAcc_mrmr)
    lgbmmean_aAcc_mrmr = np.asarray(lgbmmean_aAcc_mrmr)
    # xgbmean_aAcc_mannwhitneyu = np.asarray(xgbmean_aAcc_mannwhitneyu)
    lgbmmean_aAcc_mannwhitneyu = np.asarray(lgbmmean_aAcc_mannwhitneyu)

# Saving
save_results_path = classification_eval_folder + 'TestofKs/'
save_ext = '.npy'
if not os.path.exists(save_results_path):
    os.mkdir(save_results_path)
if search_bestsplit:
    # np.save(save_results_path + 'xgbbestsplit_aAcc_mrmr' + save_ext,
             # xgbbestsplit_aAcc_mrmr)
    np.save(save_results_path + 'lgbmbestsplit_aAcc_mrmr' + save_ext,
        lgbmbestsplit_aAcc_mrmr)
    # np.save(save_results_path + 'xgbbestsplit_aAcc_mannwhitneyu' + save_ext,
        # xgbbestsplit_aAcc_mannwhitneyu)
    np.save(save_results_path + 'lgbmbestsplit_aAcc_mannwhitneyu' + save_ext,
        lgbmbestsplit_aAcc_mannwhitneyu)
else:
    # np.save(save_results_path + 'xgbmean_aAcc_mrmr' + save_ext,
             # xgbmean_aAcc_mrmr)
    np.save(save_results_path + 'lgbmmean_aAcc_mrmr' + save_ext,
        lgbmmean_aAcc_mrmr)
    # np.save(save_results_path + 'xgbmean_aAcc_mannwhitneyu' + save_ext,
        # xgbmean_aAcc_mannwhitneyu)
    np.save(save_results_path + 'lgbmmean_aAcc_mannwhitneyu' + save_ext,
        lgbmmean_aAcc_mannwhitneyu) 



### BORUTA

selfeat_boruta_folder = pathfeatselect + 'all_borutas/'

if os.path.exists(selfeat_boruta_folder):
    print('Check if depth_boruta list match the different files.')

    depth_boruta = [2, 4, 6, 20]
    # depth_boruta = [6, 8, 10, 12, 14, 16, 18, 20]
    nbr_keptfeat_list = [56]

    for depth in depth_boruta:

        # Load the correctly selected features
        pathselfeat_boruta = selfeat_boruta_folder + 'selfeat_boruta_idx_depth' + str(depth) + '.npy'
        selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)

        # Check how many features are selected for this depth
        nbr_keptfeat = len(selfeat_boruta)
        print('nbr_keptfeat:', nbr_keptfeat)
        print('selfeat_boruta', selfeat_boruta)
        nbr_keptfeat_list.append(nbr_keptfeat)

        #### Classification training with the features kept by boruta

        if os.path.exists(pathselfeat_boruta):
            # Generate the matrix with selected feature for boruta
            featarray_boruta = SelectedFeaturesMatrix.boruta_matr(selfeat_boruta)

            #Shuffle feature arrays using the permutation index
            if not wsi_selection: 
                featarray_boruta = featarray_boruta[permutation_index,:]
          
            ##### XGBOOST
            xgboostboruta = xgboost
            if wsi_selection:
                crossvalid_results = cross_val_score(xgboostboruta, 
                                                     featarray_boruta, 
                                                     train_clarray,  
                                                     cv=10,  
                                                     scoring='balanced_accuracy')
            else:
                crossvalid_results = cross_val_score(xgboostboruta, 
                                                     featarray_boruta, 
                                                     train_clarray,  
                                                     groups=patientids_ordered,
                                                     cv=stratgroupkf,  
                                                     scoring='balanced_accuracy')
            crossvalid_meanscore = np.mean(crossvalid_results)
            crossvalid_maxscore = np.max(crossvalid_results)

            # Insert results in the corresponding lists
            if search_bestsplit:
                xgbbestsplit_aAcc_boruta.append(crossvalid_maxscore)
            else: 
                xgbmean_aAcc_boruta.append(crossvalid_meanscore)

            ##### LIGHT GBM
            lightgbmboruta = lightgbm
            if wsi_selection:
                crossvalid_results = cross_val_score(lightgbmboruta, 
                                                     featarray_boruta, 
                                                     train_clarray,  
                                                     cv=10,  
                                                     scoring='balanced_accuracy')
            else:
                crossvalid_results = cross_val_score(lightgbmboruta, 
                                                     featarray_boruta, 
                                                     train_clarray,  
                                                     groups=patientids_ordered,
                                                     cv=stratgroupkf,  
                                                     scoring='balanced_accuracy')
            crossvalid_meanscore = np.mean(crossvalid_results)
            crossvalid_maxscore = np.max(crossvalid_results)

            # Insert results in the corresponding lists
            if search_bestsplit:
                lgbmbestsplit_aAcc_boruta.append(crossvalid_maxscore)
            else: 
                lgbmmean_aAcc_boruta.append(crossvalid_meanscore)


    # Conversion to numpy  
    nbr_keptfeat_list = np.asarray(nbr_keptfeat_list)
    if search_bestsplit:
        # print('xgbbestsplit_aAcc_boruta is:', xgbbestsplit_aAcc_boruta)
        print('lgbmbestsplit_aAcc_boruta is:', lgbmbestsplit_aAcc_boruta)
        # xgbbestsplit_aAcc_boruta = np.asarray(xgbbestsplit_aAcc_boruta)
        lgbmbestsplit_aAcc_boruta = np.asarray(lgbmbestsplit_aAcc_boruta)
    else: 
        # print('xgbmean_aAcc_boruta is', xgbmean_aAcc_boruta)
        print('lgbmmean_aAcc_boruta is', lgbmmean_aAcc_boruta)
        # xgbmean_aAcc_boruta = np.asarray(xgbmean_aAcc_boruta)
        lgbmmean_aAcc_boruta = np.asarray(lgbmmean_aAcc_boruta)

    # Saving
    save_results_path = classification_eval_folder + 'TestofKs/'
    save_ext = '.npy'
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)

    np.save(save_results_path + 'nbr_keptfeat_list' + save_ext ,nbr_keptfeat_list)
    if search_bestsplit:
        # np.save(save_results_path + 'xgbbestsplit_aAcc_boruta' + save_ext,
                 # xgbbestsplit_aAcc_boruta)
        np.save(save_results_path + 'lgbmbestsplit_aAcc_boruta' + save_ext,
            lgbmbestsplit_aAcc_boruta)
    else:
        # np.save(save_results_path + 'xgbmean_aAcc_boruta' + save_ext,
                 # xgbmean_aAcc_boruta)
        np.save(save_results_path + 'lgbmmean_aAcc_boruta' + save_ext,
            lgbmmean_aAcc_boruta)



print('Evaluations  are done.')

