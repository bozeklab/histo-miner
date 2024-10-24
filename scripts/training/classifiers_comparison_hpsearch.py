#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import numpy as np
import yaml
import xgboost 
import lightgbm
from tqdm import tqdm
from attrdictionary import AttrDict as attributedict
from sklearn import linear_model, ensemble
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedKFold

import src.histo_miner.utils.misc as utils_misc




# Create a version of the code where the patient IDs are not necessary!



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


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
save_evaluations = config.parameters.bool.save_evaluations
nbr_of_splits = config.parameters.int.nbr_of_splits

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


pathfeatselect = pathtofolder 
ext = '.npy'

path_featarray = pathfeatselect + 'perwsi_featarray' + ext
path_clarray = pathfeatselect + 'perwsi_clarray' + ext
path_patientids_array = pathfeatselect + 'patientids' + ext

train_featarray = np.load(path_featarray)
train_clarray = np.load(path_clarray)
train_clarray = np.transpose(train_clarray)
print('Feature feature arrays and class arrays loaded')


path_save_results = classification_eval_folder + 'classifiers_comparison_hpsearch2.txt'



##############################################################
## Hyperparemeter Search and Cross-Validation of Classifiers
##############################################################


# Define the classifiers
##### XGBOOST
xgboost = xgboost.XGBClassifier(verbosity=0)

##### LIGHT GBM setting
lightgbm = lightgbm.LGBMClassifier(verbosity=-1)


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


print('Start Classifiers trainings...')
#TEMP
print('Only searching with xgboost')
####

### Create a new permutation and save it
permutation_index = np.random.permutation(train_clarray.size)
# np.save(pathfeatselect + 'random_permutation_index_new2.npy', permutation_index)

# ### Load permutation index not to have 0 and 1s not mixed
# permutation_index = np.load(pathfeatselect + 
#                             '/bestperm/' +
#                             'random_permutation_index_11_28_xgboost_bestmean.npy')

### Shuffle classification arrays using the permutation index
train_clarray = train_clarray[permutation_index]


### Create Stratified Group  instance for the cross validation 
stratkf = StratifiedKFold(n_splits=nbr_of_splits, shuffle=False)


#### Classification training with all features kept 

# Use all the feature (no selection) as input
genfeatarray = np.transpose(train_featarray)

# Shuffle feature arrays using the permutation index 
genfeatarray = genfeatarray[permutation_index,:]



#### XGBOOST
xgboostvanilla = xgboost
# use Grid Search to find the best set of HPs 
# (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
cv_bestmean = 0 #cv stands for cross-validation 
cv_bestsplit = 0
for paramset in tqdm(ParameterGrid(xgboost_param_grid)):
    xgboostvanilla.set_params(**paramset)
    # Evaluate the model with cross validation
    crossvalid_results = cross_val_score(xgboostvanilla, 
                                         genfeatarray, 
                                         train_clarray,  
                                         cv=stratkf,  
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


print('\n\n ** xgboost_vanilla **')
print('The best split average accuracy is:',cv_bestsplit)  
print('Corresponding set of parameters for xgboost_vanilla_bestmaxset is:',
        xgboost_vanilla_bestmaxset)
print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
print('The best mean average accuracy is:',cv_bestmean)
print('Corresponding set of parameters for xgboost_vanilla_bestmeanset is:',
        xgboost_vanilla_bestmeanset)
print('Corresponding scores for all splits are:', cv_bestmean_scorevect)


#### LIGHT GBM
# lgbm_traindata_vanilla = lightgbm.Dataset(genfeatarray, label=train_clarray) 
# #lgbm_valdata_vanilla = lgbm_traindata_vanilla.create_valid()
# lgbm_vanilla = lightgbm.train(lightgbm_paramters, 
#                               lgbm_traindata_vanilla, 
#                               lgbm_n_estimators)
# lightgbmvanilla = lightgbm
# # lgbm_vanilla = lightgbmvanilla.fit(genfeatarray, train_clarray)
# # use Grid Search to find the best set of HPs 
# # (avoid GridSearchCV cause it is also doing not necessarily wanted cross validation)
# cv_bestmean = 0 #cv stands for cross-validation 
# cv_bestsplit = 0 
# for paramset in tqdm(ParameterGrid(lgbm_param_grid)):
#     lightgbmvanilla.set_params(**paramset)
#     # Evaluate the model with cross validation
#     crossvalid_results = cross_val_score(lightgbmvanilla, 
#                                          genfeatarray, 
#                                          train_clarray,  
#                                          cv=stratkf, 
#                                          scoring='balanced_accuracy')
#     crossvalid_meanscore = np.mean(crossvalid_results)
#     crossvalid_maxscore = np.max(crossvalid_results)
#     # Keep best trainings score and parameters set
#     if crossvalid_maxscore > cv_bestsplit:
#         cv_bestsplit = crossvalid_maxscore
#         lgbm_vanilla_bestmaxset = paramset
#         cv_bestmax_scorevect = crossvalid_results
#     if crossvalid_meanscore > cv_bestmean:
#         cv_bestmean = crossvalid_meanscore 
#         lgbm_vanilla_bestmeanset = paramset
#         cv_bestmean_scorevect = crossvalid_results   

# print('\n\n ** lightgbm_vanilla **')
# print('The best split average accuracy is:',cv_bestsplit)  
# print('Corresponding set of parameters for lgbm_vanilla_bestmaxset is:',
#         lgbm_vanilla_bestmaxset)
# print('Corresponding scores for all splits are:', cv_bestmax_scorevect)
# print('The best mean average accuracy is:',cv_bestmean)
# print('Corresponding set of parameters for lgbm_vanilla_bestmeanset is:',
#         lgbm_vanilla_bestmeanset)
# print('Corresponding scores for all splits are:', cv_bestmean_scorevect)



##############################################################
## Optional: save evaluations into a file
##############################################################

if save_evaluations:
    with open(path_save_results, 'w') as file:
        file.write('\n\n ** xgboost_vanilla **')
        file.write('The best split average accuracy is:' + str(cv_bestsplit))  
        file.write('Corresponding set of parameters for xgboost_vanilla_bestmaxset is:' 
            + str(xgboost_vanilla_bestmaxset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmax_scorevect))
        file.write('The best mean average accuracy is:' + str(cv_bestmean))
        file.write('Corresponding set of parameters for xgboost_vanilla_bestmeanset is:' 
            + str(xgboost_vanilla_bestmeanset))
        file.write('Corresponding scores for all splits are:' + str( cv_bestmean_scorevect))
        # file.write('\n\n ** lightgbm_vanilla **')
        # file.write('The best split average accuracy is:' + str(cv_bestsplit))
        # file.write('Corresponding set of parameters for lgbm_vanilla_bestmaxset is:' 
        #     + str(lgbm_vanilla_bestmaxset))
        # file.write('Corresponding scores for all splits are:' + str( cv_bestmax_scorevect))
        # file.write('The best mean average accuracy is:' + str(cv_bestmean))
        # file.write('Corresponding set of parameters for lgbm_vanilla_bestmeanset is:' 
        #     + str(lgbm_vanilla_bestmeanset))
        # file.write('Corresponding scores for all splits are:' + str(cv_bestmean_scorevect))

    print('Evaluations saved at location:', path_save_results)
