#Lucas Sancéré 

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grandparent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(script_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

import random
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

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open(script_dir + "/../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(config)
featarray_folder = confighm.paths.folders.featarray_folder
classification_eval_folder = confighm.paths.folders.classification_evaluation

eval_folder_name = confighm.names.eval_folder



with open("./../../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
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
pathfeatnames = featarray_folder + 'featnames' + ext

train_featarray = np.load(featarray_folder + featarray_name + ext)
train_clarray = np.load(featarray_folder + classarray_name + ext)
featnames = np.load(pathfeatnames)
featnameslist = list(featnames)




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


### Create Stratified Group to further split the dataset into n_splits 
stratkf = StratifiedKFold(n_splits=nbr_of_splits, shuffle=False)


# Create a list of splits with all features 
splits_nested_list = list()
# Create a list of patient IDs corresponding of the splits:
splits_patientID_list = list()
for i, (train_index, test_index) in enumerate(stratkf.split(train_featarray, 
                                                                 train_clarray 
                                                                 )):
    # Generate training and test data from the indexes
    X_train = train_featarray[train_index]
    X_test = train_featarray[test_index]
    y_train = train_clarray[train_index]
    y_test = train_clarray[test_index]

    splits_nested_list.append([X_train, y_train, X_test, y_test])


# Initialization of parameters
nbr_feat = len(X_train[1])
print('nbr_feat is:',nbr_feat)



##############################################################
##  Classifying with XGBOOST 
##############################################################


if run_xgboost and not run_lgbm:


    balanced_accuracies = {"balanced_accuracies_hclustering":  {"initialization": True}}

    selfeat_hclustering_names_allsplits = [] 
    selfeat_hclustering_id_allsplits = []
    selfeat_hclustering_nbr_allsplits = []


    # Keep features with hierarchical clustering
    print('Selection of features with hierarchical clustering method...')

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

        
        listselfeat_hclustering_index = list()
        listselfeat_hclustering_names = list()
        listselfeat_number = list()

        corrmat =  np.corrcoef(X_train_tr) 
        distance_matrix = 1 - corrmat
        # beacuse of aerage it can end up that the matrix is not symetric, 
        # we can force it to be symetric
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        # symmetry_deviation = np.abs(distance_matrix - distance_matrix.T).max()
        # Perform hierarchical clustering using the 'ward' method
        Z = linkage(squareform(distance_matrix), method='ward')
        # Choose a threshold for the clustering (depends on your data and the dendrogram)
        # we choose different options for max distance 
        for maxdistvalue in np.arange(0.5, 5, 0.5):
	        clusters = fcluster(Z, maxdistvalue, criterion='distance')
	        # Create a dictionary to map cluster number -> feature indices
	        cluster_dict = {}
	        for idx, cluster_num in enumerate(clusters):
	            if cluster_num not in cluster_dict:
	                cluster_dict[cluster_num] = idx  # Store the index instead of the name
	        # Select one representative feature index from each cluster
	        selfeat_hclustering_index = [cluster_dict[cluster_num] for cluster_num in cluster_dict]
	        selfeat_hclustering_names = [featnameslist[index] for index in selfeat_hclustering_index]
	        listselfeat_hclustering_index.append(selfeat_hclustering_index)
	        listselfeat_hclustering_names.append(selfeat_hclustering_names)
	        listselfeat_number.append(len(selfeat_hclustering_names))

        selfeat_hclustering_id_allsplits.append(listselfeat_hclustering_index)
        selfeat_hclustering_names_allsplits.append(listselfeat_hclustering_names)
        selfeat_hclustering_nbr_allsplits.append(listselfeat_number)

        ########## GENERATION OF MATRIX OF SELECTED FEATURES
        # If the class was not initalized, do it. If not, reset attributes if the class instance
        # if i == 0:
        #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
        # else:
        #     selected_features_matrix.reset_attributes(X_train_tr)
        feature_array = X_train

        ### With hierarcical clustering selected features

        balanced_accuracies_hierarcicalclustering = list()

        for idx in range(0, len(listselfeat_number)):

	        if len(listselfeat_hclustering_index[idx]) == 0:   
	        	# balanced_accuracy_hclustering = None   
	            balanced_accuracies_hierarcicalclustering.append(None)           
	        else:
	            featarray_hclustering = feature_array[:, np.transpose(listselfeat_hclustering_index[idx])]

	            xgboost_hclustering_training = xgboost
	            xgboost_hclustering_training = xgboost_hclustering_training.fit(featarray_hclustering, 
	                                                                  y_train)
	            # Predictions on the test split
	            y_pred_hclustering = xgboost_hclustering_training.predict(
	                X_test[:, np.transpose(np.transpose(listselfeat_hclustering_index[idx]))]
	                )

	            # Calculate balanced accuracy for the current split
	            balanced_accuracy_hclustering = balanced_accuracy_score(y_test,
	                                                               y_pred_hclustering)

	            # add it to the list
	            balanced_accuracies_hierarcicalclustering.append(balanced_accuracy_hclustering)


        ### Store results 
        # store all resutls in the main dict knowing it will be repeated 10times
        # maybe create a nested dict, split1, split2 and so on!!
        currentsplit =  f"split_{i}"

        # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
        balanced_accuracies['balanced_accuracies_hclustering'][currentsplit] = balanced_accuracies_hierarcicalclustering

        # update changing size of kept feature for hclustering
        # length_selfeathclustering[currentsplit] = len()



##############################################################
##  Classifying with LGBM 
##############################################################


elif run_lgbm and not run_xgboost:

    raise ValueError(
        'No lgbm training configured with hierarchical clustering feature selection')


else:
    raise ValueError('run_xgboost and run_lgbm cannot be both True or both False for'
                  'the script to run')



####################################################################
## Extract mean,min,max of balanced accuracy and kept feature names 
####################################################################


## calculate and write the saving of the mean balanced accuracies
# Calculate the mean accuracies 



### Hierarcical Clustering 

balanced_accuracies_hclustering = list()

mean_balanced_accuracies_hclustering = list()
min_balanced_accuracies_hclustering = list()
max_balanced_accuracies_hclustering = list()
std_balanced_accuracies_hclustering = list()

best_mean_hclustering = 0

for idx in range(0, len(listselfeat_number)):
    
    ba_featsel_hclustering = list()

    for i in range(nbr_of_splits):

        currentsplit =  f"split_{i}"

        ba_hclustering = float(
            balanced_accuracies['balanced_accuracies_hclustering'][currentsplit][idx]
            )
        balanced_accuracies_hclustering.append(ba_hclustering)

    
    ba_featsel_hclustering = np.asarray(balanced_accuracies_hclustering)

    mean_balanced_accuracies_hclustering.append(np.mean(ba_featsel_hclustering))
    min_balanced_accuracies_hclustering.append(np.min(ba_featsel_hclustering))
    max_balanced_accuracies_hclustering.append(np.max(ba_featsel_hclustering))
    std_balanced_accuracies_hclustering.append(np.std(ba_featsel_hclustering))


    ###### Find name of selected features that leads to the best prediction
   
    # if np.mean(ba_featsel_hclustering) > best_mean_hclustering:
    #     nbr_kept_features_hclustering = 'tofill'
    #     kept_features_hclustering = [selfeat for selfeat in selfeat_mannwhitneyu_names_allsplits[0: index+1]]
    #     best_mean_hclustering = np.mean(ba_featsel_mannwhitneyu)


mean_ba_hclustering_npy = np.asarray(mean_balanced_accuracies_hclustering)
min_ba_hclustering_npy = np.asarray(min_balanced_accuracies_hclustering)
max_ba_hclustering_npy = np.asarray(max_balanced_accuracies_hclustering)
std_ba_hclustering_npy = np.asarray(std_balanced_accuracies_hclustering)

devink = True 


####################################################################
## Select best features after cross-validation
####################################################################

# We will just keep the features that are most kept on the cross
# If there is a tie, we calculate their scores, if there are most 1s 
# than we keep them

# Let it empty for hclustering for now, especially as we don't know yet
# if this clustering is efficient at all 



# keep the ones that the most represented in the different group








##############################################################
## Save numpy files
##############################################################

# # save the mean balanced accuracies for visualization
# save_results_path = classification_eval_folder + eval_folder_name + '/'
# save_ext = '.npy' 

# if not os.path.exists(classification_eval_folder):
#     os.mkdir(classification_eval_folder)

# if not os.path.exists(save_results_path):
#     os.mkdir(save_results_path)


# if run_xgboost and not run_lgbm:
#     classifier_name = 'xgboost'
# if run_lgbm and not run_xgboost:
#     classifier_name = 'lgbm'


# print('Start saving numpy in folder: ', save_results_path)

# name_mannwhitneyu_output = '_ba_mannwhitneyut_' + str(nbr_of_splits) + 'splits_' + run_name
# np.save(save_results_path + 'mean_' + classifier_name + name_mannwhitneyu_output + save_ext, 
#     mean_ba_mannwhitneyu_npy)
# np.save(save_results_path + 'min_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
#     min_ba_mannwhitneyu_npy)
# np.save(save_results_path + 'max_' + classifier_name + name_mannwhitneyu_output + save_ext, 
#     max_ba_mannwhitneyu_npy)
# np.save(save_results_path + 'std_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
#     std_ba_mannwhitneyu_npy)
# np.save(save_results_path + 'topselfeatid_' + classifier_name  + name_mannwhitneyu_output + save_ext, 
#     sorted_bestfeatindex_mannwhitneyu)


# name_mrmr_output = '_ba_mrmr_' + str(nbr_of_splits) + 'splits_' + run_name
# np.save(save_results_path + 'mean_' + classifier_name + name_mrmr_output + save_ext, 
#     mean_ba_mrmr_npy)
# np.save(save_results_path + 'max_' + classifier_name + name_mrmr_output + save_ext, 
#     min_ba_mrmr_npy)
# np.save(save_results_path + 'min_' + classifier_name + name_mrmr_output + save_ext, 
#     max_ba_mrmr_npy)
# np.save(save_results_path + 'std_' + classifier_name + name_mrmr_output + save_ext, 
#     std_ba_mrmr_npy)
# np.save(save_results_path + 'topselfeatid_' + classifier_name  + name_mrmr_output + save_ext, 
#     sorted_bestfeatindex_mrmr)

# name_boruta_output = '_ba_boruta_' + str(nbr_of_splits) + 'splits_' + run_name
# np.save(save_results_path + 'mean_' + classifier_name + name_boruta_output + save_ext, 
#     mean_ba_boruta_npy)
# np.save(save_results_path + 'max_' + classifier_name + name_boruta_output + save_ext, 
#     min_ba_boruta_npy)
# np.save(save_results_path + 'min_' + classifier_name + name_boruta_output + save_ext, 
#     max_ba_boruta_npy)
# np.save(save_results_path + 'std_' + classifier_name + name_boruta_output + save_ext, 
#     std_ba_boruta_npy)
# np.save(save_results_path + 'topselfeatid_' + classifier_name  + name_boruta_output + save_ext, 
#     sorted_bestfeatindex_boruta)

# np.save(
#     save_results_path + classifier_name  + 'nbr_feat_kept_boruta_'  + 
#     str(nbr_of_splits) + run_name + save_ext, 
#     boruta_visu_xcoord_npy
#     )

# print('Numpy saved.')



# ##############################################################
# ## Save text file
# ##############################################################


# txtfilename = classifier_name + '_' + str(nbr_of_splits) + 'splits_' +  run_name + '_info'

# save_txt_ext = '.txt'
# save_text_path = save_results_path + txtfilename + save_txt_ext


# ## ADD NAME OF CLASSIFER THAT WAS RUNNING

# print('Start saving name and number of feature kept in best case')

# with open(save_text_path, 'w') as file:
#     file.write('** With {} classifier **'.format(classifier_name))

#     file.write('\n\n\n\n ** mannwhitneyu **')
#     file.write('\n\nBest mean balanced accuracy is:' + 
#         str(best_mean_mannwhitneyu))  
#     file.write('\n\nThe number of kept features in the best scenario is:' + 
#         str(nbr_kept_features_mannwhitneyu))  
#     file.write('\n\nThese features are:' +  
#         str(kept_features_mannwhitneyu)) 
#     file.write('\n\nThe best 5 features overall are:' +  
#         str([mrmr_finalselfeat_names]))
#     # file.write('\n\nThe best 5 features are:' +  
#     # str([kept_features[0:4] for kept_features in kept_features_mannwhitneyu]))

#     file.write('\n\n\n\n ** mrmr **')
#     file.write('\n\nBest mean balanced accuracy is:' +  
#         str(best_mean_mrmr))  
#     file.write('\n\nThe number of kept features in the best scenario is:' + 
#         str(nbr_kept_features_mrmr))  
#     file.write('\n\nThese features are:' +
#         str(kept_features_mrmr)) 
#     file.write('\n\nThe best 5 features overall are:' +  
#         str([mannwhitneyu_finalselfeat_names]))
#     #file.write('\n\nThe best 5 features are:' +  
#     # str([kept_features[0:4] for kept_features in kept_features_mrmr])) 

#     file.write('\n\n\n\n ** boruta **')
#     file.write('\n\nMean balanced accuracy is:' +  
#         str(mean_ba_boruta_npy)) 
#     file.write('\n\nhe numbers of kept features are:' + 
#         str(number_feat_kept_boruta))  
#     file.write('\n\nThe best group of selected feature close of having 5 features is:' +  
#         str([mannwhitneyu_finalselfeat_names]))
#     file.write('\n\nThese features are:' + 
#         str(selfeat_boruta_names_allsplits)) 
    

# print('Text files saved.')


#### Save all splits balanced accuracies values 

#### Save roc curve information 



