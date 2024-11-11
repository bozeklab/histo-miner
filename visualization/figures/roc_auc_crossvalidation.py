#Lucas Sancéré 

import os
import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import random

import math
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
import xgboost 
import lightgbm
from attrdictionary import AttrDict as attributedict
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_curve, RocCurveDisplay, auc
from sklearn.preprocessing import StandardScaler 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter, defaultdict

from src.histo_miner.feature_selection import SelectedFeaturesMatrix, FeatureSelector
import src.histo_miner.utils.misc as utils_misc


# We directly set the best case: xgboost and we keep 10 best features 

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
eval_folder_name = confighm.names.eval_folder
pathtosavefolder = confighm.paths.folders.visualizations


with open("./../../configs/classification.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
test_lesssamples = config.parameters.bool.test_lesssamples
nbr_of_splits = config.parameters.int.nbr_of_splits
run_name = config.names.run_name

xgboost_random_state = config.classifierparam.xgboost.random_state
xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
xgboost_lr = config.classifierparam.xgboost.learning_rate
xgboost_objective = config.classifierparam.xgboost.objective

              

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




##############################################################
## Load Classifiers
##############################################################


##### XGBOOST
xgboost = xgboost.XGBClassifier(random_state= xgboost_random_state,
                                n_estimators=xgboost_n_estimators, 
                                learning_rate=xgboost_lr, 
                                objective=xgboost_objective,
                                verbosity=0)



##############################################################
## Traininig Classifiers to obtain instance prediction score
##############################################################


print('Start Classifiers trainings...')


train_featarray = np.transpose(train_featarray)

# Test with CPI data if to see how the cross validation perform with less sample
if test_lesssamples:
    # random generate numbers between 0 and 37 to remove them form the list 
    idx_rmv_raws = np.random.randint(0,37,16)# Removing the specified rows
    train_featarray = np.delete(train_featarray, idx_rmv_raws, axis=0)
    train_clarray = np.delete(train_clarray, idx_rmv_raws, axis=0)


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
##  AUC with XGBoost 
##############################################################


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6))


balanced_accuracies = {"balanced_accuracies_mrmr": {"initialization": True}}

#to calculate AUC
all_y_pred = dict()
all_y_test = dict()

length_selfeatmrmr = dict()
selfeat_mrmr_names_allsplits = []
selfeat_mrmr_id_allsplits = []

all_features_balanced_accuracy = list()


for i in range(nbr_of_splits):  

    currentsplit =  f"split_{i}"

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

    ## mr.MR calculations
    print('Selection of features with mrmr method...')
    selfeat_mrmr = feature_selector.run_mrmr(nbr_feat)
    selfeat_mrmr_index = selfeat_mrmr[0]
    # Now associate the index of selected features (selfeat_mrmr_index) to the list of names:
    selfeat_mrmr_names = [featnameslist[index] for index in selfeat_mrmr_index] 
    selfeat_mrmr_names_allsplits.append(selfeat_mrmr_names)
    selfeat_mrmr_id_allsplits.append(selfeat_mrmr_index)


    ########## GENERATION OF MATRIX OF SELECTED FEATURES
    # If the class was not initalized, do it. If not, reset attributes if the class instance
    # if i == 0:
    #     selected_features_matrix = SelectedFeaturesMatrix(X_train_tr)
    # else:
    #     selected_features_matrix.reset_attributes(X_train_tr)
    feature_array = X_train

    ########## TRAINING AND EVALUATION WITH FEATURE SELECTION
    balanced_accuracies_mrmr = list()
    all_pred = list()
    all_test = list()

    print('Calculate balanced_accuracies for decreasing number of features kept')
    ### With mrmr selected features
    nbr_keptfeat_idx = 10

    # Kept the selected features
    selfeat_mrmr_index_reduced =  selfeat_mrmr_index[0:nbr_keptfeat_idx]
    selfeat_mrmr_index_reduced = sorted(selfeat_mrmr_index_reduced)

    # Generate matrix of features
    featarray_mrmr = feature_array[:, selfeat_mrmr_index_reduced]

    #Training
    # needs to be re initialized each time!!!! Very important
    xgboost_mrmr_training = xgboost
    # actual training
    xgboost_mrmr_training_inst = xgboost_mrmr_training.fit(featarray_mrmr, 
                                                           y_train)

    # Predictions on the test split
    y_pred_mrmr = xgboost_mrmr_training_inst.predict(
        X_test[:, selfeat_mrmr_index_reduced]
        )
    # Calculate balanced accuracy for the current split
    balanced_accuracy_mrmr = balanced_accuracy_score(y_test, 
                                                     y_pred_mrmr)
    balanced_accuracies_mrmr.append(balanced_accuracy_mrmr)

    # Plot ROC curve and capture the RocCurveDisplay object
    viz = RocCurveDisplay.from_estimator(
        xgboost_mrmr_training_inst,
        X_test[:, selfeat_mrmr_index_reduced],
        y_test,
        name=f"ROC fold {currentsplit}",
        alpha=0.3,
        lw=1,
        ax=ax
    )

    # Update the label to include the AUC with three decimal places
    viz.line_.set_label(f"ROC fold {fold} (AUC = {viz.roc_auc:.3f})")

    # Interpolate the TPR for mean ROC curve calculation
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    # update changing size of kept feature for mrmr
    length_selfeatmrmr[currentsplit] = len(selfeat_mrmr_index)

    # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
    balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies_mrmr


    ### Store results 
    # store all resutls in the main dict knowing it will be repeated 10times
    # # maybe create a nested dict, split1, split2 and so on!!
    # currentsplit =  f"split_{i}"

    # # Fill the dictionnary with nested key values pairs for the different balanced accurarcy
    # balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies_mrmr

    # # Fill the dictionnary AUC cacluaiton 
    # all_y_pred['pred_test_vectors'][currentsplit] = all_pred
    # all_y_test['gt_test_vectors'][currentsplit] = all_test

    # # update changing size of kept feature for mrmr
    # length_selfeatmrmr[currentsplit] = len(selfeat_mrmr_index)



####################################################################
## Temporary check if best ba accuracy
####################################################################

## calculate and write the saving of the mean balanced accuracies
## Do for mrmr
# Calculate the mean accuracies 

mean_balanced_accuracies_mrmr = list()
min_balanced_accuracies_mrmr = list()
max_balanced_accuracies_mrmr = list()
std_balanced_accuracies_mrmr = list()

best_mean_mrmr = 0


# we want to delete the first elements of the lists when there is not the same number of features kept by mrmr.
# but we keep the first element as it is with all features

listoflengths = list()

for i in range(nbr_of_splits):

    currentsplit =  f"split_{i}"
    listoflengths.append(length_selfeatmrmr[currentsplit])

nbrkept_max_allsplits = min(listoflengths)
print(nbrkept_max_allsplits)

# remove firsts elements to be comparable in termes of number of feat kept
for i in range(nbr_of_splits):

    currentsplit =  f"split_{i}"
    # We use sclicing to keep the nbr_feat-nbrkept_max_allsplits last elements of the list 
    balanced_accuracies['balanced_accuracies_mrmr'][currentsplit] = balanced_accuracies['balanced_accuracies_mrmr'][currentsplit][-nbrkept_max_allsplits:]

    selfeat_mrmr_names_allsplits[i] = selfeat_mrmr_names_allsplits[i][-nbrkept_max_allsplits:]


# for index in range(0, nbrkept_max_allsplits):

ba_featsel_mrmr = list()

for i in range(nbr_of_splits):

    currentsplit =  f"split_{i}"

    balanced_accuracy_mrmr = np.asarray(
        balanced_accuracies['balanced_accuracies_mrmr'][currentsplit][0]
        ) 

    ba_featsel_mrmr.append(balanced_accuracy_mrmr)


ba_featsel_mrmr = np.asarray(ba_featsel_mrmr)

mean_balanced_accuracies_mrmr.append(np.mean(ba_featsel_mrmr))
min_balanced_accuracies_mrmr.append(np.min(ba_featsel_mrmr))
max_balanced_accuracies_mrmr.append(np.max(ba_featsel_mrmr))
std_balanced_accuracies_mrmr.append(np.std(ba_featsel_mrmr))

###### Find name of selected features that leads to the best prediction
kept_features_mrmr = [split[0: nbr_keptfeat_idx] for split in selfeat_mrmr_names_allsplits]
best_mean_mrmr = np.mean(ba_featsel_mrmr)


mean_ba_mrmr_npy = np.asarray(mean_balanced_accuracies_mrmr)
min_ba_mrmr_npy = np.asarray(min_balanced_accuracies_mrmr)
max_ba_mrmr_npy = np.asarray(max_balanced_accuracies_mrmr)
std_ba_mrmr_npy = np.asarray(std_balanced_accuracies_mrmr)


# for all features

all_features_balanced_accuracy_npy = np.asarray(all_features_balanced_accuracy)
mean_ba_allfeat = np.mean(all_features_balanced_accuracy_npy)

debug = True 

##############################################################
##  Visualization of AUC  
##############################################################


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=fr"Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})",
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label 'response')",
)
ax.legend(loc="lower right")


## Save figure 
#Create Name for saving
savename = 'auc_crossval_2.png'

#Saving
# if not os.path.exists(pathtosavefolder + '/CorrMatrix/'):
#     os.makedirs(pathtosavefolder + '/CorrMatrix/')

savedfig_path = pathtosavefolder + savename
plt.savefig(savedfig_path)
plt.clf()



