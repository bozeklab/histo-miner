#Lucas Sancéré -

import sys
sys.path.append('../')  # Only for Remote use on Clusters

import os
import numpy as np
import yaml
from attrdict import AttrDict as attributedict

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import joblib



###### DEV ATTENTION #####
### Add saving of the predictions
##########################



#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.tissue_analyser_main
nbr_keptfeat = config.parameters.int.nbr_keptfeat
classification_from_allfeatures = config.parameters.bool.classification_from_allfeatures
displayclass_score = config.parameters.bool.display_classification_scores
displayclass_pred = config.parameters.bool.display_classification_predictions



############################################################
## Create Paths Strings to Classifiers
############################################################


# Folder name to save models (might not be used)
modelfolder = pathtofolder +'/classification_models/'

pathnameofmodel_vanilla = modelfolder + 'nameofmodel_vanilla.joblib'
pathnameofmodel_mrmr = modelfolder + 'nameofmodel_mrmr.joblib'
pathnameofmodel_boruta = modelfolder + 'nameofmodel_boruta.joblib'
pathnameofmodel_mannwhitney = modelfolder + 'nameofmodel_mannwhitney.joblib'



############################################################
## Load inference data
############################################################

pathfeatselect = pathtofolder + '/feature_selection/'
ext = '.npy'

print('Load feeature selection numpy files...')

pathfeatarray = pathfeatselect + 'inffeatarray' + ext
pathclarray = pathfeatselect + 'infclarray' + ext
#TO DO:
#raise an error here if one of the file doesn't exist

inf_featarray = np.load(pathfeatarray)
inf_clarray = np.load(pathclarray)
inf_clarray = np.transpose(inf_clarray)


# Each time we check if the file exist because all selections are not forced to run
pathselfeat_mrmr = pathfeatselect + 'selfeat_mrmr_idx' + ext
if os.path.exists(pathselfeat_mrmr):
    selfeat_mrmr = np.load(pathselfeat_mrmr, allow_pickle=True)
pathselfeat_boruta = pathfeatselect + 'selfeat_boruta_idx' + ext
if os.path.exists(pathselfeat_boruta):
    selfeat_boruta = np.load(pathselfeat_boruta, allow_pickle=True)
pathorderedp_mannwhitneyu = pathfeatselect + 'selfeat_mannwhitneyu_idx' + ext
if os.path.exists(pathorderedp_mannwhitneyu):
    orderedp_mannwhitneyu = np.load(pathorderedp_mannwhitneyu, allow_pickle=True)
print('Loading done.')



############################################################
## Classifiers Inference
############################################################


#### Classification  with all features kept 

if classification_from_allfeatures:
    # Load test data (no feature selection)
    inf_globfeatarray = np.transpose(inf_featarray)

    # Predict the labels for new data
    ##### nameofmodel
    if os.path.exists(pathnameofmodel_vanilla):
        nameofmodel_vanilla = joblib.load(pathnameofmodel_vanilla)
        if displayclass_pred:
            nameofmodel_vanilla_pred = nameofmodel_vanilla.predict(inf_globfeatarray)
            print('nameofmodel_pred : {}'.format(nameofmodel_vanilla_pred))
        if displayclass_score:
            print("Accuracy of nameofmodel classifier:",
              nameofmodel_vanilla.score(inf_globfeatarray, inf_clarray))



#### Parse the featarray to the class SelectedFeaturesMatrix 

SelectedFeaturesMatrix = SelectedFeaturesMatrix(inf_featarray)

#### Classification training with the features kept by mrmr

if os.path.exists(pathselfeat_mrmr):
    # Load test data (that went through mrmr method)
    test_featarray_mrmr = SelectedFeaturesMatrix.mrmr_matr(selfeat_mrmr)
    # test_featarray_mrmr = np.transpose(test_featarray_mrmr)

    # Predict the labels for new data
    ##### nameofmodel
    if os.path.exists(pathnameofmodel_mrmr):
        nameofmodel_mrmr = joblib.load(pathnameofmodel_mrmr)
        if displayclass_pred:
            nameofmodel_mrmr_pred = nameofmodel_mrmr.predict(test_featarray_mrmr)
            print('nameofmodel_mrmr_pred : {}'.format(nameofmodel_mrmr_pred))
        if displayclass_score:
            print("Accuracy of nameofmodel MRMR classifier:",
              nameofmodel_mrmr.score(test_featarray_mrmr, inf_clarray))


#### Classification training with the features kept by boruta

if os.path.exists(pathselfeat_boruta):
    test_featarray_boruta = SelectedFeaturesMatrix.mrmr_matr(selfeat_boruta)
    # test_featarray_boruta = np.transpose(test_featarray_boruta)

    # Predict the labels for new data
    ##### nameofmodel
    if os.path.exists(pathnameofmodel_boruta):
        nameofmodel_boruta = joblib.load(pathnameofmodel_boruta)
        if displayclass_pred:
            nameofmodel_boruta_pred = nameofmodel_boruta.predict(test_featarray_boruta)
            print('nameofmodel_boruta_pred : {}'.format(nameofmodel_boruta_pred))
        if displayclass_score:
            print("Accuracy of nameofmodel BORUTA classifier:",
              nameofmodel_boruta.score(test_featarray_boruta, inf_clarray))


#### Classification training with the features kept by mannwhitneyu

if os.path.exists(pathorderedp_mannwhitneyu):
    test_featarray_mannwhitney = SelectedFeaturesMatrix.mannwhitney_matr(orderedp_mannwhitneyu)
    # test_featarray_mannwhitney = np.transpose(test_featarray_mannwhitney)

    # Predict the labels for new data
    ##### nameofmodel
    if os.path.exists(pathnameofmodel_mannwhitney):
        nameofmodel_mannwhitney = joblib.load(pathnameofmodel_mannwhitney)
        if displayclass_pred:
            nameofmodel_mannwhitney_pred = nameofmodel_mannwhitney.predict(test_featarray_mannwhitney)
            print('nameofmodel_mannwhitney_pred : {}'.format(nameofmodel_mannwhitney_pred))
        if displayclass_score:
            print("Accuracy of nameofmodel MANN WHITNEY classifier:",
              nameofmodel_mannwhitney.score(test_featarray_mannwhitney, inf_clarray))

