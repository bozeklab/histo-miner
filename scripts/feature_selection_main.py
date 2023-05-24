#Lucas Sancéré -

import json
import os
import time

import numpy as np
import scipy.stats
import sys
import pandas as pd
import mrmr
import boruta
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, ensemble
from src.utils.misc import convert_flatten_redundant
from src.histo_miner.feature_selection import FeatureSelector

# sys.path.append('../')  # Only for Remote use on Clusters



"""
This file is to update fully
We will abandon a bit the mrmr repo to do all the feature selections here 

-> needs to play with the last inferences from hvn to updates these repo

Use config file sinstead of putting the args here
"""



"""Collection of scripts for different Hovernet tasks"""

# Commun parameters


Parentdir = '/home/lsancere/These/CMMC/Ada_Mount/shared/scc/hovernet_prediction/pipelinetests/toysample_for_test/json-analyses'  # WITHOUT LAST '/' !
#Export result from Mann Withney U rank test, ranked best features
MannWithneyUvalues = [11, 6, 5, 12, 0]
Nbr_keptfeat = 5 # Number features ton

# run_mrmr = True
# run_boruta = True
# run_mannwhitney = True


# Keep it as BACKUp but must be deleted
import_MannWithneyU = False

# Parentdir = '/projects/ag-bozek/lucas/scc/hovernet_prediction/test_jsondata2/test1'  #WITHOUT LAST '/' !


"""Concatenate the quantification features all together in pandas DataFrame and run the different feature selections"""



####### TO ADD: Associate the index of the selected feature to the name of it (use dict probably)

# Try to use https://github.com/smazzanti/mrmr with pandas dataframes as input
# Boruta, github repo is: https://github.com/scikit-learn-contrib/boruta_py

cllist = list()  # classification list, binary with recurrence as 1 and no recurrence as 0
featureInit = False # initialisation of feature array not done yet

for root, dirs, files in os.walk(Parentdir):
    if files:  # Keep only the not empty lists of files
        for file in files:  # Because files is a list of file name here, and not a srting. You create a string with this line
            Path, Extension = os.path.splitext(file)
            PathtoParentFolder, Nameoffile = os.path.split(Path)  # PathtoParentFolder is empty, why?)
            if Extension == '.json' and 'data' in Nameoffile:
                with open(root + '/' + file, 'r') as filename:
                    data = filename.read()  # extract information of the JSON as a string
                    print(file)
                    data = json.loads(data)  # read JSON formatted string and convert it to a dict
                    data = convert_flatten_redundant(data)  # flatten the dict (with redundant keys in nested dict, see function)
                    data = {k: v for (k, v) in data.items() if v != 'Not calculated'} #HEre is some dict comprehension / dicitionnary comprehension

                    #Convert dict values into an array
                    valuearray = np.fromiter(data.values(), dtype=float)
                    #Remove nans from arrays and add a second dimension to the array in order to be concatenated later on
                    valuearray = valuearray[~np.isnan(valuearray)]
                    valuearray = np.expand_dims(valuearray, axis=1)

                    # Generate the list of WSI binary classification
                    # No list comprehension just to exhibit more clearly the error message
                    if 'recurrence' in Nameoffile: #Yes be careful the name are not the best choice
                        if 'norecurrence' in Nameoffile:
                            cllist.append(int(0))
                        else:
                            cllist.append(int(1))
                    else:
                        raise ValueError('Some features are not associated to a recurrence or norecurrence WSI classification. User must sort JSON and rename it'
                                         'with the corresponding reccurence and noreccurence caracters')

            if not featureInit:
                featarray = valuearray
                featureInit = True
            else:
                #MEMORY CONSUMING, FIND BETTER WAY IF POSSIBLE
                featarray = np.concatenate((featarray, valuearray), axis=1)

if cllist:
    if 0 not in cllist:
        raise ValueError('The data contains only no recurrence data or json named as being norecurrence. To run statistical test we need both'
                         ' recurrence and norecurrence examples')

    if 1 not in cllist:
        raise ValueError('The data contains only recurrence data or json named as being recurrence. To run statistical test we need both'
            ' recurrence and norecurrence examples')

clarray = np.asarray(cllist) #cl array for classification array (recurrence or norecurrence)
print("Feature Matrix Shape is", featarray.shape)
print("Classification Vector is", clarray)


# Run the different feature selection methods

print('mR.MR calculations (See https://github.com/smazzanti/mrmr) in progress...')
Selfeat_mrmr = FeatureSelector.run_mrmr(featarray, clarray, Nbr_keptfeat)
print('Selected Features: {}'.format(Selfeat_mrmr[0]))
print('Relevance Matrix: {}'.format(Selfeat_mrmr[1]))
print('Redundancy Matrix: {}'.format(Selfeat_mrmr[2]))

Selfeat_boruta = FeatureSelector.run_boruta(featarray, clarray)
print("Selected Feature Matrix Shape")
print(Selfeat_boruta.shape)
print(Selfeat_boruta)

Orderedp_mannwhitneyu = FeatureSelector.run_mannwhitney(featarray, clarray)
print('**Output Ordered from best p-values to worst**')
print(Orderedp_mannwhitneyu)
















