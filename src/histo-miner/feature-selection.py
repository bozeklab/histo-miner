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

class FeatureSelector:
    """
    Different methods to select features from a feature array
    """
    def __init_(self, feature_array, classification_arrray):
        """
        Parameters
        ----------
        feature_array: npy array
            Array containing all the features values for each wsi image
        classification_arrray: npy array
            Array containing the classification output (recurrence, or no recurrence) of each wsi image

        Returns
        ------
        """
        self.feature_array = feature_array
        self.classification_array = classification_arrray


    def run_mrmr(self, nbr_keptfeat, return_scores=True):
        """
        MRMR calculations to select features (https://github.com/smazzanti/mrmr)

        Parameters
        ----------
        nbr_keptfeat: int
            Number of features to keep during the feature selection process
        return_scores
            Display the mrmr scores of each features
        Returns
        -------
        selfeat_mrmr: npy array
            Array continaing the index of the selected features (the index correspond to the index of the features in the feature array)
        """
        # We create a pandas dataframe from the feature array
        try:
            X = pd.DataFrame(self.feature_array)
            X = np.transpose(X)
            X = X.astype('float32')
        except NameError:
            print(
                'The Features Array cannot be generated, this is probably because the Path or the name of files are not correct!')
        # We create a pandas Series from the classification array
        y = pd.Series(self.classification_array)
        y = y.astype('int8')
        # Run mrrmr
        selfeat_mrmr = mrmr.mrmr_classif(X=X, y=y, K=nbr_keptfeat, return_scores=return_scores)
        return selfeat_mrmr


    def run_boruta(self, class_weight='balanced', max_depth=5, random_state=1):
        """
        Boruta calculations to select features (https://github.com/scikit-learn-contrib/boruta_py)

        Parameters
        ----------
        class_weight: str
            TDL: Check the doc of RandomForestClassifier
        max_depth: int
            TDL: Check the doc of RandomForestClassifier
        random_state: int
            TDL: Check the doc of boruta.BorutaPy

        Returns
        -------
        selfeat_boruta: npy array
            Array continaing the index of the selected features (the index correspond to the index of the features in the feature array)
        """
        rf = RandomForestClassifier(n_jobs=-1, class_weight=class_weight, max_depth=max_depth)
        # Define Boruta feature selection method
        method_boruta = boruta.BorutaPy(rf, n_estimators='auto', verbose=2, random_state=random_state)
        # IMPORTANT RMQ: BorutaPy was fixed using https://github.com/scikit-learn-contrib/boruta_py/commit/e04d1a17de142679eabebebd8cedde202587fbf1
        # BorutaPy accepts nupy arrays only, maybe not the same as for mrmr
        # Use already generated numpy vectors instead of pandas dataframe
        X = np.transpose(self.feature_array)  # need to have X transposed to have correct Boruta input
        y = self.classification_array
        method_boruta.fit(X, y)
        print('Selected Feature are:', method_boruta.support_)
        # Check selected features
        # Select the chosen features from our dataframe.
        selfeat_boruta = X[:, method_boruta.support_]
        return selfeat_boruta


    def run_mannwhitney(self):
        """
        Mann-Whitney U rank test appliied on each features

        Returns
        -------
        orderedp_mannwhitneyu: dict
            Dictionary containing the p-values of each features (key: feature index, value: p-value) calculated with Mann-Whitney U rank test.
            The dictionary is ordered from the highest p-value to the lowest p-value
        """
        mannwhitneyu = dict()
        orderedp_mannwhitneyu = dict()  # p stands for p-values
        # Create 2 features arrays: 1 for all the features avalues associated to recurrence, the other one for no-recurrence
        # print(featarray)
        # print(featarray.shape[0])
        featrec = [self.feature_array[:, index] for index in range(0, self.feature_array.shape[1]) if self.classification_array[index] == 1]
        featnorec = [self.feature_array[:, index] for index in range(0, self.feature_array.shape[1]) if self.classification_array[index] == 0]
        featrec = np.asarray(featrec)
        featnorec = np.asarray(featnorec)
        for feat in range(0, self.feature_array.shape[0]):  # we navigated into the features now
            mannwhitneyu[feat] = scipy.stats.mannwhitneyu(featrec[:, feat], featnorec[:, feat])
            orderedp_mannwhitneyu[feat] = scipy.stats.mannwhitneyu(featrec[:, feat],
                                                                   featnorec[:, feat]).pvalue  # Only keep pvalues
        orderedp_mannwhitneyu = sorted(orderedp_mannwhitneyu.items(), key=lambda x: x[1])  # Order the dict by values
        return orderedp_mannwhitneyu



"""Concatenate the quantification features all together in pandas DataFrame and run MRMR"""




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



































######### CLASSIFIERS ################  /!\   -----> PREDICTION ON TRAINING DATA§ Not good!! Split the set into 2 when more data!!!!!
#For now we just check that the code is working, it is a debugging step
# Was here before


#### TO KEEP IN BACKUP FOR NOW
# if import_MannWithneyU:
#     MannWithneyU = sorted(MannWithneyUvalues)
#     featarray_mwu = np.transpose(featarray)
#     featarray_mwu = featarray_mwu[:, MannWithneyU]
#     print('Shape featarray_MWU', featarray_mwu.shape)
#     print('Shape clarray', clarray.shape)
#     # RIDGE CLASSIFIER
#     # Initialize the RidgeClassifier with an alpha value of 0.5
#     ridge_mwu = linear_model.RidgeClassifier(alpha=0.5)
#     # Fit the classifier to the data
#     ridge_mwu.fit(featarray_mwu, clarray)
#     # Predict the labels for new data
#     ridge_mwu_pred = ridge_mwu.predict(featarray_mwu)
#     print('ridge_mwu_pred : {}'.format(ridge_mwu_pred))
#     # Print the accuracy of the classifier
#     print("Accuracy of RIDGE MWU classifier:", ridge_mwu.score(featarray_mwu, clarray))
#     # LOGISTIC REGRESSION
#     lr_mwu = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
#     # Fit the classifier to the data
#     lr_mwu.fit(featarray_mwu, clarray)
#     # Predict the labels for new data
#     lr_mwu_pred = lr_mwu.predict(featarray_mwu)
#     print('lr_mwu_pred : {}'.format(lr_mwu_pred))
#     # Print the accuracy of the classifier
#     print("Accuracy of LOGISTIC MWU classifier:", lr_mrmr.score(featarray_mwu, clarray))












