#Lucas Sancéré -

import boruta
import mrmr
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import RandomForestClassifier

# import sys
# from pandas import DataFrame
# from sklearn import linear_model, ensemble
# from src.utils.misc import convert_flatten_redundant
# import json
# import os
# import time

class FeatureSelector:
    """
    Different methods to select features from a feature array
    """
    def __init_(self, feature_array: np.ndarray, classification_array: np.ndarray):
        """
        Parameters
        ----------
        feature_array: npy array
            Array containing all the features values for each wsi image
        classification_array: npy array
            Array containing the classification output (recurrence, or no recurrence) of each wsi image

        Returns
        ------
        """
        self.feature_array = feature_array
        self.classification_array = classification_array


    def run_mrmr(self, nbr_keptfeat: int, return_scores: bool = True) -> np.ndarray:
        """
        MRMR calculations to select features (https://github.com/smazzanti/mrmr)

        Parameters
        ----------
        nbr_keptfeat: int
            Number of features to keep during the feature selection process
        return_scores: bool
            Display the mrmr scores of each features
        Returns
        -------
        selfeat_mrmr: npy array
            Array continaing the index of the selected features
            (the index correspond to the index of the features in the feature array)
        """
        # We create a pandas dataframe from the feature array
        try:
            X = pd.DataFrame(self.feature_array)
            X = np.transpose(X)
            X = X.astype('float32')
        except NameError:
            print('The Features Array cannot be generated, '
                'this is probably because the Path or the name of files are not correct!')
        # We create a pandas Series from the classification array
        y = pd.Series(self.classification_array)
        y = y.astype('int8')
        # Run mrrmr
        selfeat_mrmr = mrmr.mrmr_classif(X=X, y=y, K=nbr_keptfeat, return_scores=return_scores)
        return selfeat_mrmr


    def run_boruta(self, class_weight: str = 'balanced',
                   max_depth: int = 5, random_state: int = 1) -> np.ndarray:
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
            Array continaing the index of the selected features
            (the index correspond to the index of the features in the feature array)
        """
        rf = RandomForestClassifier(n_jobs=-1, class_weight=class_weight, max_depth=max_depth)
        # Define Boruta feature selection method
        method_boruta = boruta.BorutaPy(rf, n_estimators='auto', verbose=2, random_state=random_state)
        # IMPORTANT RMQ: BorutaPy was fixed using
        # https://github.com/scikit-learn-contrib/boruta_py/commit/e04d1a17de142679eabebebd8cedde202587fbf1
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


    def run_mannwhitney(self) -> dict:
        """
        Mann-Whitney U rank test appliied on each features

        Returns
        -------
        orderedp_mannwhitneyu: dict
            Dictionary containing the p-values of each features (key: feature index, value: p-value)
            calculated with Mann-Whitney U rank test.
            The dictionary is ordered from the highest p-value to the lowest p-value
        """
        mannwhitneyu = dict()
        orderedp_mannwhitneyu = dict()  # p stands for p-values
        # Create 2 features arrays:
        # 1 for all the features avalues associated to recurrence,
        # the other one for no-recurrence
        featrec = [self.feature_array[:, index] for index in range(0, self.feature_array.shape[1])
                   if self.classification_array[index] == 1]
        featnorec = [self.feature_array[:, index] for index in range(0, self.feature_array.shape[1])
                     if self.classification_array[index] == 0]
        featrec = np.asarray(featrec)
        featnorec = np.asarray(featnorec)
        for feat in range(0, self.feature_array.shape[0]):  # we navigated into the features now
            mannwhitneyu[feat] = scipy.stats.mannwhitneyu(featrec[:, feat],
                                                          featnorec[:, feat])
            # We Only kept pvalues:
            orderedp_mannwhitneyu[feat] = scipy.stats.mannwhitneyu(featrec[:, feat],
                                                                   featnorec[:, feat]).pvalue

        # Order the dict by values:
        orderedp_mannwhitneyu = sorted(orderedp_mannwhitneyu.items(),
                                       key=lambda x: x[1])
        return orderedp_mannwhitneyu