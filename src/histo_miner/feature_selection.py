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
    def __init__(self, feature_array: np.ndarray, classification_array: np.ndarray):
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
        selfeat_mrmr_index: npy array
            Array containing the index of the selected features
            (the index correspond to the index of the features in the feature array, starts at 0)
        TO FILL
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
        selfeat_mrmr_index = selfeat_mrmr[0]
        mrmr_relevance_matrix = selfeat_mrmr[1]
        mrmr_redundancy_matrix = selfeat_mrmr[2]
        return selfeat_mrmr_index, mrmr_relevance_matrix, mrmr_redundancy_matrix


    def run_boruta(self, class_weight: str = 'balanced',
                   max_depth: int = 15, random_state: int = 1) -> np.ndarray:
        """
        Boruta calculations to select features (https://github.com/scikit-learn-contrib/boruta_py)

        Parameters
        ----------
        class_weight: {“balanced”, “balanced_subsample”}, dict 
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
            Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].
            The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
            The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.
            For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
        max_depth: int or None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or 
            until all leaves contain less than min_samples_split samples.
        random_state: int RandomState instance or None; default=None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.
        Returns
        -------
        selfeat_boruta_index: list
            List containing the index of the selected features
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
        # Calculate index of Selected Featurs 
        print('Selected Feature are:', method_boruta.support_)
        selfeat_boruta_index = [i for i, val in enumerate( method_boruta.support_) if val.any() == True]
        # Check selected features
        # Select the chosen features from our dataframe.
        # selfeatmatrix_boruta = X[:, method_boruta.support_]
        # NOT A GOOD IDEA TO CREATE the feature matrix here as the feat matrix with all
        # feature could be split later on
        return selfeat_boruta_index


    def run_mannwhitney(self, nbr_keptfeat: int) -> dict:
        """
        Mann-Whitney U rank test applied on each features

        Returns
        -------
        selfeat_mannwhitneyu_index: list
            List containing the index of the selected features
            (the index correspond to the index of the features in the feature array)
        orderedp_mannwhitneyu: dict
            Dictionary containing the p-values of each features (key: feature index, value: p-value)
            calculated with Mann-Whitney U rank test. Feature index starts at 0.
            The dictionary is ordered from the highest p-value to the lowest p-value.
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

        # Order the list by values:
        orderedp_mannwhitneyu = sorted(orderedp_mannwhitneyu.items(),
                                       key=lambda x: x[1])
        # Above we have a list of tuples,  for later processing we need a list of lists:
        orderedp_mannwhitneyu = [list(t) for t in orderedp_mannwhitneyu]
        #Find list of index from the ordered dict
        selfeat_mannwhitneyu_index = orderedp_mannwhitneyu[:nbr_keptfeat]
        selfeat_mannwhitneyu_index = [index for (index,value) in selfeat_mannwhitneyu_index]
        return selfeat_mannwhitneyu_index, orderedp_mannwhitneyu



class SelectedFeaturesMatrix:
    """
    Generate the matrix of selected features based on the output of a given feature selection method

    Note: No need for Boruta method, as the output is already the matrix of selected features
    """
    def __init__(self, feature_array: np.ndarray,):
        """
        Parameters
        ----------
        feature_array: npy array
            Array containing all the originial (before selection) features values for each wsi image
        Returns
        ------
        """
        self.feature_array = feature_array


    def mrmr_matr(self,  selfeat_mrmr_index: np.ndarray) -> np.ndarray:
        """
        Generate the matrix of selected features from mrmr method output

        Parameters
        ----------
        selfeat_mrmr_index: list
            List containing the index of the selected features mrmr
            (the index correspond to the index of the features in the feature array)
        Returns
        -------
        featarray_mrmr: {ndarray, sparse matrix} of shape (n_samples, n_features)
            Matrix with the values of features selected with mrmr
        """
        # Be sure to enter output from mrmr here
        mrmrselectedfeatures_idx = sorted(selfeat_mrmr_index)
        featarray_mrmr = np.transpose(self.feature_array)
        featarray_mrmr = featarray_mrmr[:, mrmrselectedfeatures_idx ]
        return featarray_mrmr


    def boruta_matr(self,  selfeat_boruta_index: np.ndarray) -> np.ndarray:
        """
        Generate the matrix of selected features from mrmr method output

        Parameters
        ----------
        selfeat_boruta_index: list
            List containing the index of the selected features by boruta
            (the index correspond to the index of the features in the feature array)
        Returns
        -------
        featarray_mrmr: {ndarray, sparse matrix} of shape (n_samples, n_features)
            Matrix with the values of features selected with boruta
        """
        # Be sure to enter output from mrmr here
        borutaselectedfeatures_idx = sorted(selfeat_boruta_index)
        featarray_boruta = np.transpose(self.feature_array)
        featarray_boruta = featarray_boruta[:, borutaselectedfeatures_idx]
        return featarray_boruta


    def mannwhitney_matr(self, selfeat_mannwhitneyu_index: np.ndarray) -> np.ndarray:
        """
        Generate the matrix of selected features from Mann-Whitney U rank test output

        Parameters
        ----------
        selfeat_mannwhitneyu_index: list
            List containing the index of the selected features from Mann-Whitney U rank test 
            (the index correspond to the index of the features in the feature array)
        Returns
        -------
        featarray_mrmr: {ndarray, sparse matrix} of shape (n_samples, n_features)
            Matrix with the values of features selected with Mann-Whitney U rank test
        """
        # Be sure to enter output from  Mann-Whitney U rank test here
        featarray_mannwhitney = np.transpose(self.feature_array)
        featarray_mannwhitney = featarray_mannwhitney[:, selfeat_mannwhitneyu_index]
        return featarray_mannwhitney
