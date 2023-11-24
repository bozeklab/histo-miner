#Lucas Sancéré -

import boruta
import mrmr
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import RandomForestClassifier

import sys
from pandas import DataFrame
from sklearn import linear_model, ensemble
from src.utils.misc import convert_flatten_redundant
import json
import os
import time


import yaml











# POSSIBLY NO  NEED OF THIS FILE!!!! ----------------------------------------------------------------
############# TO REMOVE ################


class Classifier:
    """
    Different methods to classify a feature array into "recurrence" and "no recurrence" cancer categories
    """
    def __init__(self):
        pass




class ClassifierTrain:
    """
    Different methods to classify a feature array into "recurrence" and "no recurrence" cancer categories
    """
    def __init__(self, config_relativepath):
        self.config_relativepath = config_relativepath


    def load_trainparameters(self):
        with open(self.config_relativepath, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # Create a config dict from which we can access the keys with dot syntax
        config = attributedict(config)
        self.ridge_random_state = config.classifierparam.ridge.random_state
        self.ridge_alpha = config.classifierparam.ridge.alpha
        self.ridge_param_grid_random_state = list(config.classifierparam.ridge.grid_dict.random_state)
        self.ridge_param_grid_alpha = list(config.classifierparam.ridge.grid_dict.alpha)

        self.lregression_random_state = config.classifierparam.logistic_regression.random_state
        self.lregression_penalty = config.classifierparam.logistic_regression.penalty
        self.lregression_solver = config.classifierparam.logistic_regression.solver
        self.lregression_multi_class = config.classifierparam.logistic_regression.multi_class
        self.lregression_class_weight = config.classifierparam.logistic_regression.class_weight
        self.lregression_param_grid_random_state = list(config.classifierparam.logistic_regression.grid_dict.random_state)
        self.lregression_param_grid_penalty = list(config.classifierparam.logistic_regression.grid_dict.penalty)
        self.lregression_param_grid_solver = list(config.classifierparam.logistic_regression.grid_dict.solver)
        self.lregression_param_grid_multi_class = list(config.classifierparam.logistic_regression.grid_dict.multi_class)
        self.lregression_param_grid_class_weight = list(config.classifierparam.logistic_regression.grid_dict.class_weight)

        self.forest_random_state = config.classifierparam.random_forest.random_state
        self.forest_n_estimators = config.classifierparam.random_forest.n_estimators
        self.forest_class_weight = config.classifierparam.random_forest.class_weight
        self.forest_param_grid_random_state = list(config.classifierparam.random_forest.grid_dict.random_state)
        self.forest_param_grid_n_estimators = list(config.classifierparam.random_forest.grid_dict.n_estimators)
        self.forest_param_grid_class_weight = list(config.classifierparam.random_forest.grid_dict.class_weight)

        self.xgboost_random_state = config.classifierparam.xgboost.random_state
        self.xgboost_n_estimators = config.classifierparam.xgboost.n_estimators
        self.xgboost_lr = config.classifierparam.xgboost.learning_rate
        self.xgboost_objective = config.classifierparam.xgboost.objective
        self.xgboost_param_grid_random_state = list(config.classifierparam.xgboost.grid_dict.random_state)
        self.xgboost_param_grid_n_estimators = list(config.classifierparam.xgboost.grid_dict.n_estimators)
        self.xgboost_param_grid_learning_rate = list(config.classifierparam.xgboost.grid_dict.learning_rate)
        self.xgboost_param_grid_objective = list(config.classifierparam.xgboost.grid_dict.objective)

        self.lgbm_random_state = config.classifierparam.light_gbm.random_state
        self.lgbm_n_estimators = config.classifierparam.light_gbm.n_estimators
        self.lgbm_lr = config.classifierparam.light_gbm.learning_rate
        self.lgbm_objective = config.classifierparam.light_gbm.objective
        self.lgbm_numleaves = config.classifierparam.light_gbm.num_leaves
        self.lgbm_param_grid_random_state = list(config.classifierparam.light_gbm.grid_dict.random_state)
        self.lgbm_param_grid_n_estimators = list(config.classifierparam.light_gbm.grid_dict.n_estimators)
        self.lgbm_param_grid_learning_rate = list(config.classifierparam.light_gbm.grid_dict.learning_rate)
        self.lgbm_param_grid_objective = list(config.classifierparam.light_gbm.grid_dict.objective)
        self.lgbm_param_grid_num_leaves = list(config.classifierparam.light_gbm.grid_dict.num_leaves)

        self.saveclassifier_ridge = config.parameters.bool.saving_classifiers.ridge
        self.saveclassifier_lr = config.parameters.bool.saving_classifiers.logistic_regression
        self.saveclassifier_forest = config.parameters.bool.saving_classifiers.random_forest
        self.saveclassifier_xgboost = config.parameters.bool.saving_classifiers.xgboost
        self.saveclassifier_lgbm = config.parameters.bool.saving_classifiers.light_gbm 



