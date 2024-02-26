#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import os
import numpy as np
import yaml
from attrdictionary import AttrDict as attributedict

from src.histo_miner.feature_selection import SelectedFeaturesMatrix
import joblib


# add the script to transform mat files into npy for instances and npy for types (using hovernet utils)



