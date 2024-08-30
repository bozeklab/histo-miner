#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import os


import yaml
import numpy as np
from attrdictionary import AttrDict as attributedict
from src.histo_miner.evaluations import plot_conf_matrix

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
confighm = attributedict(config)
pathtosavefolder = confighm.paths.folders.visualizations


with open("./../../configs/models/eval_hovernet.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
conf_matrices_folder = config.eval_paths.eval_folders.save_folder


conf_mat_name = '/confusion_matrix.npy'
conf_mat_name_truenorm = '/confusion_matrix_truenorm.npy'
conf_mat_name_prednorm = '/confusion_matrix_prednorm.npy'


conf_mat = np.load(conf_matrices_folder + conf_mat_name)
conf_mat_truenorm = np.load(conf_matrices_folder + conf_mat_name_truenorm)
conf_mat_prednorm = np.load(conf_matrices_folder + conf_mat_name_prednorm)

plot_conf_matrix(
    conf_mat, 
    conf_mat_truenorm, 
    conf_mat_prednorm, 
    pathtosavefolder
    )









