#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import os

from tqdm import tqdm
import yaml
import json
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from attrdictionary import AttrDict as attributedict
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from src.histo_miner.utils.misc import convert_flatten, convert_flatten_redundant




# create the heatmap


#############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
featselfolder = config.paths.folders.feature_selection_main
pathtosavefolder = config.paths.folders.visualizations
example_json = config.paths.files.example_json



#############################################################
## Load correlation matrix and feature names
#############################################################

matrix_relative_path = '/correlation_matrix.npy'

#load correlation matrix
corrmat = np.load(featselfolder + matrix_relative_path)

pathfeatnames = featselfolder + 'featnames' + '.npy'
featnames = np.load(pathfeatnames)
featnameslist = list(featnames)


#############################################################
## Hierarchical Clustering and Heatmaps
#############################################################


### Correlation Matrix plot

# Create a pandas frame for the correlation matrix and add the feature names for the axis 
panda_corrmat = pd.DataFrame(corrmat, columns=featnames, index=featnames)


## Plot figure
plt.figure(figsize=(35,30))
sns.heatmap(panda_corrmat,  annot_kws={"size": 7}, cmap='RdBu', annot=False, vmin=-1, vmax=1);
plt = plt.gcf()
#round(panda_corrmat,2)


## Save figure 
#Create Name for saving
savename = 'Corr_matrix_All_Cohorts_35_30.png'

#Saving
if not os.path.exists(pathtosavefolder + '/CorrMatrix/'):
    os.makedirs(pathtosavefolder + '/CorrMatrix/')
savedfig_path = pathtosavefolder + '/CorrMatrix/' + savename
plt.savefig(savedfig_path)
plt.clf()



### Hierarchical Clustering plots

## Plot the clustered correlation matrix  


dissimilarity = 1 - abs(panda_corrmat)

Z = linkage(squareform(dissimilarity, checks=False), 'complete')


# Clusterize the data
threshold = 0.8
labels = fcluster(Z, threshold, criterion='distance')

# Show the cluster
labels

# Keep the indices to sort labels
labels_order = np.argsort(labels)

# Build a new dataframe with the sorted columns
for idx, i in enumerate(panda_corrmat.columns[labels_order]):
    if idx == 0:
        clustered = pd.DataFrame(panda_corrmat[i])
    else:
        df_to_append = pd.DataFrame(panda_corrmat[i])
        clustered = pd.concat([clustered, df_to_append], axis=1)

### Plot figure
plt.figure(figsize=(35,30))
correlations = clustered.corr()
sns.heatmap(round(correlations,2), cmap='RdBu', annot=False, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);


## Save figure 
#Create Name for saving
savename = 'Corr_matrix_clustered_35_30.png'

#Saving
if not os.path.exists(pathtosavefolder + '/CorrMatrix/'):
    os.makedirs(pathtosavefolder + '/CorrMatrix/')
savedfig_path = pathtosavefolder + '/CorrMatrix/' + savename
plt.savefig(savedfig_path)
plt.clf()



## Plot correlation heatmats with dendrograms.

# plt.figure(figsize=(20,12))
sns.clustermap(correlations, method="complete", cmap='RdBu', annot=False, 
               vmin=-1, vmax=1, figsize=(35,30));


## Save figure 
#Create Name for saving
savename = 'Corr_matrix_clustered_with_dendrograms_35_30.png'

#Saving
if not os.path.exists(pathtosavefolder + '/CorrMatrix/'):
    os.makedirs(pathtosavefolder + '/CorrMatrix/')
savedfig_path = pathtosavefolder + '/CorrMatrix/' + savename
plt.savefig(savedfig_path)
plt.clf()





