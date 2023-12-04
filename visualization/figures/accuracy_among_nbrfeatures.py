#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import numpy as np
import matplotlib.pyplot as plt
import yaml
from attrdictionary import AttrDict as attributedict


#############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.classification_evaluation


#############################################################
## PLot curves
#############################################################

path2vectors = pathtofolder + '/TestofKs/' + 'AllCohorts/'
ext = '.npy'

# Load vectors from files
xgbmean_aAcc_mrmr = np.load(path2vectors + 'xgbmean_aAcc_mrmr' + ext)
xgbmean_aAcc_mannwhitneyu = np.load(path2vectors + 'xgbmean_aAcc_mannwhitneyu' + ext)
xgbmean_aAcc_boruta = np.load(path2vectors + 'xgbmean_aAcc_boruta' + ext)
xgbbestsplit_aAcc_mrmr = np.load(path2vectors + 'xgbbestsplit_aAcc_mrmr' + ext)
xgbbestsplit_aAcc_mannwhitneyu = np.load(path2vectors + 'xgbbestsplit_aAcc_mannwhitneyu' + ext)
xgbbestsplit_aAcc_boruta = np.load(path2vectors + 'xgbbestsplit_aAcc_boruta' + ext)
lgbmmean_aAcc_mrmr = np.load(path2vectors + 'lgbmmean_aAcc_mrmr' + ext)
lgbmmean_aAcc_mannwhitneyu = np.load(path2vectors + 'lgbmmean_aAcc_mannwhitneyu' + ext)
lgbmmean_aAcc_boruta = np.load(path2vectors + 'lgbmmean_aAcc_boruta' + ext)
lgbmbestsplit_aAcc_mrmr = np.load(path2vectors + 'lgbmbestsplit_aAcc_mrmr' + ext)
lgbmbestsplit_aAcc_mannwhitneyu = np.load(path2vectors + 'lgbmbestsplit_aAcc_mannwhitneyu' + ext)
lgbmbestsplit_aAcc_boruta = np.load(path2vectors + 'lgbmbestsplit_aAcc_boruta' + ext)


# Creating x coordinates for mrmr and mannwhtneyu
x = np.linspace(56, 1, len(xgbmean_aAcc_mrmr))

# Creating x coordinats for 
xboruta = np.load(path2vectors + 'nbr_keptfeat_list' + ext)


## PLot figure for xgboost
# Increase the figure width to make room for the legend
#plt.figure(figsize=(18, 8))
plt.figure(figsize=(6, 4))

# First figure with xgboost
plt.plot(x, xgbmean_aAcc_mrmr, label='cvmean_mrmr', color='darkblue')
plt.plot(x, xgbbestsplit_aAcc_mrmr, label='bestsplit_mrmr', color='lightskyblue')
plt.plot(x, xgbmean_aAcc_mannwhitneyu, label='cvmean_mannwhitney', color='darkgreen')
plt.plot(x, xgbbestsplit_aAcc_mannwhitneyu, label='bestsplit_mannwhitney', color='lightgreen')
plt.plot(xboruta, xgbmean_aAcc_boruta, label='cvmean_boruta', color='darkorange')
plt.plot(xboruta, xgbbestsplit_aAcc_boruta, label='bestsplit_boruta', color='wheat')
plt.xlim(max(x), min(x))

# Plot random classification binary accuracy
plt.axhline(y=0.5, color='black', linestyle='--', label='Random binary accuracy')

# Add labels and a legend
plt.xlabel('Number of feature kept')
plt.ylabel('Balanced Accuracy of Classification')
plt.title('Effect of number of kept features on xgboost balanced accuracy')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Set the step on the y-axis
plt.yticks(np.arange(0.45, 0.95, 0.05))  # Adjust the values as needed
plt.ylim(bottom=0.45, top=0.92)  # Weirdly we also need this line to have the 2 spots matching

# Save the plot on the root classification_evaluation directory
plt.savefig(pathtofolder + 'xgboost_result.png')
plt.clf()


## PLot figure for lgbm
plt.figure(figsize=(6, 4))

# First figure with xgboost
plt.plot(x, lgbmmean_aAcc_mrmr, label='cvmean_mrmr', color='darkblue')
plt.plot(x, lgbmbestsplit_aAcc_mrmr, label='bestsplit_mrmr', color='lightskyblue')
plt.plot(x, lgbmmean_aAcc_mannwhitneyu, label='cvmean_mannwhitney', color='darkgreen')
plt.plot(x, lgbmbestsplit_aAcc_mannwhitneyu, label='bestsplit_mannwhitney', color='lightgreen')
plt.plot(xboruta, lgbmmean_aAcc_boruta, label='cvmean_boruta', color='darkorange')
plt.plot(xboruta, lgbmbestsplit_aAcc_boruta, label='bestsplit_boruta', color='wheat')
plt.xlim(max(x), min(x))

# Plot random classification binary accuracy
plt.axhline(y=0.5, color='black', linestyle='--', label='Random binary accuracy')

# Add labels and a legend
plt.xlabel('Number of feature kept')
plt.ylabel('Balanced Accuracy of Classification')
plt.title('Effect of number of kept features on lgbm balanced accuracy')
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Set the step on the y-axis
plt.yticks(np.arange(0.45, 0.95, 0.05))  # Adjust the values as needed
plt.ylim(bottom=0.45, top=0.92)  # Weirdly we also need this line to have the 2 spots matching

# Save the plot on the root classification_evaluation directory
plt.savefig(pathtofolder + 'lgbm_result.png')
