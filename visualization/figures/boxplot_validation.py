#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import os

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotnine
import seaborn as sns
import yaml
from attrdictionary import AttrDict as attributedict
from plotnine import ggplot, aes, geom_boxplot, xlab, ylab, labs, theme, \
                    element_text, geom_density, scale_color_manual, scale_fill_manual

from src.histo_miner.utils.misc import convert_flatten, convert_flatten_redundant, rename_with_ancestors
from src.histo_miner.feature_selection import SelectedFeaturesMatrix



#############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtoworkfolder = config.paths.folders.feature_selection_main
pathtosavefolder = config.paths.folders.visualizations
path_exjson = config.paths.files.example_json
redundant_feat_names = list(config.parameters.lists.redundant_feat_names)


boxplots = config.parameters.bool.plot.boxplots
distributions = config.parameters.bool.plot.distributions
violinplots = config.parameters.bool.plot.violinplots
pca = config.parameters.bool.plot.pca
tsne = config.parameters.bool.plot.tsne
delete_outliers = config.parameters.bool.plot.delete_outliers


#############################################################
## Load feature matrix and classification array, feat names
#############################################################


featarray_name = 'perfile_stromal_val_featarraynbr'
classarray_name = 'perfile_stromal_val_clarraynbr'
ext = '.npy'

featarray = np.load(pathtoworkfolder + featarray_name + ext)
clarray = np.load(pathtoworkfolder + classarray_name + ext)
clarray_list = list(clarray)

clarray_names = ['-' if value == 0 
        else '+' if value == 1 
        else '++/+++' 
        for value in clarray_list]


# find index of selected features for viso:
# visu_featnames = ['Morphology_insideTumor_Granulocyte_areas_mean']
visu_featnames = ['H&Evalidation']

celltype = 'Stromal cells'
staintype = 'CD10 Quantification'

# visu_featnames = ['Lymphocytes_Pourcentage']

     
# visu_indexes = [featnames.index(feat) for feat in visu_featnames]


#############################################################
## Color palette
#############################################################


yellowscc = '#FFE100'
redscc = '#FF0100'   #(255, 1, 0)
orangescc = '#FFB011' #(255, 176, 17) 
greenscc = '#14E914' #(20, 233, 20)
lightbluescc = '#0EF2F6' #(14, 242, 246)
darkbluescc = '#1005F1' #(16, 5, 241)


# Define custom colors for each class
custom_palette = {
    '-': darkbluescc,  
    '+': darkbluescc,
    '++/+++': darkbluescc
}

# Map display names
display_labels = {0: '-', 1: '+', 2: '++/+++'}



#############################################################
## Plot boxplots for every features 
#############################################################

featvals = featarray
featname = celltype

df = pd.DataFrame({
    celltype: featvals,
    'Classification': clarray_names,
    staintype: ''
})

hue_order = ['-', '+', '++/+++']

plt.figure(figsize=(8, 10))
ax = sns.boxplot(x=staintype, 
                 y=celltype, 
                 hue='Classification', 
                 data=df, 
                 palette=custom_palette, 
                 hue_order=hue_order, 
                 fill=False,
                 dodge=True,
                 gap = 0.25,
                 showfliers=False  # Remove the outlier markers (white dot)
                 )


# Overlay individual sample points
sns.stripplot(
    x=staintype,
    y=celltype,
    hue='Classification',
    data=df,
    palette=custom_palette,
    hue_order=hue_order,
    dodge=True,
    size=10,  # Adjust dot size
    ax=ax,
    legend=False  # Exclude stripplot from the legend
)


ax.set_ylabel(celltype, fontsize=18)
ax.set_xlabel(staintype, fontsize=18)


# Save the plot
savename = featname + '_boxplot.svg' 
saveboxplot_path = os.path.join(pathtosavefolder, savename)
os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
plt.tight_layout()
plt.savefig(saveboxplot_path, format='svg', dpi=300)
plt.close()



#############################################################
## Plot Violin plots for every features (with plotnine) - NOT USED
#############################################################



#     for featindex in tqdm(range(0, len(featnames))):
#         featvals = featarray[featindex, :]
#         featname = featnames[featindex]

#         df = pd.DataFrame({
#             'FeatureValues': featvals, 
#             'Classification': clarray_names,
#             'FeatureName': [featname] * len(featvals)
#         })

#         plt.figure(figsize=(10, 8))
#         ax = sns.violinplot(x='FeatureName', 
#                             y='FeatureValues', 
#                             hue='Classification', 
#                             data=df, 
#                             palette=custom_palette, 
#                             hue_order=['response', 'no_response'], 
#                             dodge=True)

#         # Update legend labels
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles, [display_labels[label] for label in labels], loc='upper right')

#         # Add statistical annotation
#         if 'no_response' in df['Classification'].values and 'response' in df['Classification'].values:
#             non_responder_vals = df[df['Classification'] == 'no_response']['FeatureValues']
#             responder_vals = df[df['Classification'] == 'response']['FeatureValues']
#             p_value = ttest_ind(non_responder_vals, responder_vals).pvalue
#             y_max = df['FeatureValues'].max()
#             # Adjust x positions based on violin plot offsets
#             add_stat_annotation(ax, 0, 0.1, y_max + 0.05 * y_max, p_value)

#         ax.set_title(featname, fontsize=12)
#         ax.set_ylabel('Feature Values', fontsize=10)

#         # Save the plot
#         savename = featname + '_violinplot.png'
#         saveboxplot_path = os.path.join(pathtosavefolder, 'violinplots', 'allfeat', savename)
#         os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
#         plt.tight_layout()
#         plt.savefig(saveboxplot_path, dpi=300)
#         plt.close()



#