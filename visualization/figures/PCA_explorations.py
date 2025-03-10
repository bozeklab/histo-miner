#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import os

import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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
pathtoworkfolder = config.paths.folders.featarray_folder
pathtosavefolder = config.paths.folders.visualizations
path_exjson = config.paths.files.example_json
redundant_feat_names = list(config.parameters.lists.redundant_feat_names)


boxplots = config.parameters.bool.plot.boxplots
distributions = config.parameters.bool.plot.distributions
pca = config.parameters.bool.plot.pca
tsne = config.parameters.bool.plot.tsne
delete_outliers = config.parameters.bool.plot.delete_outliers


if not os.path.exists(pathtosavefolder):
    os.mkdir(pathtosavefolder)

    
#############################################################
## Load feature matrix and classification array, feat names
#############################################################


featarray_name = 'perwsi_featarray'
classarray_name = 'perwsi_clarray'
ext = '.npy'

featarray = np.load(pathtoworkfolder + featarray_name + ext)
clarray = np.load(pathtoworkfolder + classarray_name + ext)
clarray_list = list(clarray)

clarray_names = ['no_response' if value == 0 else 'response' for value in clarray_list]

# We can create the list of feature name just by reading on jsonfile
with open(path_exjson, 'r') as filename:
    analysisdata = filename.read()
    analysisdata = json.loads(analysisdata)

    #Be carefukl in the redundancy we have in the areas, circularity, aspect ratio and dis features
    renamed_analysisdata = rename_with_ancestors(analysisdata, redundant_feat_names)

    #Check the difference between convert_flatten and convert_flatten_redundant docstrings
    simplifieddata =  convert_flatten(renamed_analysisdata)


featnames = list(simplifieddata.keys())




############################################################
## PCA and scree plots for cohorts and samples vizualisation
## CPI data usecase 
#############################################################

# See https://scikit-learn.org/ Comparison of LDA and PCA 2D projection of Iris dataset
# For an example of use

if pca: 
    #### PCA 2D 
    pca = PCA(n_components=2)
    # Create vector for fit method
    # X = pd.DataFrame(featarray)
    # X = np.transpose(X)
    # # X = X.astype('float32')
    # # Standardize the dataset
    # # Create an instance of StandardScaler
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # Create classification target vector for visu
    # here I can just hard code the cohorts by checking the order of each sample inside feat array

    # Also check if I can add an ID for the samples ato display as a name in addition to the color corresponding to the cohort

    target = [
        1,
        1,
        0,
        1,
        0,
        3,
        2,
        2,
        5,
        5, 
        4,
        5,
        4,
        7, 
        7,
        7,
        2, 
        2,
        2,
        3,
        2,
        2,
        2, 
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        5,
        5,
        9,
        9,
        9,
        8,
        8,
        8
    ] # List of Class of cohorts from 0 to 9
    # AS it is EXPLORATOY, the list will be hardcoded
    target = np.asarray(target)

    sample_names = [
        'S03604_03',
        'S03605_01',
        'S03611_03',
        'S03612_03',
        'S03613_01',
        'S03614_01', 
        'S03615_01', 
        'S03616_01',
        'S03622_01', 
        'S03623_01', 
        'S03625_03', 
        'S03627_02',
        'S03628_03',
        'S03631_01', 
        'S03633_02', 
        'S03634_02', 
        'S03637_01', 
        'S03638_02', 
        'S03639_01',
        'S03640_01', 
        'S03641_01',
        'S03642_03',
        'S03643_01', 
        'S03644_02', #23 
        'S03645_01', 
        'S03646_02', 
        'S03647_02',
        'S03651_03', 
        'S03652_01', 
        'S03654_01', 
        'S03655_01', 
        'S03656_02',
        'S03785_01', 
        'S03786_01', 
        'S03789_01', #34 
        'TUM16', 
        'TUM17',
        'TUM18'
    ] # Name of samples to be displayed  
    # AS it is EXPLORATOY, the list will be hardcoded


    # Target names for visualization
    target_names = [
        'Salzburg-resp',    #0
        'Salzburg-dis',   #1
        'Dortmund-resp',    #2
        'Dortmund-dis',  #3
        'Cologne-resp',     #4
        'Cologne-dis',   #5
        'Oberhausen-resp',  #6
        'Oberhausen-dis',#7
        'Munich-resp',      #8
        'Munich-dis'    #9
        ]

    # Keep given cohort information and remove the rest
    new_target = [value for  value in target if value == 8 or value == 9]
    new_target_idx = [index for index, value in enumerate(target) if value == 8 or value == 9]
    new_sample_names = [sample_names[i] for i in new_target_idx]
    new_target_names = [target_names[8], target_names[9]]
 
    X = pd.DataFrame(featarray)

    # select a givin cohort
    X = X[new_target_idx]
    
    # here or after transpose we could select one cohort excusively, see with pudb
    X = np.transpose(X)


    # X = X.astype('float32')
    # Standardize the dataset
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # PCA fitting
    pca_result = pca.fit(X_scaled).transform(X_scaled)

    # 2D PCA plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
    # interesting colors = ["navy", "darkorange"]
    colors = [
        'navy', 
        'skyblue', 
        'darkgreen', 
        'palegreen', 
        'firebrick', 
        'lightcoral', 
        'indigo', 
        'plum', 
        'goldenrod', 
        'yellow'
        ]

    lw = 2

    # Set background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    #new colors
    new_colors = [colors[8], colors[9]]

    for color, target_value, target_name in zip(new_colors, [8, 9], new_target_names):
        mask = np.array(new_target) == target_value
        plt.scatter(
            pca_result[mask == 0, 0], 
            pca_result[mask == 1, 1], 
            color=color, 
            alpha=0.8, 
            lw=lw, 
            label=target_name
        )

    # Plot each cohort group and label points
    # for i in range(len(new_colors)):
    #     mask = target == i
    #     plt.scatter(
    #         pca_result[mask, 0], 
    #         pca_result[mask, 1], 
    #         color=colors[i], 
    #         alpha=0.8, 
    #         lw=lw, 
    #         label=target_names[i] if i < len(target_names) else f'Cohort {i}'
    #     )

    # Add sample names as labels with white color
    for i, name in enumerate(new_sample_names):
        plt.text(pca_result[i, 0], pca_result[i, 1], name, fontsize=5, ha='right', color='white')

    # Set axis labels and title with white color
    ax.set_xlabel('Principal Component 1', color='white')
    ax.set_ylabel('Principal Component 2', color='white')
    plt.title("PCA of SCC WSIs (all features kept)", color='white')

    # Set axis ticks and lines to white
    ax.tick_params(axis='both', colors='white')  # Ticks and labels
    for spine in ax.spines.values():
        spine.set_edgecolor('white')  # Axis lines

    # Set legend with white text
    plt.legend(
        loc="upper left", 
        bbox_to_anchor=(1, 1), 
        shadow=False, 
        scatterpoints=1, 
        fontsize=10, 
        frameon=False, 
        facecolor='black', 
        edgecolor='black', 
        labelcolor='white')
    plt.title("PCA of SCC WSIs (all features kept)")

    # Adjust the layout to prevent clipping of legend and title
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    #Create Name for saving
    savename = 'PCA_SCC_WSIs_Munich _all_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()

    print('PCAs with cohorts coloring saved.')




    #### PCA 2D 
    pca = PCA(n_components=2)
    # Create vector for fit method
    X = pd.DataFrame(featarray)
    X = np.transpose(X)
    # X = X.astype('float32')
    # Standardize the dataset
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Create classification target vector for visu
    # here I can just hard code the cohorts by checking the order of each sample inside feat array

    # Also check if I can add an ID for the samples ato display as a name in addition to the color corresponding to the cohort

    target = [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2, 
        2,
        2,
        2,
        3, 
        3,
        3,
        1, 
        1,
        1,
        1,
        1,
        1,
        1, 
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
        4,
        4
    ] # List of Class of cohorts from 0 to 9
    # AS it is EXPLORATOY, the list will be hardcoded
    target = np.asarray(target)

    sample_names = [
        'S03604_03',
        'S03605_01',
        'S03611_03',
        'S03612_03',
        'S03613_01',
        'S03614_01', 
        'S03615_01', 
        'S03616_01',
        'S03622_01', 
        'S03623_01', 
        'S03625_03', 
        'S03627_02',
        'S03628_03',
        'S03631_01', 
        'S03633_02', 
        'S03634_02', 
        'S03637_01', 
        'S03638_02', 
        'S03639_01',
        'S03640_01', 
        'S03641_01',
        'S03642_03',
        'S03643_01', 
        'S03644_02', #23 
        'S03645_01', 
        'S03646_02', 
        'S03647_02',
        'S03651_03', 
        'S03652_01', 
        'S03654_01', 
        'S03655_01', 
        'S03656_02',
        'S03785_01', 
        'S03786_01', 
        'S03789_01', #34 
        'TUM16', 
        'TUM17',
        'TUM18'
    ] # Name of samples to be displayed  
    # AS it is EXPLORATOY, the list will be hardcoded


    # Target names for visualization
    target_names = [
        'Salzburg',    #0
        'Dortmund',    #2
        'Cologne',     #4
        'Oberhausen',  #6
        'Munich',      #8
        ]

    # PCA fitting
    pca_result = pca.fit(X_scaled).transform(X_scaled)

    # 2D PCA plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
    # interesting colors = ["navy", "darkorange"]
    colors = [
        'skyblue', 
        'palegreen', 
        'lightcoral', 
        'plum', 
        'yellow'
        ]

    lw = 2

    # Set background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot each cohort group and label points
    for i in range(len(colors)):
        mask = target == i
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            color=colors[i], 
            alpha=0.8, 
            lw=lw, 
            label=target_names[i] if i < len(target_names) else f'Cohort {i}'
        )

    # # Add sample names as labels with white color
    for i, name in enumerate(sample_names):
        plt.text(pca_result[i, 0], pca_result[i, 1], name, fontsize=8, ha='right', color='white')

    # Set axis labels and title with white color
    ax.set_xlabel('Principal Component 1', color='white')
    ax.set_ylabel('Principal Component 2', color='white')
    plt.title("PCA of SCC WSIs (all features kept)", color='white')

    # Set axis ticks and lines to white
    ax.tick_params(axis='both', colors='white')  # Ticks and labels
    for spine in ax.spines.values():
        spine.set_edgecolor('white')  # Axis lines

    # Set legend with white text
    plt.legend(
        loc="upper left", 
        bbox_to_anchor=(1, 1), 
        shadow=False, 
        scatterpoints=1, 
        fontsize=10, 
        frameon=False, 
        facecolor='black', 
        edgecolor='black', 
        labelcolor='white')
    plt.title("PCA of SCC WSIs (all features kept)")

    # Adjust the layout to prevent clipping of legend and title
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    #Create Name for saving
    savename = 'PCA_SCC_WSIs_cohorts_coloring2_2D_all_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()

    print('PCAs with cohorts coloring 2 saved.')




# Assuming sample_names, featarray, and other parameters are defined as in the PCA code

if tsne:
    #### Initialize TSNE 2D
    tsne = TSNE(n_components=2, verbose=0, random_state=42)
    # Create vector for fit method
    X = pd.DataFrame(featarray)
    X = np.transpose(X)
    X = X.astype('float32')
    # Standardize the dataset
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Create classification target vector for visualization
    target = np.asarray(clarray).astype(int)

    # Define sample names as in PCA example
    sample_names = [
        'S03604_03', 'S03605_01', 'S03611_03', 'S03612_03', 'S03613_01', 'S03614_01', 'S03615_01',
        'S03616_01', 'S03622_01', 'S03623_01', 'S03625_03', 'S03627_02', 'S03628_03', 'S03631_01', 
        'S03633_02', 'S03634_02', 'S03637_01', 'S03638_02', 'S03639_01', 'S03640_01', 'S03641_01', 
        'S03642_03', 'S03643_01', 'S03644_02', 'S03645_01', 'S03646_02', 'S03647_02', 'S03651_03', 
        'S03652_01', 'S03654_01', 'S03655_01', 'S03656_02', 'S03785_01', 'S03786_01', 'S03789_01', 
        'TUM16', 'TUM17', 'TUM18'
    ]

    # TSNE fitting
    z = tsne.fit_transform(X_scaled)

    # Colors for each target group
    colors = ["royalblue", "orangered"]

    # T-SNE 2D plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, color in enumerate(colors):
        mask = target == i
        ax.scatter(z[mask, 0], z[mask, 1], color=color, alpha=0.8, label=target_names[i] if i < len(target_names) else f'Group {i}')
    
    # Add sample names as labels
    for i, name in enumerate(sample_names):
        ax.text(z[i, 0], z[i, 1], name, fontsize=8, ha='right', color='white')

    # Set axis labels, title, and other plot aesthetics
    ax.set_xlabel('t-SNE Component 1', color='white')
    ax.set_ylabel('t-SNE Component 2', color='white')
    plt.title("T-SNE of SCC WSIs (all features kept)", color='white')
    ax.tick_params(axis='both', colors='white')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Legend setup
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=10, facecolor='black', labelcolor='white')

    # Adjust layout to prevent clipping
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    savename = 'T-SNE_SCC_WSIs_2D_all_features_with_names.png'
    if not os.path.exists(pathtosavefolder + '/TSNE/'):
        os.makedirs(pathtosavefolder + '/TSNE/')
    savedtsne_path = os.path.join(pathtosavefolder, 'TSNE', savename)
    plt.savefig(savedtsne_path)
    plt.clf()

    print('T-SNE with sample names saved.')


