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
pathtoworkfolder = config.paths.folders.feature_selection_main
pathtosavefolder = config.paths.folders.visualizations
path_exjson = config.paths.files.example_json
redundant_feat_names = list(config.parameters.lists.redundant_feat_names)


boxplots = config.parameters.bool.plot.boxplots
distributions = config.parameters.bool.plot.distributions
pca = config.parameters.bool.plot.pca
tsne = config.parameters.bool.plot.tsne
delete_outliers = config.parameters.bool.plot.delete_outliers



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



#############################################################
## Plot boxplots for every features (with plotnine)
#############################################################

if boxplots:
    if delete_outliers:
        # Filter extremes quartiles
        for featindex in tqdm(range(0, len(featnames))):
            pourcentagerem = 0.1
            featvals = featarray[featindex,:]

            # Get the indices of the values to keep
            indices_to_keep = np.where(
                ((featvals > np.quantile(featvals, (pourcentagerem / 2)))
                 & (featvals < np.quantile(featvals, (1 - pourcentagerem / 2))))
                )[0]
            # Remove outliers from features vectors
            featvals_wooutliers =  featvals[indices_to_keep]
            # Remove corresponding classifications
            clarray_names_wooutliers = [clarray_names[i] for i in indices_to_keep]
            
            # Extract name of the feature
            featname = featnames[featindex]
            #Create a pandas data frame from these vectors
            df = pd.DataFrame( {'FeatureValues':featvals_wooutliers, 
                                'WSIClassification':clarray_names_wooutliers})
            #PLot the corresponding boxplot
            boxplot = (ggplot(df, aes(x='WSIClassification', y='FeatureValues')) 
                        + geom_boxplot()
                        + xlab("Whole Slide Image Classification")
                        + ylab("Feature Values (removed {}% outliers)".format(pourcentagerem*100))
                        + labs(title= featname)
                        + theme(plot_title=element_text(size=10))
                        # plotnine.ggtitle(wrapper(featname, width = 20))
                        )
            #Create Name for saving
            savename = featname + '_boxplot_filterquartile.png'

            #Saving
            if not os.path.exists(pathtosavefolder + '/boxplots/allfeat/'):
                os.makedirs(pathtosavefolder + '/boxplots/allfeat/')
            saveboxplot_path = pathtosavefolder +  '/boxplots/allfeat/' + savename
            boxplot.save(saveboxplot_path, dpi=300)
            # Filter outliers using Piercon Crriterion is also an option
    else:
        for featindex in tqdm(range(0, len(featnames))):
            featvals = featarray[featindex,:]
            featname = featnames[featindex]
            #Create a pandas data frame from these vectors
            df = pd.DataFrame( {'FeatureValues':featvals, 'WSIClassification':clarray_names})
            #PLot the corresponding boxplot
            boxplot = (ggplot(df, aes(x='WSIClassification', y='FeatureValues')) 
                        + geom_boxplot()
                        + xlab("Whole Slide Image Classification")
                        + ylab("Feature Values")
                        + labs(title= featname)
                        + theme(plot_title=element_text(size=10))
                        # plotnine.ggtitle(wrapper(featname, width = 20))
                        )
            #Create Name for saving
            savename = featname + '_boxplot.png'

            #Saving
            if not os.path.exists(pathtosavefolder + '/boxplots/allfeat/'):
                os.makedirs(pathtosavefolder + '/boxplots/allfeat/')
            saveboxplot_path = pathtosavefolder +  '/boxplots/allfeat/' + savename
            boxplot.save(saveboxplot_path, dpi=300)



########################################################################
## Plot Kernel Density distribution for every features (with plotnine)
########################################################################

if distributions:

    norec_idx = [idx for idx, value in enumerate(clarray) if value == 0]
    rec_idx = [idx for idx, value in enumerate(clarray) if value == 1]
    featarray_norec = featarray[:,norec_idx]
    featarray_rec = featarray[:,rec_idx]

    if delete_outliers:
        #set the variables
        # Filter extremes quartiles but here for both rec and no_rec vectors
        for featindex in tqdm(range(0, len(featnames))):
            pourcentagerem = 0.1
            featvals_norec = list(featarray_norec[featindex,:])
            featvals_rec = list(featarray_rec[featindex,:])

            # Get the indices of the values to keep
            indices_to_keep_norec = np.where(
                ((featvals_norec > np.quantile(featvals_norec, (pourcentagerem / 2)))
                 & (featvals_norec < np.quantile(featvals_norec, (1 - pourcentagerem / 2))))
                )[0]
            indices_to_keep_rec = np.where(
                ((featvals_rec > np.quantile(featvals_norec, (pourcentagerem / 2)))
                 & (featvals_rec < np.quantile(featvals_norec, (1 - pourcentagerem / 2))))
                )[0]
            # Remove outliers from features vectors
            featvals_wooutliers_norec =  [featvals_norec[i] for i in indices_to_keep_norec]
            featvals_wooutliers_rec =  [featvals_rec[i] for i in indices_to_keep_rec]
            
            # Extract name of the feature
            featname = featnames[featindex]
            # Create a pandas DataFrame
            df = pd.DataFrame({'FeatureValues': featvals_wooutliers_norec + featvals_wooutliers_rec,
                'Distribution': ['featvals_wooutliers_norec'] * len(featvals_wooutliers_norec) + \
                                ['featvals_wooutliers_rec'] * len(featvals_wooutliers_rec)})

            # Define colors for each distribution
            # It looks like the palette is the one of matplolib 3
            colors = {'featvals_wooutliers_norec': 'royalblue', 
                      'featvals_wooutliers_rec': 'orangered'}

            # Plot kernel density distributions for both vectors
            density_plot = (ggplot(df, aes(x='FeatureValues', 
                                           color='Distribution', 
                                           fill='Distribution'))
                            + geom_density(alpha=0.7)
                            + scale_color_manual(values=[colors['featvals_wooutliers_norec'], 
                                                         colors['featvals_wooutliers_rec']])
                            + scale_fill_manual(values=[colors['featvals_wooutliers_norec'], 
                                                        colors['featvals_wooutliers_rec']])
                            + xlab("Density")
                            + ylab("Feature Values (removed {}% outliers for each class)"
                                   .format(pourcentagerem*100)) 
                            + labs(title= featname)
                            + theme(plot_title=plotnine.element_text(size=10))
                        )
            #Create Name for saving
            savename = featname + '_distribution_filterquartile.png'

            #Saving
            if not os.path.exists(pathtosavefolder + '/density/allfeat/'):
                os.makedirs(pathtosavefolder + '/density/allfeat/')
            savedensplot_path = pathtosavefolder + '/density/allfeat/' + savename
            density_plot.save(savedensplot_path, dpi=300)
            # Filter outliers using Piercon Crriterion is also an option
    else:
        for featindex in tqdm(range(0, len(featnames))):
            featvals_norec = list(featarray_norec[featindex,:])
            featvals_rec = list(featarray_rec[featindex,:])
            featname = featnames[featindex]
            # Create a pandas DataFrame
            df = pd.DataFrame({'FeatureValues': featvals_norec + featvals_rec,
                'Distribution': ['featvals_norec'] * len(featvals_norec) + ['featvals_rec'] * len(featvals_rec)})

            # Define colors for each distribution
            # It looks like the palette is the one of matplolib 3
            colors = {'featvals_norec': 'royalblue', 'featvals_rec': 'orangered'}

            # Plot kernel density distributions for both vectors
            density_plot = (ggplot(df, aes(x='FeatureValues', 
                                           color='Distribution', 
                                           fill='Distribution'))
                            + geom_density(alpha=0.7)
                            + scale_color_manual(values=[colors['featvals_norec'], 
                                                         colors['featvals_rec']])
                            + scale_fill_manual(values=[colors['featvals_norec'], 
                                                        colors['featvals_rec']])
                            + xlab("Density")
                            + ylab("Feature Values")
                            + labs(title= featname)
                            + theme(plot_title=plotnine.element_text(size=10))
                        )
            #Create Name for saving
            savename = featname + '_distribution.png'

            #Saving
            if not os.path.exists(pathtosavefolder + '/density/allfeat/'):
                os.makedirs(pathtosavefolder + '/density/allfeat/')
            savedensplot_path = pathtosavefolder + '/density/allfeat/' + savename
            density_plot.save(savedensplot_path, dpi=300)



#############################################################
## PCA and scree plots (with plt ax.scatter and plot)
#############################################################

# See https://scikit-learn.org/ Comparison of LDA and PCA 2D projection of Iris dataset
# For an example of use

if pca: 
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
    target = clarray
    # Target names for visualization
    target_names = ['no_response', 'response']

    # PCA fitting
    pca_result = pca.fit(X_scaled).transform(X_scaled)

    # 2D PCA plot
    fig, ax = plt.subplots()
    # interesting colors = ["navy", "darkorange"]
    colors = ["royalblue", "orangered"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(
            pca_result[target == i, 0], 
            pca_result[target == i, 1], 
            color=color, 
            alpha=0.8, 
            lw=lw, 
            label=target_name
        )
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of SCC WSIs (all features kept)")

    #Create Name for saving
    savename = 'PCA_SCC_WSIs_2D_all_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()


    #### PCA 3D
    pca3D = PCA(n_components=3)
    # Create vector for fit method
    X = pd.DataFrame(featarray)
    X = np.transpose(X)
    # X = X.astype('float32')
    # Standardize the dataset
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Create classification target vector for visu
    target = clarray
    # Target names for visualization
    target_names = ['no_response', 'response']

    # PCA fitting
    pca_result = pca3D.fit(X_scaled).transform(X_scaled)

    # 3D PCA plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # interesting colors = ["navy", "darkorange"]
    colors = ["royalblue", "orangered"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):
        ax.scatter(
            pca_result[target == i, 0], 
            pca_result[target == i, 1], 
            color=color, 
            alpha=0.8, 
            lw=lw, 
            label=target_name
        )
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend(loc="best", shadow=False, scatterpoints=1)
    ax.set_title("3D PCA of SCC WSIs (all features kept)")

    #Create Name for saving
    savename = 'PCA_SCC_WSIs_3D_all_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()

    print('PCAs saved.')


    #### Scree Plot 2D
    pca_scree = PCA(n_components=20)
    # We need to fit but not to fit + transform!
    # Plus we need more components then 2 or 3
    pca2_result = pca_scree.fit(X_scaled)
    
    # Scree plot
    PC_values = np.arange(pca_scree.n_components_) + 1
    plt.plot(
        PC_values, 
        pca_scree.explained_variance_ratio_, 
        'o-', 
        linewidth=2, 
        color='royalblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title("Scree Plot of SCC WSIs (all features kept)")

    #Create Name for saving
    savename = 'ScreePlot_SCC_WSIs_all_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()

    print('Scree Plot saved.')


#############################################################
## T-SNE plots (with seaborn)
#############################################################

# Will tryr to follow what was done just above 

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
    # Create classification target vector for visu
    target = pd.Series(clarray)
    target = target.astype('int8')
    # Target names for visualization
    target_names = ['no_response', 'response']

    # TSNE fitting
    z = tsne.fit_transform(X_scaled)

    # interesting colors = ["navy", "darkorange"]
    colors = ["royalblue", "orangered"]

    # T-SNE 2D plot
    df = pd.DataFrame()
    df["y"] = target
    df["t-sne 1"] = z[:,0]
    df["t-sne 2"] = z[:,1]

    ax = sns.scatterplot(
        x="t-sne 1", 
        y="t-sne 2", 
        hue=df.y.tolist(),
        #palette=sns.color_palette("hls", 2),
        palette=colors,
        data=df
        ).set(title="SCC data T-SNE projection (all features kept)")
    plt = plt.gcf()

    #Create Name for saving
    savename = 'T-SNE_SCC_WSIs_2D_all_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/TSNE/'):
        os.makedirs(pathtosavefolder + '/TSNE/')
    savedtsne_path = pathtosavefolder + '/TSNE/' + savename
    plt.savefig(savedtsne_path)
    plt.clf()


    #### Initialize TSNE 3D

    # Explanation here
    # https://innovationyourself.com/3d-data-visualization-seaborn-in-python/
    # https://seaborn.pydata.org/generated/seaborn.scatterplot.html
   

    # tsne = TSNE(n_components=3, verbose=0, random_state=42)
    # # Create vector for fit method
    # X = pd.DataFrame(featarray)
    # X = np.transpose(X)
    # X = X.astype('float32')
    # # Standardize the dataset
    # # Create an instance of StandardScaler
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # # Create classification target vector for visu
    # target = pd.Series(clarray)
    # target = target.astype('int8')
    # # Target names for visualization
    # target_names = ['no_recurrence', 'recurrence']

    # # TSNE fitting
    # z = tsne.fit_transform(X_scaled)

    # # interesting colors = ["navy", "darkorange"]
    # colors = ["royalblue", "orangered"]

    # # T-SNE 3D plot
    # df = pd.DataFrame()
    # df["y"] = target
    # df["t-sne 1"] = z[:,0]
    # df["t-sne 2"] = z[:,1]
    # df["t-sne 3"] = z[:,2]

    # ax = sns.scatterplot(
    #     x="t-sne 1", 
    #     y="t-sne 2", 
    #     z="t-sne 3", 
    #     hue=df.y.tolist(),
    #     #palette=sns.color_palette("hls", 2),
    #     palette=colors,
    #     data=df
    #     ).set(title="SCC data T-SNE projection")
    # plt = plt.gcf()

    # #Create Name for saving
    # savename = 'T-SNE_SCC_WSIs_3D.png'

    # #Saving
    # if not os.path.exists(pathtosavefolder + '/TSNE/'):
    #     os.makedirs(pathtosavefolder + '/TSNE/')
    # savedtsne_path = pathtosavefolder + '/TSNE/' + savename
    # plt.savefig(savedtsne_path)

    print('T-SNE saved.')



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
    # for i, name in enumerate(sample_names):
    #     plt.text(pca_result[i, 0], pca_result[i, 1], name, fontsize=8, ha='right', color='white')

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
    savename = 'PCA_SCC_WSIs_cohorts_coloring_2D_all_features.png'

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
    # for i, name in enumerate(sample_names):
    #     plt.text(pca_result[i, 0], pca_result[i, 1], name, fontsize=8, ha='right', color='white')

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



