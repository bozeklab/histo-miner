#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import json
import os

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

from src.histo_miner.utils.misc import convert_flatten, convert_flatten_redundant
 
from src.histo_miner.feature_selection import SelectedFeaturesMatrix

# - Plot the correlation matrix in a nice way (seaborn?). In a first step it could stay as just the 56 features. Later-on maybe only display few names or few features, the most interesting ones
# --> maybe load it here and then make it nicer with seaborn library

### FOR NOW ONLY BORUTA SELECTED ONES


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

boxplots = config.parameters.bool.plot.boxplots
distributions = config.parameters.bool.plot.distributions
pca = config.parameters.bool.plot.pca
tsne = config.parameters.bool.plot.tsne
delete_outliers = config.parameters.bool.plot.delete_outliers



#############################################################
## Load feature matrix and classification array, feat names
#############################################################


featarray_name = 'repslidesx_selectfeat'
classarray_name = 'repslidesx_clarray'
ext = '.npy'

featarray = np.load(pathtoworkfolder + featarray_name + ext)
clarray = np.load(pathtoworkfolder + classarray_name + ext)
clarray_list = list(clarray)

clarray_names = ['no_recurrence' if value == 0 else 'recurrence' for value in clarray_list]


# We can create the list of feature name just by reading on jsonfile
with open(path_exjson, 'r') as filename:
    analysisdata = filename.read()
    analysisdata = json.loads(analysisdata)
    # flatten the dict (with redundant keys in nested dict, see function)
    # Convert flatten allow to have only the last snested key as name!
    analysisdataflat = convert_flatten(analysisdata)

featnames = list(analysisdataflat.keys())



############################################################
## Load the selected features only
############################################################

#### Parse the featarray to the class SelectedFeaturesMatrix 

selection_idx_name = 'all_borutas/selfeat_boruta_idx_depth20'
selfeat = np.load(pathtoworkfolder + selection_idx_name + ext)
selfeat_idx_list = list(selfeat)

# # Update feature matrix - SEE HOW TO CODE THIS LATER ON
# SelectedFeaturesMatrix = SelectedFeaturesMatrix(featarray)
# featarray = SelectedFeaturesMatrix.boruta_matr(selfeat)
# featarray = np.transpose(featarray)

# #Update classification array and classification array list - SEE HOW TO CODE THIS LATER ON
# # clarray_list = [label for idx, label in enumerate(clarray_list) if idx in selfeat_idx_list ]
# # clarray = np.asarray(clarray_list)
# # clarray_names = ['no_recurrence' if value == 0 else 'recurrence' for value in clarray_list]

# Update featnames
featnames = [name for idx, name in enumerate(featnames) if idx in selfeat_idx_list]



#############################################################
## Plot boxplots for the selected features
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
            if not os.path.exists(pathtosavefolder + '/boxplots/'):
                os.makedirs(pathtosavefolder + '/boxplots/')
            saveboxplot_path = pathtosavefolder +  '/boxplots/' + savename
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
            if not os.path.exists(pathtosavefolder + '/boxplots/'):
                os.makedirs(pathtosavefolder + '/boxplots/')
            saveboxplot_path = pathtosavefolder +  '/boxplots/' + savename
            boxplot.save(saveboxplot_path, dpi=300)



#############################################################
## Plot Kernel Density distribution for the selected features
#############################################################

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
            if not os.path.exists(pathtosavefolder + '/density/'):
                os.makedirs(pathtosavefolder + '/density/')
            savedensplot_path = pathtosavefolder + '/density/' + savename
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
            if not os.path.exists(pathtosavefolder + '/density/'):
                os.makedirs(pathtosavefolder + '/density/')
            savedensplot_path = pathtosavefolder + '/density/' + savename
            density_plot.save(savedensplot_path, dpi=300)



#############################################################
## PCA and biplots
#############################################################

# See https://scikit-learn.org/ Comparison of LDA and PCA 2D projection of Iris dataset
# For an example of use

if pca: 
    #### Initialize PCA 2D
    pca = PCA(n_components=2)
    # Create vector for fit method
    X = pd.DataFrame(featarray)
    X = np.transpose(X)
    X = X.astype('float32')
    # Standardize the dataset
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Create classification target vector for visu
    target = clarray
    # Target names for visualization
    target_names = ['no_recurrence', 'recurrence']

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
    plt.title("PCA of SCC WSIs (selected features)")

    #Create Name for saving
    savename = 'PCA_SCC_WSIs_2D_selected_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()


    #### Initialize PCA 3D
    pca = PCA(n_components=3)
    # Create vector for fit method
    X = pd.DataFrame(featarray)
    X = np.transpose(X)
    X = X.astype('float32')
    # Standardize the dataset
    # Create an instance of StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Create classification target vector for visu
    target = clarray
    # Target names for visualization
    target_names = ['no_recurrence', 'recurrence']

    # PCA fitting
    pca_result = pca.fit(X_scaled).transform(X_scaled)

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
    ax.set_title("3D PCA of SCC WSIs (selected features)")

    #Create Name for saving
    savename = 'PCA_SCC_WSIs_3D_selected_features.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()

    print('PCAs saved.')


    #### Initialize Scree Plot 2D
    pca_scree = PCA(n_components=4)
    # Here 4 because we cannot have more PCA components than feat
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
    plt.title("Scree Plot of SCC WSIs (selected features)")

    #Create Name for saving
    savename = 'ScreePlot_SCC_WSIs_selected_feature.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()

    print('Scree Plot saved.')


    ### Biplot 
    # Used https://statisticsglobe.com/biplot-pca-python
    # Explanation 1 TO FILL
    # We re use the pca not to do it again for nothing (see above)
    principalc1 = pca.fit_transform(X_scaled)[:,0]
    principalc2 = pca.fit_transform(X_scaled)[:,1]
    ldngs = pca.components_
    
    # Explanation 2 TO FILL
    scale_principalc1 = 1.0/(principalc1.max() - principalc1.min())
    scale_principalc2 = 1.0/(principalc2.max() - principalc2.min())
    features = featnames
    
    # Define target groups
    target_groups = np.digitize(clarray, 
                             np.quantile(clarray, 
                                         [1/3, 2/3]))

    # Plot 
    fig, ax = plt.subplots(figsize=(14, 9))
     
    for i, feature in enumerate(features):
        ax.arrow(0, 0, ldngs[0, i], 
                 ldngs[1, i], 
                 head_width=0.01, 
                 head_length=0.01)
        ax.text(ldngs[0, i] * 1.15, 
                ldngs[1, i] * 1.15, 
                feature, fontsize = 11)
     
    scatter = ax.scatter(principalc1 * scale_principalc1, 
                         principalc2 * scale_principalc2, 
                         c=target_groups, 
                         cmap='viridis')
     
    ax.set_xlabel('Principal Component 1', fontsize=20)
    ax.set_ylabel('Principal Component 2', fontsize=20)
    ax.set_title('Bitplot of SCC WSIs (selected features)', fontsize=20)
     
    ax.legend(*scatter.legend_elements(),
                        loc="lower left", 
                        title="Groups")

    #Create Name for saving
    savename = 'Biplot_SCC_WSIs_selected_feature.png'

    #Saving
    if not os.path.exists(pathtosavefolder + '/PCA/'):
        os.makedirs(pathtosavefolder + '/PCA/')
    savedpca_path = pathtosavefolder + '/PCA/' + savename
    plt.savefig(savedpca_path)
    plt.clf()

    print('Biplot saved.')


#############################################################
## T-SNE plots
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
    target_names = ['no_recurrence', 'recurrence']

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
        ).set(title="SCC data T-SNE projection (selected features)")
    plt = plt.gcf()

    #Create Name for saving
    savename = 'T-SNE_SCC_WSIs_2D_selected_features.png'

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