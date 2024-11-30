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


conversion_rate = 0.053

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

# find index of selected features for viso:
visu_featnames = ['Morphology_insideTumor_Granulocyte_areas_mean']
     
visu_indexes = [featnames.index(feat) for feat in visu_featnames if feat in featnames]


#############################################################
## Functions
#############################################################


# Function to add significance bars
def add_stat_annotation(ax, x1, x2, y, p_value):
    """Annotate the plot with p-value bars and stars."""
    significance = ''
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    elif p_value >= 0.05:
        significance = 'ns'
    
    ax.plot([x1, x2], [y, y], color="black", lw=1.5)  # Add horizontal bar
    ax.text((x1 + x2) * 0.5, y, significance, ha='center', va='bottom', color="black")


#############################################################
## Plot boxplots for every features 
#############################################################


if boxplots:

    # Define custom colors for each class
    custom_palette = {
        'response': '#FF5733',  # Vibrant Orange
        'no_response': '#3498DB'  # Vibrant Sky Blue
    }

    # Map display names
    display_labels = {'response': 'Responder', 'no_response': 'Non-responder'}

    
    if delete_outliers:
        pourcentagerem = 0.1
        for featindex in tqdm(visu_indexes):
            featvals = featarray[featindex, :]
            indices_to_keep = np.where(
                ((featvals > np.quantile(featvals, pourcentagerem / 2)) &
                 (featvals < np.quantile(featvals, 1 - pourcentagerem / 2)))
            )[0]

            featvals_wooutliers = featvals[indices_to_keep]
            clarray_names_wooutliers = [clarray_names[i] for i in indices_to_keep]

            featname = featnames[featindex]
            df = pd.DataFrame({
                'FeatureValues': featvals_wooutliers,
                'Classification': clarray_names_wooutliers,
                'FeatureName': [featname] * len(featvals_wooutliers)
            })

            plt.figure(figsize=(10, 6))
            ax = sns.boxplot(x='FeatureName', y='FeatureValues', hue='Classification', 
                             data=df, palette=custom_palette, hue_order=['response', 'no_response'], dodge=True,
                             gap = 0.25)


            # Update legend labels
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [display_labels[label] for label in labels], loc='upper right')

            # Add statistical annotation
            if 'no_response' in df['Classification'].values and 'response' in df['Classification'].values:
                non_responder_vals = df[df['Classification'] == 'no_response']['FeatureValues']
                responder_vals = df[df['Classification'] == 'response']['FeatureValues']
                p_value = ttest_ind(non_responder_vals, responder_vals).pvalue
                y_max = df['FeatureValues'].max()
                add_stat_annotation(ax, 0, 0.1, y_max + 0.05 * y_max, p_value)

            ax.set_title(f'{featname} (removed {pourcentagerem * 100}% outliers)', fontsize=12)
            ax.set_ylabel('Feature Values', fontsize=10)

            # Save the plot
            savename = featname + '_boxplot_filterquartile.png'
            saveboxplot_path = os.path.join(pathtosavefolder, 'boxplots', 'allfeat', savename)
            os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(saveboxplot_path, dpi=300)
            plt.close()

    else:
        for featindex in tqdm(visu_indexes):
            featvals = featarray[featindex, :]
            featname = featnames[featindex]

            df = pd.DataFrame({
                'FeatureValues': featvals * conversion_rate, # here we can adjust if the value is in pixel squarre or in micro meter square
                'Classification': clarray_names,
                'FeatureName': ''
            })

            plt.figure(figsize=(10, 6))
            ax = sns.boxplot(x='FeatureName', 
                             y='FeatureValues', 
                             hue='Classification', 
                             data=df, 
                             palette=custom_palette, 
                             hue_order=['response', 'no_response'], 
                             dodge=True,
                             gap = 0.25)

            # Update legend labels
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles, [display_labels[label] for label in labels], loc='upper right')
            

            # Add statistical annotation
            if 'no_response' in df['Classification'].values and 'response' in df['Classification'].values:
                non_responder_vals = df[df['Classification'] == 'no_response']['FeatureValues']
                responder_vals = df[df['Classification'] == 'response']['FeatureValues']
                p_value = ttest_ind(non_responder_vals, responder_vals).pvalue
                y_max = df['FeatureValues'].max()
                add_stat_annotation(ax, 0, 0.1, y_max + 0.05 * y_max, p_value)


            ax.set_ylabel('Feature Values', fontsize=10)

            # Remove the x-axis label
            # ax.set_xlabel(r'Mean area of granulocytes inside tumor regions ($\mu m^2$)', fontsize=12)
            ax.set_title(r'Mean area of granulocytes inside tumor regions ($\mu m^2$)', fontsize=12)

            # Save the plot
            savename = featname + '_boxplot.png'
            saveboxplot_path = os.path.join(pathtosavefolder, 'boxplots', 'allfeat', savename)
            os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(saveboxplot_path, dpi=300)
            plt.close()



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
## Plot Violin plots for every features (with plotnine)
#############################################################


if violinplots:

    # Define custom colors for each class
    custom_palette = {
        'response': '#FF5733',  # Vibrant Orange
        'no_response': '#3498DB'  # Vibrant Sky Blue
    }

    # Map display names
    display_labels = {'response': 'Responder', 'no_response': 'Non-responder'}

    if delete_outliers:
        pourcentagerem = 0.1
        for featindex in tqdm(range(0, len(featnames))):
            featvals = featarray[featindex, :]
            indices_to_keep = np.where(
                ((featvals > np.quantile(featvals, pourcentagerem / 2)) &
                 (featvals < np.quantile(featvals, 1 - pourcentagerem / 2)))
            )[0]

            featvals_wooutliers = featvals[indices_to_keep]
            clarray_names_wooutliers = [clarray_names[i] for i in indices_to_keep]

            featname = featnames[featindex]
            df = pd.DataFrame({
                'FeatureValues': featvals_wooutliers,
                'Classification': clarray_names_wooutliers,
                'FeatureName': [featname] * len(featvals_wooutliers)
            })

            plt.figure(figsize=(10, 6))
            ax = sns.violinplot(x='FeatureName', y='FeatureValues', hue='Classification', 
                                data=df, palette=custom_palette, hue_order=['response', 'no_response'], dodge=True)

            # Update legend labels
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [display_labels[label] for label in labels], loc='upper right')

            # Add statistical annotation
            if 'no_response' in df['Classification'].values and 'response' in df['Classification'].values:
                non_responder_vals = df[df['Classification'] == 'no_response']['FeatureValues']
                responder_vals = df[df['Classification'] == 'response']['FeatureValues']
                p_value = ttest_ind(non_responder_vals, responder_vals).pvalue
                y_max = df['FeatureValues'].max()
                # Adjust x positions based on violin plot offsets
                add_stat_annotation(ax, 0, 0.1, y_max + 0.05 * y_max, p_value)

            ax.set_title(f'{featname} (removed {pourcentagerem * 100}% outliers)', fontsize=12)
            ax.set_ylabel('Feature Values', fontsize=10)

            # Save the plot
            savename = featname + '_violinplot_filterquartile.png'
            saveboxplot_path = os.path.join(pathtosavefolder, 'violinplots', 'allfeat', savename)
            os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(saveboxplot_path, dpi=300)
            plt.close()

    else:
        for featindex in tqdm(range(0, len(featnames))):
            featvals = featarray[featindex, :]
            featname = featnames[featindex]

            df = pd.DataFrame({
                'FeatureValues': featvals, 
                'Classification': clarray_names,
                'FeatureName': [featname] * len(featvals)
            })

            plt.figure(figsize=(10, 6))
            ax = sns.violinplot(x='FeatureName', 
                                y='FeatureValues', 
                                hue='Classification', 
                                data=df, 
                                palette=custom_palette, 
                                hue_order=['response', 'no_response'], 
                                dodge=True)

            # Update legend labels
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [display_labels[label] for label in labels], loc='upper right')

            # Add statistical annotation
            if 'no_response' in df['Classification'].values and 'response' in df['Classification'].values:
                non_responder_vals = df[df['Classification'] == 'no_response']['FeatureValues']
                responder_vals = df[df['Classification'] == 'response']['FeatureValues']
                p_value = ttest_ind(non_responder_vals, responder_vals).pvalue
                y_max = df['FeatureValues'].max()
                # Adjust x positions based on violin plot offsets
                add_stat_annotation(ax, 0, 0.1, y_max + 0.05 * y_max, p_value)

            ax.set_title(featname, fontsize=12)
            ax.set_ylabel('Feature Values', fontsize=10)

            # Save the plot
            savename = featname + '_violinplot.png'
            saveboxplot_path = os.path.join(pathtosavefolder, 'violinplots', 'allfeat', savename)
            os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(saveboxplot_path, dpi=300)
            plt.close()



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



#