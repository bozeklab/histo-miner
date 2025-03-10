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
pathtoworkfolder = config.paths.folders.featarray_folder
pathtosavefolder = config.paths.folders.visualizations
path_exjson = config.paths.files.example_json
redundant_feat_names = list(config.parameters.lists.redundant_feat_names)


boxplots = config.parameters.bool.plot.boxplots
distributions = config.parameters.bool.plot.distributions
violinplots = config.parameters.bool.plot.violinplots
pca = config.parameters.bool.plot.pca
tsne = config.parameters.bool.plot.tsne
delete_outliers = config.parameters.bool.plot.delete_outliers


visu_featnames = [
    # 'Pourcentage_Lymphocytes_allcellsinTumorVicinity'
    # 'LogRatio_Granulocytes_Lymphocytes_inTumorVicinity'
    # 'Pourcentage_Lymphocytes_allcellsinTumor'
    'Distances_of_cells_in_Tumor_Regions_DistClosest_Granulocytes_PlasmaCells_inTumor_dist_mean'
    ]


# xlabelname = r'Percentage of lymphocytes among cells in tumor vicinity'
# xlabelname = r'Ratio numbers granulocytes/lymphocytes in tumor vicinity(log)'
# xlabelname = r'Percentage of lymphocytes among cells in tumor regions'
xlabelname = r'Mean closest distance between granulocytes and plasma cells in tumor regions'




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


     
visu_indexes = [featnames.index(feat) for feat in visu_featnames if feat in featnames]


#############################################################
## Functions
#############################################################


# Function to add significance bars with p-value in scientific notation
def add_stat_annotation(ax, x1, x2, y, p_value):
    """Annotate the plot with p-value bars and p-value text in scientific notation."""
    # Format the p-value using scientific notation
    significance = f"p = {p_value:.2e}"  # Display p-value in scientific notation with 2 decimal places

    # Add horizontal bar
    ax.plot([x1, x2], [y, y], color="white", lw=1.5)
    
    # Add the p-value text above the bar
    ax.text(
        (x1 + x2) * 0.5,
        y,
        significance,
        ha='center',
        va='bottom',
        color="black",
        fontsize=13  # Optional: adjust font size
    )



def add_stat_annotation_stars(ax, x1, x2, y, p_value):
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
## Plot boxplots for selected features 
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


        ###!! ONLY FOR EXPLORATION PURPOSES AS THERE IS NO JUSTIFICATION IN DELETING OUTLIERS 

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

            plt.figure(figsize=(8, 10))
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

            hue_order = ['response', 'no_response']

            plt.figure(figsize=(8, 10))
            ax = sns.boxplot(x='FeatureName', 
                             y='FeatureValues', 
                             hue='Classification', 
                             data=df, 
                             palette=custom_palette, 
                             hue_order=hue_order, 
                             fill=False,
                             dodge=True,
                             gap = 0.25,
                             showfliers=False  # Remove the outlier markers (white dot)
                             )

            # # Make the boxplot interiors transparent
            # for patch in ax.patches:
            #     patch.set_facecolor('none')  # Make the interior transparent
            #     patch.set_edgecolor(patch.get_edgecolor())  # Keep the edge color
            #     patch.set_linewidth(2)  # Optional: make edges thicker

            # Overlay individual sample points
            sns.stripplot(
                x='FeatureName',
                y='FeatureValues',
                hue='Classification',
                data=df,
                palette=custom_palette,
                hue_order=hue_order,
                dodge=True,
                size=9,  # Adjust dot size
                ax=ax,
                legend=False  # Exclude stripplot from the legend
            )

            # Update legend labels
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, [display_labels[label] for label in labels], loc='upper right', fontsize=13)
            

            # Add statistical annotation
            if 'no_response' in df['Classification'].values and 'response' in df['Classification'].values:
                non_responder_vals = df[df['Classification'] == 'no_response']['FeatureValues']
                responder_vals = df[df['Classification'] == 'response']['FeatureValues']
                p_value = ttest_ind(non_responder_vals, responder_vals).pvalue
                y_max = df['FeatureValues'].max()
                add_stat_annotation(ax, 0, 0.1, y_max + 0.05 * y_max, p_value)


            ax.set_ylabel('Feature Values', fontsize=18)

            # Remove the x-axis label
            # ax.set_xlabel(r'Mean area of granulocytes inside tumors regions ($\mu m^2$)', fontsize=18)
            ax.set_xlabel(xlabelname, fontsize=18)
            # ax.set_xlabel('Porcentage of lymphocytes among all cells (whole WSI)', fontsize=18)

            # Save the plot
            savename = featname + '_boxplot.svg' 
            saveboxplot_path = os.path.join(pathtosavefolder, 'boxplots', 'allfeat', savename)
            os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(saveboxplot_path, format='svg', dpi=300)
            plt.close()



########################################################################
## Plot Kernel Density distribution for selected features (with plotnine)
########################################################################

if distributions:

    norec_idx = [idx for idx, value in enumerate(clarray) if value == 0]
    rec_idx = [idx for idx, value in enumerate(clarray) if value == 1]
    featarray_norec = featarray[:,norec_idx]
    featarray_rec = featarray[:,rec_idx]

    if delete_outliers:

        ###!! ONLY FOR EXPLORATION PURPOSES AS THERE IS NO JUSTIFICATION IN DELETING OUTLIERS 

        #set the variables
        # Filter extremes quartiles but here for both rec and no_rec vectors
        for featindex in tqdm(visu_indexes):
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
        for featindex in tqdm(visu_indexes):
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

            # Save the plot
            savename = featname + '_distribution.svg' 
            saveboxplot_path = os.path.join(pathtosavefolder, 'density', 'allfeat', savename)
            os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(saveboxplot_path, format='svg', dpi=300)
            plt.close()


#############################################################
## Plot Violin plots for selected features (with plotnine)
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

        ###!! ONLY FOR EXPLORATION PURPOSES AS THERE IS NO JUSTIFICATION IN DELETING OUTLIERS 
        
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

            plt.figure(figsize=(10, 8))
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
        for featindex in tqdm(visu_indexes):
            featvals = featarray[featindex, :]
            featname = featnames[featindex]

            df = pd.DataFrame({
                'FeatureValues': featvals, 
                'Classification': clarray_names,
                'FeatureName': [featname] * len(featvals)
            })

            plt.figure(figsize=(10, 8))
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
            savename = featname + '_violinplot.svg'
            saveboxplot_path = os.path.join(pathtosavefolder, 'violinplots', 'allfeat', savename)
            os.makedirs(os.path.dirname(saveboxplot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(saveboxplot_path, format='svg', dpi=300)
            plt.close()

