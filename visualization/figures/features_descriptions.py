#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotnine
import yaml
from attrdictionary import AttrDict as attributedict
from plotnine import ggplot, aes, geom_boxplot, xlab, ylab, labs, theme, \
                    element_text, geom_density, scale_color_manual, scale_fill_manual

from src.histo_miner.utils.misc import convert_flatten, convert_flatten_redundant
 

# - Plot the correlation matrix in a nice way (seaborn?). In a first step it could stay as just the 56 features. Later-on maybe only display few names or few features, the most interesting ones
# --> maybe load it here and then make it nicer with seaborn library

# - Box-plot per features inside one class 
# - Distribution of features inside one class
# --> Need to load the feature matrix already generated and to sort all the feature of one class in one vector
# --> Look for box-plots libraries and just plot the distribution as well


## Later on
# - t-SNE plot
# - PCA plot  



#############################################################
## Load configs parameter
#############################################################


# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtomainfolder = config.paths.folders.main
pathtoworkfolder = config.paths.folders.feature_selection_main
pathtosavefolder = config.paths.folders.visualizations
example_json = config.names.example_json

boxplots = config.parameters.bool.plot.boxplots
distributions = config.parameters.bool.plot.distributions
delete_outliers = config.parameters.bool.plot.delete_outliers



#############################################################
## Load correlation matrix
#############################################################

print('Some informatino that the correlation matrix need to exist already')

matrix_path = '/correlations/correlation_matrix.npy'



#############################################################
## Load feature matrix and classification array, feat names
#############################################################


featarray_name = 'featarray'
classarray_name = 'clarray'
ext = '.npy'

featarray = np.load(pathtoworkfolder + featarray_name + ext)
clarray = np.load(pathtoworkfolder + classarray_name + ext)
clarray_list = list(clarray)

clarray_names = ['no_recurrence' if value == 0 else 'recurrence' for value in clarray_list]

# We can create the list of feature name just by reading on jsonfile
pathto_sortedfolder = pathtomainfolder + '/' + 'tissue_analyses_sorted/'
path_exjson = pathto_sortedfolder + example_json
with open(path_exjson, 'r') as filename:
    analysisdata = filename.read()
    analysisdata = json.loads(analysisdata)
    # flatten the dict (with redundant keys in nested dict, see function)
    analysisdataflat = convert_flatten_redundant(analysisdata)

featnames = list(analysisdataflat.keys())



#############################################################
## Plot boxplots for every features
#############################################################

if boxplots:
    if delete_outliers:
        # There are plenty of ways to detect outliers, here we use the easiest one
        # just removing a given pourcentage of bottom and top values
        pass
    else:
        for featindex in range(0, len(featnames)):
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
                        + theme(plot_title=element_text(size=7))
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
## Plot Kernel Density distribution for every features
#############################################################

if distributions:
    norec_idx = [idx for idx, value in enumerate(clarray) if value == 0]
    rec_idx = [idx for idx, value in enumerate(clarray) if value == 1]
    featarray_norec = featarray[:,norec_idx]
    featarray_rec = featarray[:,rec_idx]

    if delete_outliers:
        # There are plenty of ways to detect outliers, here we use the easiest one
        # just removing a given pourcentage of bottom and top values
        pass
    else:
        for featindex in range(0, len(featnames)):
            featvals_norec = list(featarray_norec[featindex,:])
            featvals_rec = list(featarray_rec[featindex,:])
            featname = featnames[featindex]
            # Create a pandas DataFrame
            df = pd.DataFrame({'FeatureValues': featvals_norec + featvals_rec,
                'Distribution': ['featvals_norec'] * len(featvals_norec) + ['featvals_rec'] * len(featvals_rec)})

            # Define colors for each distribution
            colors = {'featvals_norec': 'blue', 'featvals_rec': 'red'}

            # Plot kernel density distributions for both vectors
            density_plot = (ggplot(df, aes(x='FeatureValues', 
                                           color='Distribution', 
                                           fill='Distribution'))
                            + geom_density(alpha=0.7)
                            + scale_color_manual(values=[colors['featvals_rec'], 
                                                         colors['featvals_norec']])
                            + scale_fill_manual(values=[colors['featvals_rec'], 
                                                        colors['featvals_norec']])
                            + xlab("Whole Slide Image Classification")
                            + ylab("Feature Values")
                            + labs(title= featname)
                            + theme(plot_title=plotnine.element_text(size=7))
                        )
            #Create Name for saving
            savename = featname + '_distribution.png'

            #Saving
            if not os.path.exists(pathtosavefolder + '/density/'):
                os.makedirs(pathtosavefolder + '/density/')
            savedensplot_path = pathtosavefolder + '/density/' + savename
            density_plot.save(savedensplot_path, dpi=300)

# We can try to create a l


#############################################################
## PLot curves
#############################################################
