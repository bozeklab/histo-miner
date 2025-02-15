#Lucas Sancéré -

import os
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
eval_folder_name = config.names.eval_folder
pathtosavefolder = config.paths.folders.visualizations

## Put conditions on which one to plot (no different codes, it is not necessary here)
# We don't necessarely display everything at the same time but it is still a possiblity 
#Boleans

mrmr_viz = True 
mannwhitneyu_viz = False
boruta_viz = False 

if not os.path.exists(pathtosavefolder):
    os.mkdir(pathtosavefolder)


#############################################################
## PLot curves
#############################################################

path2vectors = pathtofolder + '/' + eval_folder_name + '/' 
ext = '.npy'


# For xgboost 

# Load vectors from files
if mrmr_viz:
    xgbmean_aAcc_mrmr = np.load(path2vectors + 'mean_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy')
    xgbmax_aAcc_mrmr = np.load(path2vectors + 'min_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy')
    xgbmin_aAcc_mrmr = np.load(path2vectors + 'max_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy')
    xgbstd_aAcc_mrmr = np.load(path2vectors + 'std_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy')
    # xgbbestsplit_aAcc_mrmr = np.load(path2vectors + 'xgbbestsplit_aAcc_mrmr' + ext)
    # Creating x coordinates for mrmr (also works with mannwhtneyu running)
    # Generating the line fo std with elment wise addition
    std_up = xgbmean_aAcc_mrmr + xgbstd_aAcc_mrmr
    std_down = xgbmean_aAcc_mrmr  - xgbstd_aAcc_mrmr

    x = np.linspace(280, 1, len(xgbmean_aAcc_mrmr))

if mannwhitneyu_viz:
    xgbmean_aAcc_mannwhitneyu = np.load(path2vectors + 'mean_xgboost_ba_mannwhitneyut_5splits_preprint_xgboost_CPI_80_0-1_v1.npy')
    xgbmax_aAcc_mannwhitneyu = np.load(path2vectors + 'max_xgboost_ba_mannwhitneyut_5splits_preprint_xgboost_CPI_80_0-1_v1.npy')
    xgbmin_aAcc_mannwhitneyu = np.load(path2vectors + 'min_xgboost_ba_mannwhitneyut_5splits_preprint_xgboost_CPI_80_0-1_v1.npy')
    xgbstd_aAcc_mannwhitneyu = np.load(path2vectors + 'std_xgboost_ba_mannwhitneyut_5splits_preprint_xgboost_CPI_80_0-1_v1.npy')
    # xgbbestsplit_aAcc_mannwhitneyu = np.load(path2vectors + 'xgbbestsplit_aAcc_mannwhitneyu' + ext)
    # Generating the line fo std with elment wise addition
    std_up = xgbmean_aAcc_mannwhitneyu + xgbstd_aAcc_mannwhitneyu
    std_down = xgbmean_aAcc_mannwhitneyu  - xgbstd_aAcc_mannwhitneyu
    
    x = np.linspace(280, 1, len(xgbmean_aAcc_mannwhitneyu))

if boruta_viz:
    xgbmean_aAcc_boruta = np.load(path2vectors + 'min_xgboost_ba_boruta_5splits_preprint1_noscaler.npy')
    # xgbbestsplit_aAcc_boruta = np.load(path2vectors + 'xgbbestsplit_aAcc_boruta' + ext)
    if os.path.exists(path2vectors + 'xgboostnbr_feat_kept_boruta_5preprint1_noscaler.npy'):
         nbr_featkept_xgb_boruta = np.load(path2vectors + 'xgboostnbr_feat_kept_boruta_5preprint1_noscaler.npy')
         xgb_xboruta = nbr_featkept_xgb_boruta



# PLot figure for xgboost
# Increase the figure width to make room for the legend
plt.figure(figsize=(16, 10))
# plt.figure(figsize=(6, 4))

# First figure with xgboost
if mrmr_viz:
    plt.plot(x, xgbmean_aAcc_mrmr, label='cvmean_mrmr', color='darkblue')
    plt.plot(x, std_up, color='lightskyblue')
    plt.plot(x, std_down, label='standard deviation', color='lightskyblue')
    # plt.plot(x, xgbmax_aAcc_mrmr, color='lightskyblue')
    # plt.plot(x, xgbmin_aAcc_mrmr, label='best and worst split', color='lightskyblue')


if mannwhitneyu_viz:
    plt.plot(x, xgbmean_aAcc_mannwhitneyu, label='cvmean_mannwhitney', color='darkgreen')
    plt.plot(x, std_up, color='lightgreen')
    plt.plot(x, std_down, label='standard deviation', color='lightgreen')
    # plt.plot(x, xgbmax_aAcc_mannwhitneyu, color='lightgreen')
    # plt.plot(x, xgbmin_aAcc_mannwhitneyu, label='best and worst split', color='lightgreen')

if boruta_viz:
        if os.path.exists(path2vectors + 'xgboostnbr_feat_kept_boruta_5preprint1_noscaler.npy'):
            plt.plot(xgb_xboruta, xgbmean_aAcc_boruta, label='cvmean_boruta', color='darkorange')
            # plt.plot(xboruta, xgbbestsplit_aAcc_boruta, label='bestsplit_boruta', color='wheat')
        else:
            print("Path with all boruta information not found ")

plt.xlim(max(x), min(x))

# Plot random classification binary accuracy
plt.axhline(y=0.5, color='black', linestyle='--', label='Random binary accuracy')

# Add labels and a legend
plt.xlabel('Number of feature kept')
plt.ylabel('Balanced Accuracy of Classification')
plt.title('Effect of number of kept features on xgboost balanced accuracy')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.85), ncol=1)

# Set the step on the y-axis
plt.yticks(np.arange(0.1, 1.01, 0.1))  # Adjust the values as needed
plt.ylim(bottom=0.00, top=1.01)  # Weirdly we also need this line to have the 2 spots matching

# Save the plot on the root classification_evaluation directory
plt.savefig(pathtosavefolder + 'std_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v1.png')
plt.clf()



# For lgbm 

# lgbmmean_aAcc_mrmr = np.load(path2vectors + 'mean_ba_mrmr_lgbm_10splits_allCohorts' + ext)
# lgbmmean_aAcc_mannwhitneyu = np.load(path2vectors + 'mean_ba_mannwhitneyu_lgbm_10splits_allCohorts' + ext)
# lgbmmean_aAcc_boruta = np.load(path2vectors + 'mean_ba_boruta_lgbm_10splits_allCohorts' + ext)
# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_lgbm_10splits_allCohorts' + ext):
#      nbr_featkept_lgbm_boruta = np.load(path2vectors + 'nbr_feat_kept_boruta_lgbm_10splits_allCohorts' + ext)
# lgbmbestsplit_aAcc_mrmr = np.load(path2vectors + 'lgbmbestsplit_aAcc_mrmr' + ext)
# lgbmbestsplit_aAcc_mannwhitneyu = np.load(path2vectors + 'lgbmbestsplit_aAcc_mannwhitneyu' + ext)
# lgbmbestsplit_aAcc_boruta = np.load(path2vectors + 'lgbmbestsplit_aAcc_boruta' + ext)
# # Creating x coordinates for mrmr and mannwhtneyu
# x = np.linspace(56, 1, len(lgbmmean_aAcc_mrmr))


# if os.path.exists(path2vectors + 'xgboostnbr_feat_kept_boruta_5npretest1' + ext):
#     # need to duplicate the value for boruta 
#     xgbmean_aAcc_boruta = [xgbmean_aAcc_boruta[0], xgbmean_aAcc_boruta[0]]
#     











# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_lgbm_10splits_allCohorts' + ext):
# # need to duplicate the value for boruta 
#     lgbmmean_aAcc_boruta = [lgbmmean_aAcc_boruta[0], lgbmmean_aAcc_boruta[0]]   
#     lgbm_xboruta = nbr_featkept_lgbm_boruta


# ## PLot figure for lgbm
# plt.figure(figsize=(6, 4))

# # First figure with xgboost
# plt.plot(x, lgbmmean_aAcc_mrmr, label='cvmean_mrmr', color='darkblue')
# # plt.plot(x, lgbmbestsplit_aAcc_mrmr, label='bestsplit_mrmr', color='lightskyblue')
# plt.plot(x, lgbmmean_aAcc_mannwhitneyu, label='cvmean_mannwhitney', color='darkgreen')
# # plt.plot(x, lgbmbestsplit_aAcc_mannwhitneyu, label='bestsplit_mannwhitney', color='lightgreen')
# plt.plot(lgbm_xboruta, lgbmmean_aAcc_boruta, label='cvmean_boruta', color='darkorange')

# # plt.plot(xboruta, lgbmbestsplit_aAcc_boruta, label='bestsplit_boruta', color='wheat')
# plt.xlim(max(x), min(x))

# # Plot random classification binary accuracy
# plt.axhline(y=0.5, color='black', linestyle='--', label='Random binary accuracy')

# # Add labels and a legend
# plt.xlabel('Number of feature kept')
# plt.ylabel('Balanced Accuracy of Classification')
# plt.title('Effect of number of kept features on lgbm balanced accuracy')
# #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# # Set the step on the y-axis
# plt.yticks(np.arange(0.45, 0.95, 0.05))  # Adjust the values as needed
# plt.ylim(bottom=0.45, top=0.92)  # Weirdly we also need this line to have the 2 spots matching

# # Save the plot on the root classification_evaluation directory
# plt.savefig(pathtosavefolder + 'varfeat-classic-ranking-lgbm-logs-vici.png')
# plt.clf()




#############################################################
## PLot figures where features are removed in inverse order
############################################################

#### By default, the snippet is not running
plot_inverse = False
# this boolean is hardcoded because the following snippet is here 
# to reproduce secondary figures of the paper, but shouldn't run in normal
# use of the code. 

if plot_inverse:

    path2vectors = pathtofolder + '/TestofKs/' + 'AllCohorts/'
    ext = '.npy'

    # Load vectors from files
    xgbmean_aAcc_mrmr = np.load(path2vectors + 'xgbmean_aAcc_mrmr_inversed' + ext)
    xgbmean_aAcc_mannwhitneyu = np.load(path2vectors + 'xgbmean_aAcc_mannwhitneyu_inversed' + ext)
    xgbbestsplit_aAcc_mrmr = np.load(path2vectors + 'xgbbestsplit_aAcc_mrmr_inversed' + ext)
    xgbbestsplit_aAcc_mannwhitneyu = np.load(path2vectors + 'xgbbestsplit_aAcc_mannwhitneyu_inversed' + ext)
    lgbmmean_aAcc_mrmr = np.load(path2vectors + 'lgbmmean_aAcc_mrmr_inversed' + ext)
    lgbmmean_aAcc_mannwhitneyu = np.load(path2vectors + 'lgbmmean_aAcc_mannwhitneyu_inversed' + ext)
    lgbmbestsplit_aAcc_mrmr = np.load(path2vectors + 'lgbmbestsplit_aAcc_mrmr_inversed' + ext)
    lgbmbestsplit_aAcc_mannwhitneyu = np.load(path2vectors + 'lgbmbestsplit_aAcc_mannwhitneyu_inversed' + ext)


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
    plt.xlim(max(x), min(x))

    # Plot random classification binary accuracy
    plt.axhline(y=0.5, color='black', linestyle='--', label='Random binary accuracy')

    # Add labels and a legend
    plt.xlabel('Number of feature kept')
    plt.ylabel('Balanced Accuracy of Classification')
    plt.title('Effect of number of kept features in inverse order on xgboost balanced accuracy')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set the step on the y-axis
    plt.yticks(np.arange(0.45, 0.95, 0.05))  # Adjust the values as needed
    plt.ylim(bottom=0.45, top=0.92)  # Weirdly we also need this line to have the 2 spots matching

    # Save the plot on the root classification_evaluation directory
    plt.savefig(pathtofolder + 'xgboost_result_inverse_order.png')
    plt.clf()


    ## PLot figure for lgbm
    plt.figure(figsize=(6, 4))

    # First figure with xgboost
    plt.plot(x, lgbmmean_aAcc_mrmr, label='cvmean_mrmr', color='darkblue')
    plt.plot(x, lgbmbestsplit_aAcc_mrmr, label='bestsplit_mrmr', color='lightskyblue')
    plt.plot(x, lgbmmean_aAcc_mannwhitneyu, label='cvmean_mannwhitney', color='darkgreen')
    plt.plot(x, lgbmbestsplit_aAcc_mannwhitneyu, label='bestsplit_mannwhitney', color='lightgreen')
    plt.xlim(max(x), min(x))

    # Plot random classification binary accuracy
    plt.axhline(y=0.5, color='black', linestyle='--', label='Random binary accuracy')

    # Add labels and a legend
    plt.xlabel('Number of feature kept')
    plt.ylabel('Balanced Accuracy of Classification')
    plt.title('Effect of number of kept features in inverse order on lgbm balanced accuracy')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set the step on the y-axis
    plt.yticks(np.arange(0.45, 0.95, 0.05))  # Adjust the values as needed
    plt.ylim(bottom=0.45, top=0.92)  # Weirdly we also need this line to have the 2 spots matching

    # Save the plot on the root classification_evaluation directory
    plt.savefig(pathtofolder + 'lgbm_result_inverse_order.png')
    plt.clf()





#############################################################
## When several Borutas
############################################################



# xgbmean_aAcc_boruta_2 = np.load(path2vectors + 'mean_ba_boruta_xgboost_2_10splits' + ext)
# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_2_xgboost_10splits' + ext):
#      nbr_featkept_xgb_boruta_2 = np.load(path2vectors + 'nbr_feat_kept_boruta_2_xgboost_10splits' + ext)

# xgbmean_aAcc_boruta_3 = np.load(path2vectors + 'mean_ba_boruta_xgboost_3_10splits' + ext)
# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_3_xgboost_10splits' + ext):
#      nbr_featkept_xgb_boruta_3 = np.load(path2vectors + 'nbr_feat_kept_boruta_3_xgboost_10splits' + ext)
# lgbmmean_aAcc_boruta_2 = np.load(path2vectors + 'mean_ba_boruta_lgbm_2_10splits' + ext)
# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_2_lgbm_10splits' + ext):
#      nbr_featkept_lgbm_boruta_2 = np.load(path2vectors + 'nbr_feat_kept_boruta_2_lgbm_10splits' + ext)

# lgbmmean_aAcc_boruta_3 = np.load(path2vectors + 'mean_ba_boruta_lgbm_3_10splits' + ext)
# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_3_lgbm_10splits' + ext):
#      nbr_featkept_lgbm_boruta_3 = np.load(path2vectors + 'nbr_feat_kept_boruta_3_lgbm_10splits' + ext)



# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_2_xgboost_10splits' + ext):
#     # need to duplicate the value for boruta 
#     xgbmean_aAcc_boruta_2 = [xgbmean_aAcc_boruta_2[0], xgbmean_aAcc_boruta_2[0]]
#     xgb_xboruta_2 = nbr_featkept_xgb_boruta_2

# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_3_xgboost_10splits' + ext):
#     # need to duplicate the value for boruta 
#     xgbmean_aAcc_boruta_3 = [xgbmean_aAcc_boruta_3[0], xgbmean_aAcc_boruta_3[0]]
#     xgb_xboruta_3 = nbr_featkept_xgb_boruta_3

# plt.plot(xgb_xboruta_2, xgbmean_aAcc_boruta_2, label='cvmean_boruta', color='red')
# plt.plot(xgb_xboruta_3, xgbmean_aAcc_boruta_3, label='cvmean_boruta', color='orange')
# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_2_lgbm_10splits' + ext):
#     # need to duplicate the value for boruta 
#     lgbmmean_aAcc_boruta_2 = [lgbmmean_aAcc_boruta_2[0], lgbmmean_aAcc_boruta_2[0]]
#     lgbm_xboruta_2 = nbr_featkept_lgbm_boruta_2

# if os.path.exists(path2vectors + 'nbr_feat_kept_boruta_3_lgbm_10splits' + ext):
#     # need to duplicate the value for boruta 
#     lgbmmean_aAcc_boruta_3 = [lgbmmean_aAcc_boruta_3[0], lgbmmean_aAcc_boruta_3[0]]
#     lgbm_xboruta_3 = nbr_featkept_lgbm_boruta_3

# plt.plot(lgbm_xboruta_2, xgbmean_aAcc_boruta_2, label='cvmean_boruta', color='red')
# plt.plot(lgbm_xboruta_3, xgbmean_aAcc_boruta_3, label='cvmean_boruta', color='orange')