

import csv  
import os
import numpy as np
from tqdm import tqdm
import json
from tqdm import tqdm
import shutil

#####################################################################
## 1
#####################################################################

# pathtosave_file = '/data/shared/scc/histo-miner/data_analysis/crea_patient_ID_lst/orderedlist.csv'
# pathtojsons_folder = '/data/shared/scc/histo-miner/data_analysis/crea_patient_ID_lst/recurrence/Regensburg'

# # filenamelist = list()
# # for root, dirs, files in os.walk(pathtojsons_folder):
# #     if files:  # Keep only the not empty lists of files
# #         # Because files is a list of file name here, and not a srting. You create a string with this:
# #         for file in files: 
# #         	filename = os.path.splitext(file)
# #         	filenamelist.append(filename)

# # filenamelist.sort()
# # with open(pathtosave_file, 'w', encoding='UTF8') as f:
# #      writer = csv.writer(f)
# #      for filename in filenamelist:
# # 	     writer.writerow(filename)



# study = np.load("/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/feature_selection/patientids.npy", allow_pickle=True)
# pass

#####################################################################
## 2 add distances calculations to jsons
#####################################################################
	        	
# pathtonew = '/data/lsancere/Data_General/Predictions/scc_lucas/tissue_analyses_2_withcohorts/cohorts/Regensburg/recurrence/'
# pathtoold = '/data/shared/scc/histo-miner/data_analysis/tissue_analyses/tissue_analyses_indiv_cohorts_withlogs_withvicinity/tissue_analyses_sorted_Regensburg_nospace_in_names/recurrence/'

# # Creaet 2 list of files 

# ########  Create lists with the paths of the files to process
# new_jsonfiles = list()
# for root, dirs, files in os.walk(pathtonew):
#     if files:  # Keep only the not empty lists of files
#         # Because files is a list of file name here, and not a srting. You create a string with this:
#         for file in files:
#             namewoext, extension = os.path.splitext(file)
#             filepath = root + '/' + file
#             # Knowing that root is the path to the directory of the selected file,
#             # root + file is the complete path
#             if extension == '.json':
#                 new_jsonfiles.append(filepath)


# #### Update of the jsons

# for newjson in tqdm(new_jsonfiles):

#     corresponding_oldjson = pathtoold + '/' + os.path.split(newjson)[1]

#     with open(corresponding_oldjson, 'r') as filename:
#     	oldjsonfile = filename.read()
#     	oldjson_dict = json.loads(oldjsonfile)
#     	# now extacrt dist nested dict 
#     	dist_nested_dict = oldjson_dict['CalculationsDistinsideTumor']['Distances_of_cells_in_Tumor_Regions']

#     with open(newjson, 'r') as filename:
#     	newjsonfile = filename.read()
#     	newjson_dict = json.loads(newjsonfile)
#     	# sanity check 
#     	filenotempty = True
#     	newjson_dict['CalculationsDistinsideTumor']['Distances_of_cells_in_Tumor_Regions'] = dist_nested_dict
#     	# new_json_content = json.load(newjson_dict)
        
#     if filenotempty:
#         with open(newjson, 'w') as filename:
#         	# it will overwritte the json as watned
#             json.dump(newjson_dict, filename)       

# print('All json files updated ')


#####################################################################
## 3 Edit csvs
#####################################################################


# # Specify the path to the CSV file (this will be overwritten)
# file_path = "/data/shared/scc/histo-miner/data_analysis/crea_patient_ID_lst/AllCohorts_PatientIDs_test.csv"  # Replace with the actual path to your CSV file

# # Read the CSV file and store the modified content in memory
# with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
#     reader = csv.reader(infile)
#     # Create a list to store the modified rows
#     modified_rows = []
    
#     for row in reader:
#         # Replace spaces with underscores for each cell
#         new_row = [cell.replace(' ', '_') if isinstance(cell, str) else cell for cell in row]
#         modified_rows.append(new_row)

# # Open the same CSV file in write mode and overwrite with modified content
# with open(file_path, mode='w', newline='', encoding='utf-8') as outfile:
#     writer = csv.writer(outfile)
#     # Write the modified rows back to the CSV file
#     writer.writerows(modified_rows)

# print(f"File '{file_path}' has been overwritten with modified content.")



#####################################################################
## 4 Open npys to check std, min and max of cross val splits
# #####################################################################


# # Path to npy file
# folderpath = '/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/classification_evaluation_allpreprint/results/kept/26samples/'
# folderpath = '/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/classification_evaluation_allpreprint/results/kept/alldata/'
# folderpath = '/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/classification_evaluation_allpreprint/results/kept/34samples/'
folderpath = '/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/classification_evaluation_allpreprint_rndfilte/results/'


# file_path_std = folderpath + 'std_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_26samples_v4.npy'
# file_path_min = folderpath + 'min_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_26samples_v4.npy'
# file_path_max = folderpath + 'max_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_26samples_v4.npy'
# file_path_mean = folderpath + 'mean_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_26samples_v4.npy'

# file_topfeat = folderpath + 'topselfeatid_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_26samples_v4.npy'
# file_path_std = folderpath + 'std_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4.npy'
# file_path_min = folderpath + 'min_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4.npy'
# file_path_max = folderpath + 'max_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4.npy'
# file_path_mean = folderpath + 'mean_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4.npy'
# file_topfeat = folderpath + 'topselfeatid_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4.npy'

# mRMR and XGboost experimental 34 left
# file_path_std = folderpath + 'std_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_34samples_v4.npy'
# file_path_min = folderpath + 'min_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_34samples_v4.npy'
# file_path_max = folderpath + 'max_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_34samples_v4.npy'
# file_path_mean = folderpath + 'mean_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_34samples_v4.npy'
# file_topfeat = folderpath + 'topselfeatid_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_34samples_v4.npy'


# mRMR and XGBoost 
# file_path_std = folderpath + 'std_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy'
# file_path_min = folderpath + 'min_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy'
# file_path_max = folderpath + 'max_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy'
# file_path_mean = folderpath + 'mean_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy'
# file_topfeat = folderpath + 'topselfeatid_xgboost_ba_mrmr_5splits_preprint_CPI_80_0-1_v4check.npy'

# Mann Whintney and XGBoost 
file_path_std = folderpath + 'std_xgboost_ba_mannwhitneyut_5splits_preprint_CPI_80_0-1_v4check.npy'
file_path_min = folderpath + 'min_xgboost_ba_mannwhitneyut_5splits_preprint_CPI_80_0-1_v4check.npy'
file_path_max = folderpath + 'max_xgboost_ba_mannwhitneyut_5splits_preprint_CPI_80_0-1_v4check.npy'
file_path_mean = folderpath + 'mean_xgboost_ba_mannwhitneyut_5splits_preprint_CPI_80_0-1_v4check.npy'
file_topfeat = folderpath + 'topselfeatid_xgboost_ba_mannwhitneyut_5splits_preprint_CPI_80_0-1_v4check.npy'

# Boruta and XGBoost 
# file_path_std = folderpath + 'std_xgboost_ba_boruta_5splits_preprint_CPI_80_0-1_v4check.npy'
# file_path_min = folderpath + 'min_xgboost_ba_boruta_5splits_preprint_CPI_80_0-1_v4check.npy'
# file_path_max = folderpath + 'max_xgboost_ba_boruta_5splits_preprint_CPI_80_0-1_v4check.npy'
# file_path_mean = folderpath + 'mean_xgboost_ba_boruta_5splits_preprint_CPI_80_0-1_v4check.npy'

stdval = np.load(file_path_std)
minval = np.load(file_path_min)
maxval = np.load(file_path_max)
meanval = np.load(file_path_mean)
# featval = np.load(file_topfeat)


# Check variable in debugger, for instance by breakpoint in pudb 
devink_pudb = True


#####################################################################
## 5 Copy paste files from the list:
#####################################################################


# Path
# source_parentfolder= '/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/' 
# source_folderpath1 = source_parentfolder + '/cSCC_CPI_folders/cSCC_CPI/'
# source_folderpath2 = source_parentfolder + '/cSCC_CPI_extra_folders/cSCC_CPI_extra1/'
# source_folderpath3 = source_parentfolder + '/cSCC_CPI_extra_folders/cSCC_CPI_extra2/'
# source_folderpath4 = source_parentfolder + '/cSCC_CPI_extra_folders/cSCC_CPI_extra3/'
# target_folderpath = '/data/lsancere/Data_General/TrainingSets/histo-miner-all-datasets/cSCC_CPI/'

# list_names_to_copy = [
# 	'S03605_01',
# 	'S03611_03',
# 	'S03612_03',
# 	'S03613_01',
# 	'S03614_01',
# 	'S03615_01',
# 	'S03616_01',
# 	'S03622_01',
# 	'S03623_01',
# 	'S03625_03',
# 	'S03627_02',
# 	'S03628_03',
# 	'S03631_01',
# 	'S03633_02',
# 	'S03637_01',
# 	'S03638_02',
# 	'S03639_01',
# 	'S03640_01',
# 	'S03642_03',
# 	'S03643_01',
# 	'S03644_02',
# 	'S03645_01',
# 	'S03646_02',
# 	'S03647_02',
# 	'S03651_03',
# 	'S03652_01',
# 	'S03654_01',
# 	'S03655_01',
# 	'S03656_02',
# 	'S03788_01',
# 	'S03791_01',
# 	'S03792_01',
# 	'S03793_01'
# ]

# ext = '.ndpi'

# files_in_folderpath1 = os.listdir(source_folderpath1)
# files_in_folderpath2 = os.listdir(source_folderpath2)
# files_in_folderpath3 = os.listdir(source_folderpath3)
# files_in_folderpath4 = os.listdir(source_folderpath4)


# # Function to copy file from source to target
# def copy_file(source, target):
#     try:
#         shutil.copy(source, target)
#         print(f"Copied {source} to {target}")
#     except Exception as e:
#         print(f"Error copying {source} to {target}: {e}")

# # Loop through the filenames to copy
# for filename in tqdm(list_names_to_copy):
#     source_file = filename + ext
    
#     if source_file in files_in_folderpath1:
#         copy_file(os.path.join(source_folderpath1, source_file), target_folderpath)
#     elif source_file in files_in_folderpath2:
#         copy_file(os.path.join(source_folderpath2, source_file), target_folderpath)
#     elif source_file in files_in_folderpath3:
#         copy_file(os.path.join(source_folderpath3, source_file), target_folderpath)
#     elif source_file in files_in_folderpath4:
#         copy_file(os.path.join(source_folderpath4, source_file), target_folderpath)
#     else:
#         print(f"{source_file} not found in any source folder")











