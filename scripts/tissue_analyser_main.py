
import json
import os
import sys

import numpy as np
import scipy.stats
import yaml
from attrdict import AttrDict as attributedict


from src.histo_miner import hovernet_utils, segmenter_utils


"""Collection of scripts for different Hovernet tasks"""


##################################### Load configs parameter #############################


# Import parameters values from config file by generating a dict. The lists will be imported as tuples.
with open("./../configs/histo_miner/tissue_analyser.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
parentdir = config.paths.folders.main
hovernet_mode = str(config.parameters.str.hovernet_mode)
values2change = list(config.parameters.lists.values2change)
newvalues = list(config.parameters.lists.newvalues)
selectedclasses = list(config.parameters.lists.selectedclasses)
class_name_as_key = list(config.parameters.lists.classnameaskey)

str2replace_tilemode = config.names.managment.str2replace_tilemode
newstr_tilemode = config.names.managment.newstr_tilemode
str2replace2_tilemode = config.names.managment.str2replace2_tilemode
newstr2_tilemode = config.names.managment.newstr2_tilemode
str2replace_wsimode = config.names.managment.str2replace_wsimode
newstr_wsimode = config.names.managment.newstr_wsimode
str2replace2_wsimode = config.names.managment.str2replace2_wsimode
newstr2_wsimode = config.names.managment.newstr2_wsimode

##################################### Classes for functions ###############################


# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NpEncoder, self).default(obj)


#################################### Select the task to run #######################################

"""
Code snippet 1 : Generate ClassMaps from InstanceMaps and ClassVectors OR
                 Generate ClassVector from InstanceMaps and ClassMaps
Code snippet 2 : Update each json files to be compatible with QuPath
Code snippet 3 : Count number of cells per type and calculate density related metrics from HVN JSON outputs
Code snippet 4 : Run Stats test on the data (for exemple  Mann-Whitney U rank test)
Code snippet 5 : **MOVED** Concatenate the quantification features all together in pandas DataFrame and run MRMR  
"""
#
# codesnippet = 3
#
#
# ############################################## Code Snippet 1 #############################################
#
# """  Generate ClassMaps from InstanceMaps and ClassVectors OR
#      Generate ClassVector from InstanceMaps and ClassMaps    """
#
# generate = 'CentroidPredClassVector'
#
# ## Generate ClassMap Parameters
# if generate is 'ClassMaps':
#     instancefolder = '/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Hovernet/' \
#                      'Carina-Corinna-Johannes-Data/PannukeOriginal/Tests/InferenceOnValid/opt5set3/mat/InstancesMaps/'
#     classvector_folder = '/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Hovernet/' \
#                         'Carina-Corinna-Johannes-Data/PannukeOriginal/Tests/InferenceOnValid/opt5set3/mat/TypesMaps/'
#     savename = '_ClassMap'
#
# ## Generate ClassVector Parameters
# if generate is 'ClassVectors':
#     instancefolder = '/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Hovernet/' \
#                      'Carina-Corinna-Johannes-Data/PannukeOriginal/Training/Train/Labels/'
#     classmapfolder = instancefolder
#     savename = '_ClassVector'
#
# ## Generate CentroidGTClassVector Parameters
# if generate is 'CentroidGTClassVector':
#     instancefolder = '/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Hovernet/' \
#                      'Carina-Corinna-Johannes-Data/PannukeOriginal/Training2/Valid/ValidInstanceMaps/'
#     classmapfolder = '/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Hovernet/' \
#                      'Carina-Corinna-Johannes-Data/PannukeOriginal/Training2/Valid/ValidClassMaps/'
#     savename = '_CentroidClassVectors'
#
# ## Generate CentroidPredClassVector Parameters
# if generate is 'CentroidPredClassVector':
#     centroidfolder = '/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Hovernet/' \
#                      'Carina-Corinna-Johannes-Data/PannukeOriginal/Training2/CentroidsBorderCorrValid/'
#     classmapfolder = '/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Hovernet/' \
#                      'Carina-Corinna-Johannes-Data/PannukeOriginal/Tests/InferenceOnValid/opt5set3/mat/ClassMaps/'
#     savename = '_CentroidClassVectors'  # No parameter to import from ClassicImageProcessing
#
# # Imported from ClassImageProcessing
#
# if codesnippet == 1:
#     if generate is 'ClassMaps':
#         classmap_from_classvector(instancefolder, classvector_folder, savename)
#     if generate is 'ClassVectors':
#         # for root, dirs, files in os.walk(Instancefolder):
#         #     if dirs:
#         #         for directory in dirs:
#         #             PathToFolder = Instancefolder + directory + '/'
#         classvector_from_classmap(instancefolder, classmapfolder, savename)
#     if generate is 'CentroidGTClassVector':
#         centroidclassvector_from_gtclassmap(instancefolder, classmapfolder, savename, noborder=True)
#     if generate is 'CentroidPredClassVector':
#         centroidclassvector_from_predclassmap(centroidfolder, classmapfolder, savename)
#
#     print('Code Snippet 1 executed')



#############################################################
## Processing of inferences results for Tissue Analyser
#############################################################

"""Update each json files to be compatible with Tissue Analyser AND QuPath"""

"""Update each mask output """

## Parameters

# Not useful except for larning purpose
# param = ManageJSONparameters()  # Load ManageJSONparameters from ManageJSON script2021-05-07_16.21.29
# param_tile = Tile()
# param_wsi = WSI()

# Load each parametes as define in ManageJSON script
if hovernet_mode == 'tile':
    string2replace = str(str2replace_tilemode)
    newstring = str(newstr_tilemode)
    string2replace2 = str(str2replace2_tilemode)
    newstring2 = str(newstr2_tilemode)
elif hovernet_mode == 'wsi':
    string2replace = str(str2replace_wsimode)
    newstring = str(newstr_wsimode)
    string2replace2 = str(str2replace2_wsimode)
    newstring2 = str(newstr2_wsimode)
    # string2replace = '{"mag": 40, "nuc": {'
    # newstring = '{'
    # string2replace2 = '}}}'
    # newstring2 = '}}'
else:
    print("hovernet_mode string is not correct, tile mode choosen by default")
    string2replace = str(str2replace_tilemode)
    newstring = str(newstr_tilemode)
    string2replace2 = str(str2replace2_tilemode)
    newstring2 = str(newstr2_tilemode)


print('The folders must contains only hovernet predictions and segmenter predictions files')
for root, dirs, files in os.walk(parentdir):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a srting. You create a string with this:
        for file in files:
            path, extension = os.path.splitext(file)
            # Update each JSON file
            filepath = root + '/' + file
            # Knowing that root is the path to the directory of the selected file,
            # root + file is the complete path
            if extension == '.json':
                print('Detected json file:', file)
                print('Path to file :', filepath)
                hovernet_utils.replacestring_json(filepath, string2replace,
                                                  newstring, string2replace2,
                                                  newstring2)
                pathtofolder, filename = os.path.split(file)
                print('Updated json')
            # if extension != '.json':
            #     print('Detected mask file '
            #           '(has to be in a pillow supported format - like .png)', file)
            #     print('Path to file :', filepath)
            #     segmenter_utils.change_pix_values(filepath, values2change,
            #                                       newvalues)
            #     print('Updated mask file')


print('All json files updated with mode {}'.format(hovernet_mode))
print('All mask files updated')


#############################################################
## Tissue Analysis, Extraction of features
#############################################################









































































######################################### Code Snippet 3 #######################################
#
# """Count number of cells per type and calculate density related metrics from HVN JSON outputs"""
#
# # Parameter
#
# class_names = ['"type": 0', '"type": 1', '"type": 2', '"type": 3', '"type": 4', '"type": 5']
# maskmapdownfactor = 32
#
#
# if codesnippet == 3:
#     for root, dirs, files in os.walk(parentdir):
#         if files:  # Keep only the not empty lists of files
#             # Because files is a list of file name here, and not a srting. You create a string with this:
#             for file in files:
#                 path, extension = os.path.splitext(file)
#                 path_to_parentfolder, nameoffile = os.path.split(path)  # path_to_parentfolder is empty, why?)
#                 if extension == '.json' and 'data' not in nameoffile:
#                     if os.path.exists(parentdir + '/' + nameoffile + '_data2.json'):
#                         print('Detected an already processed file:', nameoffile)
#                         continue
#                     else:
#                         print('Detected json file:', file)
#                         # Knowing that root is the path to the directory of the selected file,
#                         # root + file is the complete path
#                         # Creating the dictionnary to count the cells using countjson function
#                         jsonfilepath = root + '/' + file
#                         print('Process count of cells per cell type in the whole slide image...')
#                         classcountsdict = countjson(jsonfilepath, class_names)
#                         allcells_in_wsi_dict = classcountsdict
#                         print('Allcells_inWSI_dict generated as follow:', allcells_in_wsi_dict)
#                         print('Note: at this stage the correspondance between "type number" and "cell type name" '
#                               'is not done yet')
#                         # Create the path to Mask map binarized and Class JSON and save it into a variable
#                         if os.path.exists(jsonfilepath) and os.path.exists(jsonfilepath.replace(extension, '.png')):
#
#                             # Create path for the maskmap
#                             maskmappath = jsonfilepath.replace(extension, '.png')
#                             print('Detected mask file:', maskmappath)
#
#                             #Change pixel values of the maskmaps
#                             # TO CHECK - very long step GPU or no GPU
#                             # change_pix_values(Maskmappath, [1, 2], [0, 255], UseGPU=False)
#
#                             # Analysis
#                             tumor_tot_area = count_pix_value(maskmappath, 255) * maskmapdownfactor
#                             print('Process cells identification '
#                                   '(number of cells and tot area of cells) inside tumor regions...')
#                             cellsratio_inmask_dict = cellsratio_insidemask_classjson(maskmappath,
#                                                                                      jsonfilepath,
#                                                                                      selectedclasses,
#                                                                                      maskmapdownfactor=maskmapdownfactor,
#                                                                                      classnameaskey=class_name_as_key)
#                             print('Cellsratio_inmask_dict generated as follow:', cellsratio_inmask_dict)
#                             print('Process distance calculcations inside tumor regions...')
#                             cellsdist_inmask_dict = mpcell2celldist_classjson(jsonfilepath,
#                                                                               selectedclasses,
#                                                                               cellfilter='Tumor',
#                                                                               maskmap=maskmappath,
#                                                                               maskmapdownfactor=maskmapdownfactor,
#                                                                               tumormargin=None)
#                             print('Cellsdist_inmask_dict generated as follow:', cellsdist_inmask_dict)
#
#                         else:
#                             cellsratio_inmask_dict = None
#                             cellsdist_inmask_dict = None
#                             print('Cellsratio_inmask_dict not generated')
#                             print('Cellsdist_inmask_dict not generated')
#
#                         jsondata = hvn_outputproperties(allcells_in_wsi_dict,
#                                                         cellsratio_inmask_dict,
#                                                         cellsdist_inmask_dict,
#                                                         masknature='Tumor',
#                                                         areaofmask=tumor_tot_area)
#
#                         # Write information inside a json file and save it
#                         with open(parentdir + '/' + nameoffile + '_data.json', 'w') as outfile:
#                             json.dump(jsondata, outfile, cls=NpEncoder)
#
#                         print('Json file written :', path_to_parentfolder + nameoffile + '_data.json')
#
#     print('Done')
#     print('Code Snippet 3 executed')
#
#
# ############################################# Code Snippet 4 #######################################
#
# """Run Stats test on the data (for exemple  Mann-Whitney U rank test)"""
# """ Moved to mrmr as well"""
#
#
#
# # Parameters
#
# globaldictrec = dict()
# globaldictnorec = dict()
# mannwhitneyu = dict()
# orderedp_mannwhitneyu = dict()
#
# if codesnippet == 4:
#     for root, dirs, files in os.walk(parentdir):
#         if files:  # Keep only the not empty lists of files
#             # Because files is a list of file name here, and not a srting. You create a string with this:
#             for file in files:
#                 path, extension = os.path.splitext(file)
#                 path_to_parentfolder, nameoffile = os.path.split(path)  # path_to_parentfolder is empty, why?)
#
#                 # Separate the 2 groups of data, recurrence and none reccurence
#                 if extension == '.json' and 'data' and 'recurrence' in nameoffile:
#                     if 'norecurrence' in nameoffile:
#                         with open(root + '/' + file, 'r') as filename:
#                             data = filename.read()  # extract information of the JSON as a string
#                             print(file)
#                             data = json.loads(data)  # read JSON formatted string and convert it to a dict
#                             data = convert_flatten_redundant(
#                                 data)  # flatten the dict (with redeundant keys, see function)
#                             for key in data:
#                                 # if the key is already in the dict, we happend to the value
#                                 # (which is a list) the new element
#                                 if key in globaldictnorec:
#                                     globaldictnorec[key].append(data[
#                                                                     key])
#                                 else:
#                                     globaldictnorec[key] = [data[
#                                                                 key]]
#                                     # create a list of one element as 'value' and not just an int!!
#                                     # Like this we can happend to a list after
#                     else:
#                         with open(root + '/' + file, 'r') as filename:
#                             data = filename.read()  # extract information of the JSON as a string
#                             print(file)
#                             data = json.loads(data)  # read JSON formatted string and convert it to a dict
#                             data = convert_flatten_redundant(
#                                 data)  # flatten the dict (with redundant keys in nested dict, see function)
#                             for key in data:
#                                 if key in globaldictrec:
#                                     globaldictrec[key].append(data[key])
#                                 else:
#                                     globaldictrec[key] = [data[key]]
#
#
#     # Remove the keys with 'None' as values
#     del globaldictrec['CalculationsPaperMetrics_SCD']
#     del globaldictrec['CalculationsPaperMetrics_ITLR']
#     del globaldictnorec['CalculationsPaperMetrics_SCD']
#     del globaldictnorec['CalculationsPaperMetrics_ITLR']
#
#     # Calculate Mann-Whitney U rank test on two independent samples
#     for key in globaldictrec:  # we can check the key in any of the 2 recurrence dict() as they share the same keys
#         mannwhitneyu[key] = scipy.stats.mannwhitneyu(globaldictrec[key], globaldictnorec[key])
#         orderedp_mannwhitneyu[key] = scipy.stats.mannwhitneyu(globaldictrec[key],
#                                                               globaldictnorec[key]).pvalue  # Only keep pvalues
#
#     orderedp_mannwhitneyu = sorted(orderedp_mannwhitneyu.items(), key=lambda x: x[1])  # Order the dict by values
#
#     print('**Output of Mann-Whitney U-rank test**')
#     print(mannwhitneyu)
#     print('**Output Ordered from best p-values to worst**')
#     print(orderedp_mannwhitneyu)
#
#
# ############################################# Code Snippet 5 #######################################
#
# """Concatenate the quantification features all together in pandas DataFrame and run MRMR"""
# """This now done outside the mainHVN because need of another ENV for running pandas with MRMR
# -> should downgrade pandas in the futur"""
#

