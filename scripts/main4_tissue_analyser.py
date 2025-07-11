#Lucas Sancéré -

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(script_dir)   # subdir/
sys.path.append(parent_dir)   # project/

import json
import yaml
from attrdictionary import AttrDict as attributedict

from src.histo_miner import tissue_analyser as analyser
from src.histo_miner.utils.misc import NpEncoder



###################################################################
## Load configs parameter
###################################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open(script_dir + "/../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.tissue_analyser_main
pathanalyserout = config.paths.folders.tissue_analyser_output
calculate_morphologies = config.parameters.bool.calculate_morphologies
calculate_vicinity = config.parameters.bool.calculate_vicinity
calculate_distances = config.parameters.bool.calculate_distances
maskmap_downfactor = config.parameters.int.maskmap_downfactor
default_tumormargin = config.parameters.int.default_tumormargin
maskmapext = str(config.parameters.str.maskmap_ext)
selectedcls_ratio = list(config.parameters.lists.selectedcls_ratio)
selectedcls_ratiovic = list(config.parameters.lists.selectedcls_ratiovic)
selectedcls_dist = list(config.parameters.lists.selectedcls_dist)
classnames = list(config.parameters.lists.classnames)
classnames_injson = list(config.parameters.lists.classnames_injson)



#################################################################
## Tissue Analysis, Extraction of features
#################################################################

"""Tissue Analysis"""


if not os.path.exists(pathanalyserout):
    os.mkdir(pathanalyserout)
        

########  Create list with the paths of the files to analyse
jsonfiles = list()
for root, dirs, files in os.walk(pathtofolder):
    if files:  # Keep only the not empty lists of files
        # Because files is a list of file name here, and not a single path string. Create a string with this:
        for file in files:
            namewoext, extension = os.path.splitext(file)
            filepath = root + '/' + file
            if extension == '.json' and not any(keyword in namewoext for keyword in ['data', 'analysed', 'cellnbr']):
                # 'data' not in name of files means it is not a json file coming from the analysis
                if os.path.exists(pathanalyserout + namewoext + '_analysed.json') or os.path.exists(
                    pathanalyserout + namewoext + '_vicinity_analysed.json'
                    ): 
                    print('Detected an already processed file:', file)
                    continue
                else:
                    print('Detected hovernet output json file not already analysed:', file)
                    jsonfiles.append(filepath)



######## Process the files
for jsonfile in jsonfiles:
   # Creating the dictionnary to count the cells using countjson function:
    pathwoext = os.path.splitext(jsonfile)[0]
    namewoext = os.path.splitext(os.path.split(jsonfile)[1])[0] 
    pathtosavewoext = pathanalyserout + '/' +  namewoext

    print('********** \nProcess file:', pathwoext)
    print('Process count of cells per cell type in the whole slide image...')
    classcountsdict = analyser.counthvnjson(
                                 jsonfile, 
                                 classnames_injson, 
                                 classnameaskey=classnames)
    allcells_in_wsi_dict = classcountsdict
    print('\nAllcells_inWSI_dict generated as follow:', allcells_in_wsi_dict)


    # Write information inside a json file about cell nbr per class and save it
    with open(pathtosavewoext + '_cellnbr.json', 'w') as outfile:
        json.dump(allcells_in_wsi_dict, outfile, cls=NpEncoder)

    print('Json file written :', pathwoext + '_cellnbr.json \n**********')


    # Create the path to Mask map binarized and Class JSON and save it into a variable
    if os.path.exists(jsonfile) and os.path.exists(jsonfile.replace('.json', maskmapext)):

        # Create path for the maskmap
        maskmappath = jsonfile.replace('.json', maskmapext)
        print('Detected mask file:', maskmappath)

        # Analysis
        # Tumor Area -----------
        tumor_tot_area = analyser.count_pix_value(maskmappath, 255) * (maskmap_downfactor ** 2)
        print('Process cells identification '
              '(number of cells and tot area of cells) inside tumor regions...')

        # Calculation of percentages of cells and ratios -----
        if calculate_vicinity:
            cells_inregion_dict = analyser.cells_classandmargin_classjson(
                                                            maskmappath, 
                                                            jsonfile,
                                                            selectedcls_ratio,
                                                            selectedcls_ratiovic,
                                                            maskmapdownfactor=maskmap_downfactor,
                                                            classnameaskey=classnames,
                                                            tumormargin=default_tumormargin
                                                            )
        
        else:
            cells_inregion_dict = analyser.cells_insidemask_classjson(
                                                            maskmappath, 
                                                            jsonfile, 
                                                            selectedcls_ratio,
                                                            maskmapdownfactor=maskmap_downfactor,
                                                            classnameaskey=classnames
                                                            )

        print('cells_inregion_dict generated as follow:', cells_inregion_dict)

        # Calculation of morphology features -----
        morph_inregion_dict = dict()

        if calculate_morphologies:

            if calculate_vicinity:
                
                morph_inregion_dict = analyser.morph_classandmargin_classjson(
                                                                maskmappath, 
                                                                jsonfile,
                                                                selectedcls_ratio,
                                                                selectedcls_ratiovic,
                                                                maskmapdownfactor=maskmap_downfactor,
                                                                classnameaskey=classnames,
                                                                tumormargin=default_tumormargin
                                                                )
            
            else:
                morph_inregion_dict = analyser.morph_insidemask_classjson(
                                                                maskmappath, 
                                                                jsonfile, 
                                                                selectedcls_ratio,
                                                                maskmapdownfactor=maskmap_downfactor,
                                                                classnameaskey=classnames
                                                                )
    
            print('morph_inregion_dict generated!')
            # a bit too large to pring: print('morph_inregion_dict generated as follow:', morph_inregion_dict)
            
        
        # Distance calculation --------
        # Needs to create an empty dict even if the distances are not calculated
        cellsdist_inmask_dict = dict()
        
        if calculate_distances: 
            print('Process distance calculcations inside tumor regions...')  
            cellsdist_inmask_dict = analyser.mpcell2celldist_classjson(
                jsonfile, 
                selectedcls_dist,
                cellfilter='Tumor',
                maskmap=maskmappath,
                maskmapdownfactor=maskmap_downfactor,
                tumormargin=None)

            print('Cellsdist_inmask_dict generated as follow:', cellsdist_inmask_dict)


    else:
        tumor_tot_area = 0
        cells_inregion_dict = None
        morph_inregion_dict = None
        cellsdist_inmask_dict = None
        print('Cells_inregion_dict not generated')
        print('Morph_inregion_dict not generated')
        print('Cellsdist_inmask_dict not generated')



    jsondata = analyser.hvn_outputproperties(
                                    allcells_in_wsi_dict,
                                    cells_inregion_dict,
                                    morph_inregion_dict, 
                                    cellsdist_inmask_dict,
                                    masktype='Tumor',
                                    calculate_vicinity=calculate_vicinity,
                                    areaofmask=tumor_tot_area, 
                                    selectedcls_ratio=selectedcls_ratio,
                                    selectedcls_ratiovicinity=selectedcls_ratiovic, 
                                    selectedcls_dist=selectedcls_dist
                                    )


    # Write information inside a json file and save it for the whole analysis
    if calculate_vicinity:
        with open(pathtosavewoext + '_vicinity_analysed.json', 'w') as outfile:
            json.dump(jsondata, outfile, cls=NpEncoder)

        print('Json file written :', pathtosavewoext + '_vicinity_analysed.json \n**********')

    else: 
        with open(pathtosavewoext + '_analysed.json', 'w') as outfile:
            json.dump(jsondata, outfile, cls=NpEncoder)

        print('Json file written :', pathtosavewoext + '_analysed.json \n**********')



print('Tissue Analysis Done')
