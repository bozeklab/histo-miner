#Lucas Sancéré -

import json
import os
import shutil
import numpy as np


### Utils Functions

def anaylser2featselect(folderpath: str, recnaming: list = ('no_recurrence','recurrence')):
    """
    Move all the output files from the tissue analysis to % folders
    to perform the feature selection    

    Parameters
    ----------
    folderpath: str
        Path to the folder containing all the json output of the tissue analysis.
        The json files could be in subdirectories as well
    recnaming: list
        TO COMPLETE ---------

    Returns
    -------
    """
    tissueanalyser_folder = folderpath + '/' + 'tissue_analysis_sorted'
    if not os.path.exists(tissueanalyser_folder):
        os.makedirs(tissueanalyser_folder)
    norec_analyse_folder = tissueanalyser_folder + '/' + 'no_recurrence'
    if not os.path.exists(norec_analyse_folder):
        os.makedirs(norec_analyse_folder)
    rec_analyse_folder = tissueanalyser_folder + '/' + 'recurrence'
    if not os.path.exists(rec_analyse_folder):
        os.makedirs(rec_analyse_folder)
    
    norecurrencestr = str(recnaming[0])
    recurrencestr = str(recnaming[1])
    for root, dirs, files in os.walk(folderpath):
        if files:  # Keep only the not empty lists of files
            # Because files is a list of file name here, and not a srting. You create a string with this:
            for file in files:
                path, extension = os.path.splitext(file)
                path_to_parentfolder, nameoffile = os.path.split(path)
                if extension == '.json' and 'analysed' in nameoffile:
                    if norecurrencestr in path:
                        #Move the json file to sort it with the other no recurrence WSIs and rename the file as well to make
                        # no_recurrence strings appear in the name
                        shutil.move(file, norec_analyse_folder + '/' + nameoffile.replace('analysed', 'no_recurrence_analysed'))
                    # Be careful if one of the two string is also included in the other one. So to avoid issue we do:
                    elif not norecurrencestr in path and recurrencestr in path: 
                        shutil.move(file, rec_analyse_folder + '/' + nameoffile.replace('analysed', 'recurrence_analysed'))
                    else: 
                        raise ValueError('Some features are not associated to a recurrence or norecurrence WSI classification.'
                                          'User need a way to discriminate between the two cases, using folder naming.'
                                          'For now, the strings choosen in recnaming argument are: {}'
                                          'User can change this argument and check recnaming docstring to have more information'
                                          .format(recnaming))