

import csv  
import os
import numpy as np
 
pathtosave_file = '/data/shared/scc/histo-miner/data_analysis/crea_patient_ID_lst/orderedlist.csv'
pathtojsons_folder = '/data/shared/scc/histo-miner/data_analysis/crea_patient_ID_lst/recurrence/Regensburg'

# filenamelist = list()
# for root, dirs, files in os.walk(pathtojsons_folder):
#     if files:  # Keep only the not empty lists of files
#         # Because files is a list of file name here, and not a srting. You create a string with this:
#         for file in files: 
#         	filename = os.path.splitext(file)
#         	filenamelist.append(filename)

# filenamelist.sort()
# with open(pathtosave_file, 'w', encoding='UTF8') as f:
#      writer = csv.writer(f)
#      for filename in filenamelist:
# 	     writer.writerow(filename)



study = np.load("/data/lsancere/Data_General/Collaborators_Dataset/cSCC_CPI_all/Analyses_cSCC_CPI/feature_selection/patientids.npy", allow_pickle=True)
pass
	        	
