#Lucas Sancéré -

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import os
import glob
import numpy as np
import yaml
from tqdm import tqdm
from attrdictionary import AttrDict as attributedict
from itertools import product

from src.histo_miner.evaluations import get_fast_pq, remap_label
from src.histo_miner.hovernet_utils import classmap_from_classvector
import joblib


# add the script to transform mat files into npy for instances and npy for types (using hovernet utils)


#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/histo_miner_pipeline.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.paths.folders.workingfolder 


instancemapfolder = pathtofolder + '/InfonVal_output/mat/InstancesMaps/'
classvectorfolder = pathtofolder + '/InfonVal_output/mat/InstancesTypes/'
predfolder = pathtofolder + '/InfonVal_output/mat/InstancesTypes/ClassMaps/'
gtfolder = pathtofolder + 'Labels/'


createclassmap = False
nbr_class = 5

if createclassmap:
	classmap_from_classvector(instancemapfolder=instancemapfolder, 
							  classvectorfolder=classvectorfolder, 
							  savename='')
	print("Createclassmap ran. Script stopped.")
	# sys.exit()
	
# Initializa the lists of PQs value for each class (here 5)
# Do a loop later on in case we have more than 5 classes
pq_for_class1 = []
pq_for_class2 = []
pq_for_class3 = []
pq_for_class4 = []
pq_for_class5 = []

# Count number of time rmp probalems as an issue

count_miss_pred = 0
countlegitcalculations = 0
count_remap_prblms = 0


count_remap_prblms_cls1 = 0
count_remap_prblms_cls2 = 0
count_remap_prblms_cls3 = 0
count_remap_prblms_cls4 = 0
count_remap_prblms_cls5 = 0

countlegitcalculations_cls1 = 0
countlegitcalculations_cls2 = 0
countlegitcalculations_cls3 = 0
countlegitcalculations_cls4 = 0
countlegitcalculations_cls5 = 0

files = os.path.join(predfolder, '*.npy')
files = glob.glob(files)
for path in tqdm(files):
    if os.path.exists(path):
    	#keep name of the file and not only path
	    fname = os.path.split(path)[1]
	    print('Patch beeing processed is,',fname)

		# load pred classmap and instmap
	    prednpy_classmap = np.load(path)
	    predinstmap_name = instancemapfolder + fname
	    prednpy_instmap = np.load(predinstmap_name)

	    # load gt classmap and instmap 
	    gtname = gtfolder + fname 
	    gtnpy_hvnformat = np.load(gtname)
	    gtnpy_instmap = gtnpy_hvnformat[:, :, 3]
	    gtnpy_classmap = gtnpy_hvnformat[:, :, 4]

		# Initialize the list for the pq values
	    list_pq_for_each_class = []

	    for label in range(1, nbr_class + 1):
	    	

	    	if label in gtnpy_classmap and label in prednpy_classmap:

	    		
	    		countlegitcalculations += 1

	    		#very hugly but needed quickly for the presentation
	    		if label == 1:
	    			countlegitcalculations_cls1 += 1
	    		if label == 2:
	    			countlegitcalculations_cls2 += 1
	    		if label == 3:
	    			countlegitcalculations_cls3 += 1
	    		if label == 4:
	    			countlegitcalculations_cls4 += 1
	    		if label == 5:
	    			countlegitcalculations_cls5 += 1

	    		pred_newinstmap = prednpy_instmap
	    		gt_newinstmap = gtnpy_instmap

	    		for x, y in product(range(prednpy_classmap.shape[0]), range(prednpy_classmap.shape[1])):

					# we check if the class if not the one of the object and not background neither
					# then we transform as background all instances not corresponding to the selected class 
	    			if prednpy_classmap[x,y] != 0 and prednpy_classmap[x,y] != label:
	    				pred_newinstmap[x,y] = 0
					# we also update the gr file	
	    			if gtnpy_classmap[x,y] != 0 and gtnpy_classmap[x,y] != label:
	    				gt_newinstmap[x,y] = 0

				 

				# to ensure that the instance numbering is contiguous
	    		pred_newinstmap = remap_label(pred_newinstmap, by_size=False)
	    		gt_newinstmap = remap_label(gt_newinstmap, by_size=False)

	    		if np.max(pred_newinstmap) == 0 or np.max(gt_newinstmap) == 0:
	    			# problem on the remap label
	    			list_pq_for_each_class.append(str(None))

	    			count_remap_prblms += 1
	    			
	    		    #very hugly but needed quickly for the presentation
		    		if label == 1:
		    			count_remap_prblms_cls1 += 1
		    		if label == 2:
		    			count_remap_prblms_cls2 += 1
		    		if label == 3:
		    			count_remap_prblms_cls3 += 1
		    		if label == 4:
		    			count_remap_prblms_cls4 += 1
		    		if label == 5:
		    			count_remap_prblms_cls5 += 1

	    		else:
					#calculate panoptic quality results
		    		panoptic_evals = get_fast_pq(pred_newinstmap, gt_newinstmap)
					# exctract value of panoptic quality
		    		pq_value = panoptic_evals[0][2]

		    		list_pq_for_each_class.append(pq_value)

	    	elif label in gtnpy_classmap and label not in prednpy_classmap:
	    		list_pq_for_each_class.append(float(0))
	    		count_miss_pred += 1


	    	else:
	    		list_pq_for_each_class.append(str(None)) 

	    		

	    # Do a loop later on in case we have more than 5 classes
	    pq_for_class1.append(list_pq_for_each_class[0])
	    pq_for_class2.append(list_pq_for_each_class[1])
	    pq_for_class3.append(list_pq_for_each_class[2])
	    pq_for_class4.append(list_pq_for_each_class[3])
	    pq_for_class5.append(list_pq_for_each_class[4])



# transform the lists into numpys and remove the None 
pq_class1 = np.asarray(pq_for_class1)
pq_class1 = pq_class1[pq_class1 != str(None)]
pq_class1 = pq_class1.astype('float32')
pq_class2 = np.asarray(pq_for_class2)
pq_class2 = pq_class2[pq_class2 != str(None)]
pq_class2 = pq_class2.astype('float32')
pq_class3 = np.asarray(pq_for_class3)
pq_class3 = pq_class3[pq_class3 != str(None)]
pq_class3 = pq_class3.astype('float32')
pq_class4 = np.asarray(pq_for_class4)
pq_class4 = pq_class4[pq_class4 != str(None)]
pq_class4 = pq_class4.astype('float32')
pq_class5 = np.asarray(pq_for_class5)
pq_class5 = pq_class5[pq_class5 != str(None)]
pq_class5 = pq_class5.astype('float32')


# calculate the mean for each list
mean_pq_class1 = np.mean(pq_class1)
mean_pq_class2 = np.mean(pq_class2)
mean_pq_class3 = np.mean(pq_class3)
mean_pq_class4 = np.mean(pq_class4)
mean_pq_class5 = np.mean(pq_class5)


devink = 0






        		

			

			# we need to add a selection in case the gt as no cells of the label


