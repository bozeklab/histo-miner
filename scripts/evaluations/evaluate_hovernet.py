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
import cv2
import copy

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
pathtofolder = config.eval_paths.eval_folders.parentfolder
instancemap_subfolder = config.eval_paths.eval_folders.instancemap_subfolder
classvector_subfolder = config.eval_paths.eval_folders.classvector_subfolder
prediction_subfolder = config.eval_paths.eval_folders.prediction_subfolder
gt_subfolder = config.eval_paths.eval_folders.gt_subfolder

instancemapfolder = pathtofolder + instancemap_subfolder
classvectorfolder = pathtofolder + classvector_subfolder 
predfolder = pathtofolder + prediction_subfolder
gtfolder = pathtofolder + gt_subfolder

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

# Count number of time remap problems occurs (should be None!)

count_legitmiss_pred = 0
count_legitcalculations = 0
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
		
		# load pred classmap
		prednpy_classmap = np.load(path)

		# load gt classmap
		gtname = gtfolder + fname
		gtnpy_hvnformat = np.load(gtname)
		gtnpy_classmap = gtnpy_hvnformat[:, :, 4]

		# predinstmap_name = instancemapfolder + fname
		# prednpy_instmap = np.load(predinstmap_name)
		# gtnpy_instmap = gtnpy_hvnformat[:, :, 3]

		# Generate inst map from class map: because the coordinates should match object,
		# Not like with predicted instance map

		prednpy_genebin_instmap = copy.deepcopy(prednpy_classmap)

		#Make a binary map
		for x, y in product(range(prednpy_genebin_instmap.shape[0]), range(prednpy_genebin_instmap.shape[1])):
			if prednpy_genebin_instmap[x,y] > 0 :
				prednpy_genebin_instmap[x,y] = 1

		# Connected components from the binary map
		prednpy_genebin_instmap = prednpy_genebin_instmap.astype('uint8')
		prednum_labels, prednpy_geneinstmap = cv2.connectedComponents(prednpy_genebin_instmap, connectivity=8)

		gtnpy_genebin_instmap = copy.deepcopy(gtnpy_classmap)

		#Make a binary map
		for x, y in product(range(gtnpy_genebin_instmap.shape[0]), range(gtnpy_genebin_instmap.shape[1])):
			if gtnpy_genebin_instmap[x,y] > 0 :
				gtnpy_genebin_instmap[x,y] = 1

		# Connected components from the binary map   
		gtnpy_genebin_instmap = gtnpy_genebin_instmap.astype('uint8')	 
		gtnum_labels, gtnpy_geneinstmap = cv2.connectedComponents(gtnpy_genebin_instmap, connectivity=8)

		# Initialize the list for the pq values
		list_pq_for_each_class = []

		for label in range(1, nbr_class + 1):

			if label in gtnpy_classmap and label in prednpy_classmap:
				count_legitcalculations += 1

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

				pred_newinstmap = np.zeros(prednpy_geneinstmap.shape)
				gt_newinstmap = np.zeros(gtnpy_geneinstmap.shape)

				for x, y in product(range(prednpy_classmap.shape[0]), range(prednpy_classmap.shape[1])):
					# we check if the class if not the one of the object and not background neither (for speed reason)

					# then we transform as background all instances not corresponding to the selected class
					if prednpy_classmap[x,y] == label:
						pred_newinstmap[x,y] = prednpy_geneinstmap[x,y]

					# we also update the gt file
					if gtnpy_classmap[x,y] == label:
						gt_newinstmap[x,y] = gtnpy_geneinstmap[x,y]

				# to ensure that the instance numbering is contiguous
				pred_newinstmap = remap_label(pred_newinstmap, by_size=False)
				gt_newinstmap = remap_label(gt_newinstmap, by_size=False)

				if np.max(pred_newinstmap) == 0 or np.max(gt_newinstmap) == 0:

					# problem on the remap label
					maxpred = np.max(pred_newinstmap)
					maxgt = np.max(gt_newinstmap)

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

			# To fasten calculation in the case of miss prediction
			# We know that pq = 0, no need to loose time calculating it
			elif label in gtnpy_classmap and label not in prednpy_classmap:
				list_pq_for_each_class.append(float(0))
				count_legitmiss_pred += 1

			# If ground truth is empty, skip from calculation 
			# this is the same approach in CellVIT code used to evaluate cellVIT validation
			# In this way Hovernet and cellVIT results are comparable
			# The miss classification of the prediction will be already taken into account
			# When comparing the GT of the error class
			# So we skip here not to calculate it twice
			# https://github.com/TIO-IKIM/CellViT/blob/main/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py
			# Code:  -------
            # if len(np.unique(target_nuclei_instance_class)) == 1:
            # pq_tmp = np.nan
            # -> there is only background and not the considered class
            # --------------
			elif label in prednpy_classmap and label not in gtnpy_classmap:
				list_pq_for_each_class.append(str(None))
				

		    # In this case the class is not in pred and not in GT, so no calculation needed
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


devink = True






        		

			

# we need to add a selection in case the gt as no cells of the label


