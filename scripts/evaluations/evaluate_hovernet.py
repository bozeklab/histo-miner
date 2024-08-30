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

from src.histo_miner.evaluations import get_fast_pq, remap_label, pairing_cells
from src.histo_miner.hovernet_utils import classmap_from_classvector
import joblib
import cv2
import copy
import sklearn.metrics

# add the script to transform mat files into npy for instances and npy for types (using hovernet utils)


#############################################################
## Load configs parameter
#############################################################

# Import parameters values from config file by generating a dict.
# The lists will be imported as tuples.
with open("./../../configs/models/eval_hovernet.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# Create a config dict from which we can access the keys with dot syntax
config = attributedict(config)
pathtofolder = config.eval_paths.eval_folders.parentfolder
instancemap_subfolder = config.eval_paths.eval_folders.instancemap_subfolder
classvector_subfolder = config.eval_paths.eval_folders.classvector_subfolder
prediction_subfolder = config.eval_paths.eval_folders.prediction_subfolder
gt_subfolder = config.eval_paths.eval_folders.gt_subfolder
save_folder = config.eval_paths.eval_folders.save_folder

calculate_pq = config.parameters.bool.calculate_pq
calculate_confusion = config.parameters.bool.calculate_confusion
create_classmap = config.parameters.bool.create_classmap
nbr_classes = config.parameters.int.nbr_classes

# Create full path based on given config
instancemapfolder = pathtofolder + instancemap_subfolder
classvectorfolder = pathtofolder + classvector_subfolder 
predfolder = pathtofolder + prediction_subfolder
gtfolder = pathtofolder + gt_subfolder



#############################################################
## Create Class Map if needed
#############################################################

# Generate class map from TypeMaps output from hovernet! The gt classmap is already generated when
# the training set is created
if create_classmap:
	classmap_from_classvector(instancemapfolder=instancemapfolder, 
							  classvectorfolder=classvectorfolder, 
							  savename='')
	print("Createclassmap ran. Script stopped. Change config to prevent Class Map creation")
	sys.exit()




#############################################################
## PQ Calculation. 
#############################################################


if calculate_pq:
	
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

			# load pred instmap
			predinstmap_name = instancemapfolder + fname
			prednpy_instmap = np.load(predinstmap_name)
			 
			# load gt instmap
			gtnpy_instmap = gtnpy_hvnformat[:, :, 3]

			# Initialize the list for the pq values
			list_pq_for_each_class = []

			# We do the pq calculations for each class 
			for label in range(1, nbr_classes + 1):

				if label in gtnpy_classmap and label in prednpy_classmap:
					count_legitcalculations += 1

					pred_newinstmap = np.zeros(prednpy_instmap.shape)
					gt_newinstmap = np.zeros(gtnpy_instmap.shape)

					for x, y in product(range(prednpy_classmap.shape[0]), range(prednpy_classmap.shape[1])):
						# we check if the class if not the one of the object and not background neither (for speed reason)

						# then we transform as background all instances not corresponding to the selected class
						if prednpy_classmap[x,y] == label:
							pred_newinstmap[x,y] = prednpy_instmap[x,y]

						# we also update the gt file
						if gtnpy_classmap[x,y] == label:
							gt_newinstmap[x,y] = gtnpy_instmap[x,y]

					# to ensure that the instance numbering is contiguous
					pred_newinstmap = remap_label(pred_newinstmap, by_size=False)
					gt_newinstmap = remap_label(gt_newinstmap, by_size=False)

					if np.max(pred_newinstmap) == 0 or np.max(gt_newinstmap) == 0:

						# problem on the remap label
						maxpred = np.max(pred_newinstmap)
						maxgt = np.max(gt_newinstmap)

						list_pq_for_each_class.append(str(None))
						count_remap_prblms += 1

						# here raise an error instead (find the best one to raise)


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

# Read variables directly on debugger - no save
debugink = True


#############################################################
## Confusion matrix Calculation. 
#############################################################


if calculate_confusion:

	alltrue_labels = list()
	allpred_labels = list()


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

			# load pred instmap
			predinstmap_name = instancemapfolder + fname
			prednpy_instmap = np.load(predinstmap_name)
			 
			# load gt instmap
			gtnpy_instmap = gtnpy_hvnformat[:, :, 3]

			# to ensure that the instance numbering is contiguous
			pred_newinstmap = remap_label(prednpy_instmap, by_size=False)
			gt_newinstmap = remap_label(gtnpy_instmap, by_size=False)

			paired_true_id, paired_pred_id = pairing_cells(gt_newinstmap, pred_newinstmap)

		    # Maybe ADD sanity checks here
		    # first the len of paired should be the same
		    # and no element with 0s in the class map 

			for label in paired_true_id:
				indices = np.argwhere(gt_newinstmap == label)
				centroid = indices.mean(axis=0)
				centroid = [int(centroid[0]), int(centroid[1])]

				true_class = gtnpy_classmap[centroid[0], centroid[1]]
				alltrue_labels.append(true_class)

			for label in paired_pred_id: 
				indices = np.argwhere(pred_newinstmap == label)
				centroid = indices.mean(axis=0)
				centroid = [int(centroid[0]), int(centroid[1])]

				true_class = prednpy_classmap[centroid[0], centroid[1]]
				allpred_labels.append(true_class)

	# Remove 0s (we evaluate only classification here)

	# We need a sanity check by removing all background class prediction (not detected)
	idxzeros_true_labels = [idx for idx, value in enumerate(alltrue_labels) if value == 0]
	idxzeros_pred_labels = [idx for idx, value in enumerate(allpred_labels) if value == 0]

	# We create a list of indexes where there is a 0 in at least one of the vectors
	unique_idxzeros = set(idxzeros_true_labels + idxzeros_pred_labels)
	idx_zeros = sorted(list(unique_idxzeros))

	# We refine the prediction by removing item where they were a 0, but for both lists not to change order
	alltrue_labels = [item for i, item in enumerate(alltrue_labels) if i not in idx_zeros]
	allpred_labels = [item for i, item in enumerate(allpred_labels) if i not in idx_zeros]

	# Calculate confusiton matrices

	conf_mat = sklearn.metrics.confusion_matrix(alltrue_labels, allpred_labels)

	conf_mat_true_normalized = sklearn.metrics.confusion_matrix(
    	alltrue_labels,
    	allpred_labels,
    	normalize='true')
	conf_mat_pred_normalized = sklearn.metrics.confusion_matrix(
    	alltrue_labels,
    	allpred_labels,
    	normalize='pred')

	conf_mat_name = 'confusion_matrix.npy'
	conf_mat_name_truenorm = 'confusion_matrix_truenorm.npy'
	conf_mat_name_prednorm = 'confusion_matrix_prednorm.npy'

	if not os.path.exists(save_folder):
		os.mkdir(save_folder)

	np.save(save_folder + '/' + conf_mat_name, conf_mat)
	np.save(save_folder + '/' + conf_mat_name_truenorm, conf_mat_true_normalized)
	np.save(save_folder + '/' + conf_mat_name_prednorm, conf_mat_pred_normalized)

	print('Confusion matrices files saved in folder {}'.format(save_folder))

