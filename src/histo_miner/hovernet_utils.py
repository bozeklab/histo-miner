#Lucas Sancéré -

import glob
import itertools
import os

import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import json
from scipy.io import loadmat



##### HOVERNET OUTPUT CONVERSIONS AND PROCESSING


def replacestring_json(file: str,
                       string2replace: str,
                       newstring: str,
                       string2replace_2r: str,
                       newstring_2r: str):
    """
    Open a json file and modify some caracters according to input args. Caracters contained in String2replace will be
    replace by the caracters contained in Newstring
    If you need to modify other cartacters
    (it should happen often because you need to be sure you don't loose some paranthesis for exemple)
    you can add the optionnal arguments string2replace_2r and newstring_2r working as the first ones.

    BECARFUL: if you apply twice this function to the same json you can modify it in a not expected way.

    Parameters:
    -----------
    file: str
        path to the .json file
    string2replace: str
    newstring: str
    string2replace_2r: str
        If a second change is needed, optionnal. _2r stands for 2 round.
    newstring_2r: str
        If a second change is needed, optionnal. _2r stands for 2 round.
    Returns:
    --------
    """
    with open(file, 'r') as filename:
        # try:
        content = filename.read()
        clean = content.replace(string2replace, newstring)  # cleanup here
        if string2replace_2r: # If there is another string you want ot replace in the file
            clean2 = clean.replace(string2replace_2r, newstring_2r)
            clean = clean2
        new_json = json.loads(clean)
        filenotempty = True

        # except json.decoder.JSONDecodeError:   #In case of an empty json in the folder
        #     filenotempty = False
        #     print('On file is empty or corrupted, file skipped')

    if filenotempty:
        with open(file, 'w') as filename:
            json.dump(new_json, filename)


def extr_matinstances(file: str, savename: str) -> np.ndarray:
    """
    Extract instance maps from .mat file

    Parameters:
    file: str
        path to the .mat file
    savename: str
        name of the file to save. By default it will be saved in the same folder.
        To change it add a path prefix to the file name. For example: 'Output/filename'
    Returns:
    --------
    npyarray: np.ndarray
        Array containing the data from the .mat file (the array is also saved as a numpy file)
    """
    result_mat = loadmat(file)
    inst_map = result_mat['inst_map']
    npinst_map = np.array(inst_map)
    pathtofolder, filename = os.path.split(file)
    np.save((pathtofolder + '/' + savename + '.npy'), npinst_map)
    return npinst_map


def extr_mattypes(file: str, savename: str) -> np.ndarray:
    """
    Extract types maps (classification) from .mat file

    Parameters:
    -----------
    file : str
        path to the .mat file
    savename : str
        name of the file to save. By default it will be saved in the same folder.
        To change it add a path prefix to the file name. For example: 'Output/filename'
    Returns:
    --------
    npyarray: np.ndarray
        Array containing the data from the .mat file (the array is also saved as a numpy file
    """
    result_mat = loadmat(file)
    inst_types = result_mat['inst_type']
    inst_types = np.array(inst_types)
    pathtofolder, filename = os.path.split(file)
    np.save((pathtofolder + '/' + savename + '.npy'), inst_types)
    return inst_types



##### TRAINING UTILS: CREATE TRAINING PATCHES

def gen_hvn_training_bigpatches_from_masks(images_npypannuke_format: np.ndarray,
                                           masks_npypannuke_format: np.ndarray,
                                           pathtosave: str):
    """
    Create a comprehensive traininig patch, containing all the training images at once. This trainingpatch is a numpy
    file that will be to use for HoVernet training. The format is the following (as described in HoverNetRepo:
    A 5 dimensional numpy array with channels [RGB, inst, type]. Here, type is the ground truth of the nuclear type.
    I.e every pixel ranges from 0-K,  where 0 is background and K is the number of classes.

    Once the big patch (also called comprehensive patch) is generated, one will need to divide into smaller patch,
    with one patch represent an image (and only one). To this, use another function (ExtractDataNpy).

    Name stands for Generate From mask an HVN comprehensive training patch.

    Parameters:
    -----------
    images_npypannuke_format: np.ndarray
        3D matrix containing image pixels and a RGB channels
    masks_npypannuke_format: np.ndarray
        an array of 6 channel instance-wise masks
        (0: Neoplastic cells,
        1: Inflammatory,
        2: Connective/Soft tissue cells,
        3: Dead Cells,
        4: Epithelial,
        5: Background)
    pathtosave: str
        path to the folder where the npy file will be saved
    Returns:
    --------
    """
    skinimages = np.load(images_npypannuke_format)
    skinmasks = np.load(masks_npypannuke_format)
    trainingpatch = skinimages
    trainingpatch = trainingpatch.astype(np.uint8)
    # create a matrix full of zeros with the same 2 first channel as skinimages
    addchannel1 = np.zeros((int(trainingpatch.shape[0]), int(trainingpatch.shape[1]), int(trainingpatch.shape[2]), 1))

    # Add 2 new channnels on axis3, the Insance Segmentation layer and the Type Classification layer
    trainingpatch = np.concatenate((trainingpatch, addchannel1), axis=3)
    trainingpatch = np.concatenate((trainingpatch, addchannel1), axis=3)

    trainingpatch = trainingpatch.astype(int)

    # Fill the Insance Segmentation layer and the Type Classification layer of trainingpatch
    for _class in range(skinmasks.shape[3]-1): # loop over classes, don't do the last one, as it is background class
     for _image in range(skinmasks.shape[0]): # loop over images
         # If the element is not 0 in any class, copy it in the intance segmentation layer
        for x, y in itertools.product(range(skinmasks.shape[1]), range(skinmasks.shape[2])):
            if skinmasks[_image, x, y, _class] != 0: #Check the non 0 pixels: pixels of cells
                # Creation of Instance Segmentation Layer
                trainingpatch[_image, x, y, 3] = int(skinmasks[_image, x, y, _class])
                # # Creation of Type Classification Layer
                # the first class is with label 0 and we want to start at label
                trainingpatch[_image, x, y, 4] = _class + 1
    #Change all 1 from the new channels to O
    trainingpatch[:, :, :, 3][trainingpatch[:, :, :, 3] == 1] = 0
    trainingpatch[:, :, :, 4][trainingpatch[:, :, :, 4] == 1] = 0
    np.save((pathtosave + 'trainingpatches.npy'), trainingpatch)


def genclassmap_from_2maps(instancemap_folder: str, classmapgt_folder: str, savename: str):
    """
    Generate a type map (or class map) for Hovernet Training, using an input Instance Map and another input Class Map.
    The new Class Map generated will have the segmented object of the input Instance Map, and for each instance object,
    the pixel values will be changed and will correspond to the pixel value with the most occurence
    in the same location of the map in the Input Class map.
    Naming of input files has to follow the rules describe in parameters section.

    Then the new Class map generated will be saved in the instancemap_folder arg according to the savename arg choosen.

    Parameters:
    -----------
    instancemap_folder: str
        path to the .npy InstanceMap files. 'InstanceMap' caracters has to be in the name of the files. For instance:
        sample1_InstanceMap.npy
    classmapgt_folder: str
        path to the .npy ClassMap grountruth files. The names as to be the same as the InstanceMap files but with
        'ClassMap' caracters instead of 'InstanceMap' caracters. Following previous example: sample1_ClassMap.npy
    savename: str
        Part of the name of the file to save. The name will be on the form of InstanceMap Files name but replacing
        'InstanceMap' caracters by the ones choosen
    Returns:
    --------
    """
    instance_maps_path = os.path.join(instancemap_folder, '*.npy')
    files_instance_maps = glob.glob(instance_maps_path)
    for fname in tqdm(files_instance_maps):
        #Load the Groundtruth class image
        instancemappath = os.path.splitext(fname)[0] #remove extension from the path
        instancemapname = os.path.split(instancemappath)[1] #keep only name of the image
        class_map_gt_path = classmapgt_folder + instancemapname.replace('InstanceMap', 'ClassMap') \
                         + os.path.splitext(fname)[1]  # add the extension at the end
        if not os.path.exists(class_map_gt_path): #check if there is an associated classmap, if not continue the loop
            continue
        class_gt = np.load(class_map_gt_path)
        #Load all the regions from input Instance Map
        instance_map = np.load(fname)
        regions = regionprops(instance_map)
        #New numpy array to generate
        outputclassmap = np.zeros(class_gt.shape) # Create a new map full of 0 with the same shape as the 2 other images

        # Find the most represented class in the Grountruth image
        flatt_class_gt = np.ndarray.flatten(class_gt)
        countspixelvalues = np.bincount(flatt_class_gt)  # List of counts of pixels from the whole array
        countspixelvalueslist = list(countspixelvalues)
        del countspixelvalueslist[0] # The background class is not taken into account
        most_represented_class = np.argmax(countspixelvalueslist) + 1
        # We add one because the background class was deleted and then class N is now at index N-1
        print('For the Image', os.path.split(fname)[1],
              ' the Most represented class (in number of pixels not objects) is Class number:', most_represented_class)

        for region_id, region in enumerate(regions):
            # Create a vector that will collect all pixel values for the same location of the object but
            # in Class Grountruth Image
            class_values_gt = []
            coordinate_list = region.coords

            for coordinates in coordinate_list:
                if class_gt[coordinates[0], coordinates[1]] != 0: #Don't keep backround values
                    class_values_gt.append(class_gt[coordinates[0], coordinates[1]])
            if not class_values_gt: # If the list is empty because the region is only filled with background pixels
                classlabel = most_represented_class
            else:  # If the list is not empty because the region is not only filled with background pixels
                # The class choosen is the one whith the highest number of occurence in classlabelValuesGt
                classlabel = max(class_values_gt, key=class_values_gt.count)

            # Needs to restart the loop again, because the first needs to end to set up classlabel variable
            for coordinates in coordinate_list:
                outputclassmap[coordinates[0], coordinates[1]] = classlabel # Give Class Pixel values to the new map
        corefname, filename = os.path.split(fname)[0], os.path.split(fname)[1]
        np.save((corefname + filename.replace('InstanceMap', savename)), outputclassmap)


def gen_hvn_training_patches(rawimage_folder: str,
                             instancemap_folder: str,
                             classmap_folder: str,
                             pathtosave: str):
    """
    Generate a training patch per image for Hovernet training.
    To do so, concatenate together the raw RGB image in numpy array format,
    the GT instance class, and the GT class (also called type) map.

    Parameters:
    -----------
    rawimage_folder: str
        path to the .npy RawImage files. 'RawImage' caracters has to be in the name of the files. For instance:
        sample1_RawImage.npy
    instancemap_folder: str
        path to the .npy InstanceMap files. The names has to be the same as the RawImage files but with
        'InstanceMap' caracters instead of 'RawImage' caracters. Following previous example: sample1_InstanceMap.npy
    classmap_folder: str:
        path to the .npy ClassMap files. The names has to be the same as the InstanceMap files but with
        'ClassMap' caracters instead of 'RawImage' caracters. Following previous example: sample1_ClassMap.npy
    pathtosave: str
        path to the folder where the npy file will be saved
    Returns:
    --------
    """
    rawimage_folder_path = os.path.join(rawimage_folder, '*.npy')
    files_raw_images = glob.glob(rawimage_folder_path)
    for fname in tqdm(files_raw_images):
        rawimagepath = os.path.splitext(fname)[0] # remove extension from the path
        rawimagename = os.path.split(rawimagepath)[1] # keep only name of the image
        output_path = pathtosave + rawimagename.replace('RawImage', '') + '.npy'
        if not os.path.exists(output_path): # Not to re-create a training patch that already exists
            instance_map_path = instancemap_folder + \
                              rawimagename.replace('RawImage', 'InstanceMap') + os.path.splitext(fname)[1]
            # We added the extension at the end
            class_map_path = classmap_folder + rawimagename.replace('RawImage', 'ClassMap') + os.path.splitext(fname)[1]
            # We added the extension at the end

            rgb_array = np.load(fname)
            instance_map = np.load(instance_map_path)
            class_map = np.load(class_map_path)
            # Add a third dimension to then concatenate to RGB array:
            expanded_instance_map = np.expand_dims(instance_map, axis=2)
            # Add a third dimension to then concatenate to RGB array:
            expanded_class_map = np.expand_dims(class_map, axis=2)
            # Create the "5D array" which is 5 slices of a 2D Matrix (so to me more 3D)
            trainingpatch = np.concatenate((rgb_array, expanded_instance_map, expanded_class_map), axis=2)
            np.save(output_path, trainingpatch)
        else:
            print('Warning: You tried to generate a training patch that was already generated previously'
                  ', skipping...')