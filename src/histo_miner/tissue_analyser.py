#Lucas Sancéré -

import copy
import glob
import json
import math
import multiprocessing as mp
import os
from ast import literal_eval
from itertools import product

import PIL
import cv2
import imagesize
import numpy as np
import shapely.geometry
import yaml
from PIL import Image
from attrdict import AttrDict as attributedict
from skimage.measure import regionprops, label
from skimage.util import view_as_blocks
from sklearn.preprocessing import binarize
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 10000000000000


## Functions


def count_pix_value(file: str, value: int) -> int:
    """
    Count number of pixels with specified value.

     Parameters:
    -----------
    file: str
        path to the image. The extension of the image can be any PILLOW supported format
    value: int
        Pixel Value of pixels user wants to count
    Returns:
    --------
    pixelcounter: int
        Number of pixels with the specified value
    """
    image = Image.open(file)  # PIL support lots of formats
    array = np.array(image)  # Transform PIL Image format into numpy array
    # newarray = array
    pixelcounter = 0
    for x, y in product(range(array.shape[0]), range(array.shape[1])):
        if array[x, y] == value:
            pixelcounter += 1
    return pixelcounter


def countjson(file: str, searchedwords: list) -> dict:
    """
    Count occurence of different words in a json file. The list of words is provided by searchwords.

    Parameters
    ----------
    file: str
        path to the .json file
    searchedwords: list
        list of words user wants to count occurance for
    Returns
    -------
    wordcountsdict: dict
        dict countaining as key the different words cf searchwords list and as value the number of occurence of the
        key word
    """
    wordcountsdict = dict()
    with open(file, 'r') as filename:
        data = filename.read()
        for word in tqdm(searchedwords):
            wordcount = data.count(word)
            wordcountsdict[word] = wordcount
    return wordcountsdict


def counthvnjson(file: str, searchedwords: list, classnameaskey: list = None) -> dict:
    """
    Count occurence of different cell class in a json output from hovernet predictions. 
    The list of cell classes is provided by searchwords.

    Parameters
    ----------
    file: str
        path to the .json file
    searchedwords: list
        list of strings user wants to count occurence for
    classnameaskey: list, optional
        List object containing the name of the classes to replace their number in the final output.
        To say it an other way numinstanceperclass list will be replaced by a dictionnary with class names as keys.
    Returns
    -------
    wordcountsdict: dict
        dict countaining as key the different words cf searchwords list and as value the number of occurence of the
        key word
    """
    wordcountsdict = dict()
    with open(file, 'r') as filename:
        data = filename.read()
        for word in tqdm(searchedwords):
            wordcount = data.count(word)
            wordcountsdict[word] = wordcount
    if not classnameaskey: 
        return wordcountsdict
    else:
        wordcountvalues = list(wordcountsdict.values())
        wordcountsdict = dict(zip(['Background'] + classnameaskey, wordcountvalues))
        return wordcountsdict


def cells_insidemask_classjson(maskmap: str, classjson: str, selectedclasses: list,
                                    maskmapdownfactor: int = 1, classnameaskey: list = None) -> dict:
    """
    Calculate number of instances from each class contained in "selectedclasses", that are inside the mask from maskmap.
    Maskmap and classjson containing information of all json class are used as input.

    Note: the json has to be correctly formated, see ForHoverNet.MainHvn
    Codesnippet 1: Update each json files to be compatible with QuPath

    Parameters
    ----------
    maskmap: str
        Path to the binary image, mask of a specific region of the original image.
        The image must be in PIL supported format.
    classjson: str
        Path to the json file (for instance output of hovernet then formatted according to the note above).
        It must contains, inside each object ID key (numbers):
        - a 'centroid' key, containing the centroid coordinates of the object
        - a 'type" key, containing the class type of the object (classification)
        - a 'contour' key, containing the coordinates of border points of the object
    selectedclasses: list
        List containing the different class from what the user wants the caclulation to be done.
    maskmapdownfactor: int, optional
        Set what was the downsample factor when the maksmap (tumor regions) were generated
    classnameaskey: list, optional
        List object containing the name of the classes to replace their number in the final output.
        To say it an other way numinstanceperclass list will be replaced by a dictionnary with class names as keys.
    Returns
    -------
    outputdict: dict
        Dictionnary containing NUMBER TO DEFINE keys:
        - "masktotarea": int : total area in pixel of the mask
        - "list_numinstanceperclass": list : number of instances inside the mask region for each selected class
    """
    with open(classjson, 'r') as filename:
        classjson = json.load(filename)  # data must be a dictionnary
    # Loading of mask map is really heavy (local ressources might be not enough)
    maskmap = Image.open(maskmap)
    maskmap = np.array(maskmap)  # The maskmap siize is not the same as the input image, it is downsampled
    # Check shape of the input file
    if len(maskmap.shape) != 2 and len(maskmap.shape) != 3:
        raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. Image type not supported ')
    elif len(maskmap.shape) == 3:
        if maskmap.shape[2] == 3:
            maskmap = maskmap[:, :, 0]  # Keep only one channel of the image if the image is 3 channels (RGB)
        else:
            raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. Image type not supported ')
    # Extract centroids + Class information for each nucleus in the dictionnary and also countour coordinates
    # loop on dict
    allnucl_info = [[int(classjson[nucleus]['centroid'][0]),
                     int(classjson[nucleus]['centroid'][1]),
                     classjson[nucleus]['type'],
                     classjson[nucleus]['contour']]
                    for nucleus in classjson.keys()]  # should extract only first level keys
    # Idea of creating a separated list for coordinates is too keep for now
    # nucl_coordinates = [classjson[nucleus]['contour'] for nucleus in classjson.keys()
    numinstanceperclass = np.zeros(len(selectedclasses))
    totareainstanceperclass = np.zeros(len(selectedclasses))
    maskmapdownfactor = int(maskmapdownfactor)
    # Normally not necessaty but depend how the value is given as output (could be str)

    for count, nucl_info in tqdm(enumerate(allnucl_info)):
        # Check if cell is inside tumor (mask) region
        if maskmap[int(nucl_info[1] / maskmapdownfactor), int(nucl_info[0] / maskmapdownfactor)] == 255:
            if nucl_info[2] in selectedclasses:  # Chech the class of the nucleus
                indexclass = selectedclasses.index(nucl_info[2])
                numinstanceperclass[indexclass] += 1
                instancenumber = count
                # Add Area Calculation by importing all the edges of polygons
                polygoninfo = shapely.geometry.Polygon(nucl_info[3])
                instancearea = polygoninfo.area
                totareainstanceperclass[indexclass] += instancearea
    # print('count=', count)

    numinstanceperclass = numinstanceperclass.astype(int)
    totareainstanceperclass = totareainstanceperclass.astype(int)

    # Aggregate all the informations about number and areas of cells in the masked regions
    # create a dictionnary of list if classnameaskey is not given as input
    # (then the class keys corresponds to the index of the value in the list)
    # or a dictionnary of dictionnaries if classnameaskey is given
    if not classnameaskey:
        outputlist = {"list_numinstanceperclass": numinstanceperclass,
                      "list_totareainstanceperclass": totareainstanceperclass}
        return outputlist
    else:
        #update classnames with the selectedclasses
        updateclassnameaskey = [classnameaskey[index-1] for index in selectedclasses]
        #now we use zip method to match class number with its name 
        numinstanceperclass_dict = dict(zip(updateclassnameaskey, numinstanceperclass))
        totareainstanceperclass_dict = dict(zip(updateclassnameaskey, totareainstanceperclass))
        outputdict = {"dict_numinstanceperclass": numinstanceperclass_dict,
                      "dict_totareainstanceperclass": totareainstanceperclass_dict}
        return outputdict


def cell2celldist_classjson(classjson: str, selectedclasses: list,
                            cellfilter: str = 'Tumor',
                            maskmap: str = '',
                            maskmapdownfactor: int = 1,
                            tumormargin: int = None) -> list:
    """
    Use single processing to calculate average of the closest neighboor distances between all cells from one type
    and all cells from another.
    Calculate this average for all cell types pairs.

    Parameters
    ----------
    classjson: str
        Path to the json file (for instance output of hovernet then formatted according to the note above).
        It must contains, inside each object ID key (numbers):
        - a 'centroid' key, containing the centroid coordinates of the object
        - a 'type" key, containing the class type of the object (classification)
    selectedclasses: list
        Array containing the different class from what the user wants the caclulation to be done.
    cellfilter: str, optional
        Indicate if we select all the cells for distance calculations or only a subset of them.
        If empty, will take all the cells.
        If 'Tumor', will use the maskmap to define the tumor regions and then calculate distances only
        inside these tumor regions
        If 'TumorMargin', will use the maskmap to define the tumor regions, then extend the region with a Margin
        defined in tumormargin
        and theb calculate distances only inside these extended tumor regions
    maskmap: str, optional
        Path to the binary image, mask of a specific region (here tumor) of the original image.
        The image must be in PIL supported format.
        If no maskmap selected AND no cellfilters, the distance calculation will be done for the whole WSI.
    maskmapdownfactor: int, optional
        Set what was the downsample factor when the maksmap (tumor regions) were generated
    tumormargin: int, optional
        Definition IN PIXEL of the margin around the tumor regions to take into consideration
        for the cell to cell distance calculation.
    Returns
    -------
    dist_nestedlist : list
        The dist_nestedlist contains average of the closest neighboor distances between all cells from one type and
        all cells from another
        It is generated as the following (example for 5 classes)
        [ [ [dist class 0 to class 1] [dist class 0 to class 2] [dist class 0 to class 3] [dist class 0 to class 4] ],
          [ [dist class 1 to class 2] [dist class 1 to class 3] [dist class 1 to class 4] ] ,
          [ [dist class 2 to class 3] [dist class 2 to class 4] ],  [ [dist class 3 to class 4] ]]
    """
    with open(classjson, 'r') as filename:
        classjson = json.load(filename)  # data must be a dictionnary
    if cellfilter == 'Tumor':
        maskmap = Image.open(maskmap)
        maskmap = np.array(maskmap)  # The maskmap size is not the same as the input image, it is downsampled
        # Check shape of the input file
        if len(maskmap.shape) != 2 and len(maskmap.shape) != 3:
            raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                             'Image type not supported ')
        elif len(maskmap.shape) == 3:
            if maskmap.shape[2] == 3:
                maskmap = maskmap[:, :, 0]  # Keep only one channel of the image if the image is 3 channels (RGB)
            else:
                raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                                 'Image type not supported ')
        # Create a connected component maps to have different values for the different tumor regions
        # (not connected ones)
        num_labels, tumorid_map = cv2.connectedComponents(maskmap, connectivity=8)
        regions = regionprops(tumorid_map)
    if cellfilter == 'TumorMargin':
        maskmap = Image.open(maskmap)
        maskmap = np.array(maskmap)  # The maskmap size is not the same as the input image, it is downsampled
        # Keep only one channel of the image if the image is 3 channels (RGB)
        if len(maskmap.shape) != 2 and len(maskmap.shape) != 3:
            raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                             'Image type not supported ')
        elif len(maskmap.shape) == 3:
            if maskmap.shape[2] == 3:
                maskmap = maskmap[:, :, 0]  # Keep only one channel of the image if the image is 3 channels (RGB)
            else:
                raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                                 'Image type not supported ')
        # TumorMargin should be in pixel as input directly
        kernel_size = int(tumormargin / 2)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        maskmap = cv2.dilate(maskmap, kernel)
        # Create a connected component maps to have different values for the different tumor regions
        # (not connected ones)
        num_labels, tumorid_map = cv2.connectedComponents(maskmap, connectivity=8)
        regions = regionprops(tumorid_map)

    # Extract centroids + Class information for each nucleus in the dictionnary
    allnucl_info = [
        [int(classjson[nucleus]['centroid'][0]),
         int(classjson[nucleus]['centroid'][1]),
         classjson[nucleus]['type']]
        for nucleus in classjson.keys()]  # should extract only first level keys

    dist_nestedlist = list()
    for sourceclass in selectedclasses:
        sourceclass_allavgdist = list()
        for targetclass in selectedclasses:

            if targetclass <= sourceclass:
                # In fact avoid to calculate distance calculations already done or between the same class
                continue

            else:
                print("Currently processing distance calculation "
                      "between cells of Source Class {} and cells of Target Class {}".format(sourceclass, targetclass))
                # In the case we take all the cells inside the tumor region only
                # or we take all the cells inside the tumor region + a margin
                # Keep in mind that the maskmap (tumormap) is a downsampled version of the WSI
                if cellfilter == 'Tumor' or cellfilter == 'TumorMargin':
                    if cellfilter == 'Tumor':
                        print('Keeping the cells inside Tumor regions only.')
                    if cellfilter == 'TumorMargin':
                        print('Keeping the cells inside Tumor regions '
                              '(including a Margin of {} selected by the user only.'.format(tumormargin))
                    # keep only the nucleus of source class inside the tumor region:
                    sourceclass_list = [nucl_info for nucl_info in allnucl_info
                                        if nucl_info[2] == sourceclass and
                                        maskmap[int(nucl_info[1] / maskmapdownfactor),
                                                int(nucl_info[0] / maskmapdownfactor)] == 255]
                    # keep only the nucleus of target class inside the tumor region:
                    targetclass_list = [nucl_info for nucl_info in allnucl_info
                                        if nucl_info[2] == targetclass and
                                        maskmap[int(nucl_info[1] / maskmapdownfactor),
                                                int(nucl_info[0] / maskmapdownfactor)] == 255]
                # In the case we take all the cells
                else:
                    # keep only the nucleus of source class:
                    sourceclass_list = [nucl_info for nucl_info in allnucl_info if nucl_info[2] == sourceclass]
                    # keep only the nucleus of target class:
                    targetclass_list = [nucl_info for nucl_info in allnucl_info if nucl_info[2] == targetclass]

                # maybe create a 2 different reseacrch areas size and if there is no cells in the first area,
                # go to the second,
                # if there is no cells in the second area,
                # go to everything
                # create a chain of this the areea size will be linked to the tumor region area
                # (maybe taking the extreme points)

                # Pick a  nucleus of source class, calculate distance with it and all the other class,
                # keep the lowest and delete the list
                allmindist = list()
                min_dist = float('inf')
                for source_info in tqdm(sourceclass_list):
                    alldist = list()
                    all_trgpoints = list()
                    if cellfilter == 'Tumor' or cellfilter == 'TumorMargin':
                        for target_info in targetclass_list:
                            source_tumor_id = tumorid_map[int(source_info[1] / maskmapdownfactor),
                                                         int(source_info[0] / maskmapdownfactor)]
                            target_tumor_id = tumorid_map[int(target_info[1] / maskmapdownfactor),
                                                         int(target_info[0] / maskmapdownfactor)]
                            if source_tumor_id == target_tumor_id:
                                # we calculate the distance only if we are in the same tumor region
                                all_trgpoints.append([int(target_info[0]), int(target_info[1])])
                                # /!\ x and y are kept inverted (as in the json)
                    else:
                        all_trgpoints = [[int(target_info[0]),
                                          int(target_info[1])] for target_info in targetclass_list]
                    selectedtrg_points = list()
                    multfactor = 1
                    # define the bounding box of the tumor region, and the length and wide of the bbox
                    bboxcoord = [r.bbox for r in regions if r.label == source_tumor_id]
                    # Bounding box (min_row, min_col, max_row, max_col).
                    bbmin_row, bbmax_row, bbmin_col, bbmax_col = bboxcoord[0][0], \
                                                                 bboxcoord[0][2], \
                                                                 bboxcoord[0][1], \
                                                                 bboxcoord[0][3]
                    bboxlength = bbmax_col - bbmin_col
                    # bboxcoord mmust be a LIST of ONE TUPLE, this explain the double brackets.
                    bboxwide = bbmax_row - bbmin_row
                    # bboxcoord mmust be a LIST of ONE TUPLE, this explain the double brackets.
                    # BE CAREFUL ALL THESE LENGHT ARE IN THE DOWNSIZE MAP
                    # Find all the points belonging to the subset of the bounding box around the source class cell
                    # We continue to expand the subset size if we don't find any cell
                    # or until the subset is the bounding box itself
                    while len(selectedtrg_points) == 0 and multfactor < 20.5:
                        pourcentage = 0.05 * multfactor
                        xminthr = source_info[0] - bboxlength * pourcentage * maskmapdownfactor
                        xmaxthr = source_info[0] + bboxlength * pourcentage * maskmapdownfactor
                        yminthr = source_info[1] - bboxwide * pourcentage * maskmapdownfactor
                        ymaxthr = source_info[1] + bboxwide * pourcentage * maskmapdownfactor
                        selectedtrg_points = [trgpoint for trgpoint in all_trgpoints if
                                              max(xminthr, bbmin_col * maskmapdownfactor)
                                              <= trgpoint[0] <= min(xmaxthr, bbmax_col * maskmapdownfactor) and
                                              max(yminthr, bbmin_row)
                                              <= trgpoint[1] <= min(ymaxthr, bbmax_row * maskmapdownfactor)]

                        multfactor += 1
                    # We calculate all the distances for the points in the subset
                    # print('len of selectedtrg_points', len(selectedtrg_points))
                    for selectedtrg_point in selectedtrg_points:
                        dist = math.sqrt((int(source_info[0]) - selectedtrg_point[0]) ** 2 +
                                         (int(source_info[1]) - selectedtrg_point[1]) ** 2)  # distance calculation
                        # In alldist there is all the distances between cell of source class and
                        # all cells of target class
                        alldist.append(dist)
                    # We keep only the min distance
                    if alldist:
                        # We need to check if the list is not empty to continue
                        # (because it is possible that no target cells are in the same tumor region)
                        min_dist = min(alldist)
                    del alldist
                    # In allmindist there is all the minimum distances between each cell of source class and
                    # all cells of target class
                    if not min_dist:  # in case there is no neighbour beetwen the 2 classes
                        dist_nestedlist.append({})
                    if min_dist:  # redundant but better readibility
                        allmindist.append(min_dist)

                del sourceclass_list
                del targetclass_list
                if min_dist:
                    avgdist = sum(allmindist) / len(allmindist)  # take the average of all closest neighbour distance
            # In sourceclass_allavgdist
            if min_dist:
                sourceclass_allavgdist.append(avgdist)

        # created nested list with the sourceclass_allavgdist except if sourceclass_allavgdist is empty
        # (in the last occurence it is empty)
        if sourceclass_allavgdist:
            dist_nestedlist.append(sourceclass_allavgdist)

    # The dist_nestedlist contains average of the closest neighboor distances between
    # all cells from one type and all cells from another
    # To see the structure of the nested list, check the doc_string

    return dist_nestedlist
    # Keep it as a nested list and then with hvn outproperties we will transform this nested list into a dict with
    # corresponding keys!


def mpcell2celldist_classjson(classjson: str, selectedclasses: list,
                              cellfilter: str = 'Tumor',
                              maskmap: str = '',
                              maskmapdownfactor: int = 1,
                              tumormargin: int = None) -> list:
    """
    Use multiprocessing to calculate average of the closest neighboor distances between all cells from one type and
    all cells from another.
    Calculate this average for all cell types pairs.

    Parameters
    ----------
    classjson: str
        Path to the json file (for instance output of hovernet then formatted according to the note above).
        It must contains, inside each object ID key (numbers):
        - a 'centroid' key, containing the centroid coordinates of the object
        - a 'type" key, containing the class type of the object (classification)
    selectedclasses: list
        List containing the different class from what the user wants the caclulation to be done.
    cellfilter: str, optional
        Indicate if we select all the cells for distance calculations or only a subset of them.
        If empty, will take all the cells.
        If 'Tumor', will use the maskmap to define the tumor regions and then calculate distances only
        inside these tumor regions
        If 'TumorMargin', will use the maskmap to define the tumor regions, then extend the region with a Margin
        defined in tumormargin
        and theb calculate distances only inside these extended tumor regions
    maskmap: str, optional
        Path to the binary image, mask of a specific region (here tumor) of the original image.
        The image must be in PIL supported format.
        If no maskmap selected AND no cellfilters, the distance calculation will be done for the whole WSI.
    maskmapdownfactor: int, optional
        Set what was the downsample factor when the maksmap (tumor regions) were generated
    tumormargin: int, optional
        Definition IN PIXEL of the margin around the tumor regions to take into consideration
        for the cell to cell distance calculation.
    Returns
    -------
    dist_nestedlist : list
        The dist_nestedlist contains average of the closest neighboor distances between all cells from one type and
        all cells from another
        It is generated as the following (example for 5 classes)
        [ [ [dist class 0 to class 1] [dist class 0 to class 2] [dist class 0 to class 3] [dist class 0 to class 4] ],
          [ [dist class 1 to class 2] [dist class 1 to class 3] [dist class 1 to class 4] ] ,
          [ [dist class 2 to class 3] [dist class 2 to class 4] ],  [ [dist class 3 to class 4] ]]
    """
    with open(classjson, 'r') as filename:
        classjson = json.load(filename)  # data must be a dictionnary
    if cellfilter == 'Tumor':
        maskmap = Image.open(maskmap)
        maskmap = np.array(maskmap)  # The maskmap size is not the same as the input image, it is downsampled
        # Check shape of the input file
        if len(maskmap.shape) != 2 and len(maskmap.shape) != 3:
            raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                             'Image type not supported ')
        elif len(maskmap.shape) == 3:
            if maskmap.shape[2] == 3:
                maskmap = maskmap[:, :, 0]  # Keep only one channel of the image if the image is 3 channels (RGB)
            else:
                raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                                 'Image type not supported ')
        # Create a connected component maps to have different values for the different tumor regions
        # (not connected ones)
        num_labels, tumorid_map = cv2.connectedComponents(maskmap, connectivity=8)
        regions = regionprops(tumorid_map)
    if cellfilter == 'TumorMargin':
        maskmap = Image.open(maskmap)
        maskmap = np.array(maskmap)  # The maskmap size is not the same as the input image, it is downsampled
        # Keep only one channel of the image if the image is 3 channels (RGB)
        if len(maskmap.shape) != 2 and len(maskmap.shape) != 3:
            raise ValueError('The input image (maskmap) is not an image of 1 or 3 channels. '
                             'Image type not supported ')
        elif len(maskmap.shape) == 3:
            if maskmap.shape[2] == 3:
                maskmap = maskmap[:, :, 0]  # Keep only one channel of the image if the image is 3 channels (RGB)
            else:
                raise ValueError(
                    'The input image (maskmap) is not an image of 1 or 3 channels. '
                    'Image type not supported ')
        # TumorMargin should be in pixel as input directly
        kernel_size = int(tumormargin / 2)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        maskmap = cv2.dilate(maskmap, kernel)
        # Create a connected component maps to have different values for the different tumor regions
        # (not connected ones)
        num_labels, tumorid_map = cv2.connectedComponents(maskmap, connectivity=8)
        regions = regionprops(tumorid_map)

    # Extract centroids + Class information for each nucleus in the dictionnary
    allnucl_info = [
        [int(classjson[nucleus]['centroid'][0]),
         int(classjson[nucleus]['centroid'][1]),
         classjson[nucleus]['type']]
        for nucleus in classjson.keys()]  # should extract only first level keys

    dist_nestedlist = list()
    queuenames_list = list()
    print("All the distance calculations will run in parallel...")
    print("One progress bar per _COMPLETED_ process (one process = calculation of distances between 2 cell classes)")
    print("Number of needed CPU is (number-of-selectedclass * number-of-selectedclass - 1) / 2 ")
    print("In this run, the number of needed CPU is {}".format(
        int((len(selectedclasses) * (len(selectedclasses) - 1)) / 2)))
    for sourceclass in selectedclasses:
        sourceclass_allavgdist = list()
        for targetclass in selectedclasses:

            if targetclass <= sourceclass:
                # In fact avoid to calculate distance calculations already done or between the same class
                continue

            else:
                # We create different queue names to launch different processes in parallel
                queuenames_list.append(
                    ['queuedist_' + 'sourceclass' + str(sourceclass) + '_targetclass' + str(targetclass)]
                )
                queuenames_list[-1] = mp.Queue()
                p = mp.Process(target=multipro_distc2c,
                               args=(allnucl_info,
                                     sourceclass,
                                     targetclass,
                                     regions,
                                     maskmap,
                                     tumorid_map,
                                     cellfilter,
                                     maskmapdownfactor,
                                     queuenames_list[-1])
                                     # tumormargin,
                                     # dist_nestedlist,
                              )
                p.start()

    p.join()
    avgdist_list = [qnames.get() for qnames in queuenames_list]

    # Creation of a nested list from the list genrated to mimic the single process function output
    # (so to be consistent)
    dist_nestedlist = list()
    for k in range(1, len(selectedclasses)):
        dist_nestedlist.append(avgdist_list[0:len(selectedclasses) - k])
        del avgdist_list[0:len(selectedclasses) - k]

    print(dist_nestedlist)
    return dist_nestedlist


def multipro_distc2c(allnucl_info,
                     sourceclass,
                     targetclass,
                     regions,
                     maskmap,
                     tumorid_map,
                     cellfilter,
                     maskmapdownfactor,
                     queue):
    """
    Function to allow multiprocessing on the distance calculation.
    See mpcell2celldist_classjson function

    Parameters
    ----------

    Returns
    -------

    """
    # In the case we take all the cells inside the tumor region only
    # or we take all the cells inside the tumor region + a margin
    # Keep in mind that the maskmap (tumormap) is a downsampled version of the WSI
    if cellfilter == 'Tumor' or cellfilter == 'TumorMargin':
        # keep only the nucleus of source class inside the tumor region:
        sourceclass_list = [nucl_info for nucl_info in allnucl_info
                            if nucl_info[2] == sourceclass and
                            maskmap[int(nucl_info[1] / maskmapdownfactor),
                                    int(nucl_info[0] / maskmapdownfactor)] == 255]

        # keep only the nucleus of target class inside the tumor region:
        targetclass_list = [nucl_info for nucl_info in allnucl_info
                            if nucl_info[2] == targetclass and
                            maskmap[int(nucl_info[1] / maskmapdownfactor),
                                    int(nucl_info[0] / maskmapdownfactor)] == 255]
    # In the case we take all the cells
    else:
        sourceclass_list = [nucl_info for nucl_info in allnucl_info if
                            nucl_info[2] == sourceclass]  # keep only the nucleus of source class
        targetclass_list = [nucl_info for nucl_info in allnucl_info if
                            nucl_info[2] == targetclass]  # keep only the nucleus of target class

    # maybe create a 2 different reseacrch areas size and if there is no cells in the first area, go to the second,
    # if there is no cells in the second area  go to everything
    # create a chain of this the areea size will be linked to the tumor region area (maybe taking the extreme points)

    # Pick a  nucleus of source class, calculate distance with it and all the other class, keep
    # the lowest and delete the list
    allmindist = list()
    min_dist = []
    for source_info in tqdm(sourceclass_list):
        alldist = list()
        all_trgpoints = list()
        if cellfilter == 'Tumor' or cellfilter == 'TumorMargin':
            for target_info in targetclass_list:
                source_tumor_id = tumorid_map[int(source_info[1] / maskmapdownfactor),
                                              int(source_info[0] / maskmapdownfactor)]
                target_tumor_id = tumorid_map[int(target_info[1] / maskmapdownfactor),
                                              int(target_info[0] / maskmapdownfactor)]
                if source_tumor_id == target_tumor_id:
                    # we calculate the distance only if we are in the same tumor region
                    all_trgpoints.append(
                        [int(target_info[0]), int(target_info[1])])  # /!\ x and y are kept inverted (as in the json)
        else:
            all_trgpoints = [[int(target_info[0]), int(target_info[1])] for target_info in targetclass_list]
        selectedtrg_points = list()
        multfactor = 1
        # define the bounding box of the tumor region, and the length and wide of the bbox
        bboxcoord = [r.bbox for r in regions if r.label == source_tumor_id]
        # Bounding box (min_row, min_col, max_row, max_col).
        bbmin_row, bbmax_row, bbmin_col, bbmax_col = bboxcoord[0][0], bboxcoord[0][2], bboxcoord[0][1], bboxcoord[0][3]
        bboxlength = bbmax_col - bbmin_col
        # bboxcoord mmust be a LIST of ONE TUPLE, this explain the double brackets.
        bboxwide = bbmax_row - bbmin_row
        # bboxcoord mmust be a LIST of ONE TUPLE, this explain the double brackets.
        # BE CAREFUL ALL THESE LENGHT ARE IN THE DOWNSIZE MAP
        # Find all the points belonging to the subset of the bounding box around the source class cell
        # We continue to expand the subset size if we don't find any cell or until the subset is the bounding box itself
        while len(selectedtrg_points) == 0 and multfactor < 20.5:
            pourcentage = 0.05 * multfactor
            xminthr = source_info[0] - bboxlength * pourcentage * maskmapdownfactor
            xmaxthr = source_info[0] + bboxlength * pourcentage * maskmapdownfactor
            yminthr = source_info[1] - bboxwide * pourcentage * maskmapdownfactor
            ymaxthr = source_info[1] + bboxwide * pourcentage * maskmapdownfactor
            selectedtrg_points = [trgpoint for trgpoint in all_trgpoints if
                                  max(xminthr, bbmin_col * maskmapdownfactor)
                                  <= trgpoint[0] <= min(xmaxthr, bbmax_col * maskmapdownfactor) and
                                  max(yminthr, bbmin_row)
                                  <= trgpoint[1] <= min(ymaxthr, bbmax_row * maskmapdownfactor)]

            multfactor += 1
        # We calculate all the distances for the points in the subset
        # print('len of selectedtrg_points', len(selectedtrg_points))
        for selectedtrg_point in selectedtrg_points:
            dist = math.sqrt((int(source_info[0]) - selectedtrg_point[0]) ** 2 + (
                    int(source_info[1]) - selectedtrg_point[1]) ** 2)  # distance calculation
            # In alldist there is all the distances between cell of source class and all cells of target class
            alldist.append(dist)
        # We keep only the min distance
        if alldist:  # We need to check if the list is not empty to continue
            # (because it is possible that no target cells are in the same tumor region)
            min_dist = min(alldist)
        del alldist
        # In allmindist there is all the minimum distances between each cell of source class and
        # all cells of target class

        # if not min_dist:  # in case there is no neighbour beetwen the 2 classes
        #     dist_nestedlist.append({})

        if min_dist:  # redundant but better readibility
            allmindist.append(min_dist)

    del sourceclass_list
    del targetclass_list
    if min_dist:
        avgdist = sum(allmindist) / len(allmindist)  # take the average of all closest neighbour distance
    else:
        avgdist = False

    # In sourceclass_allavgdist

    queue.put(avgdist)

    # outlist = [avgdist, min_dist]
    # list(map(queue.put, outlist))


def hvn_outputproperties(allcells_in_wsi_dict: dict = None,
                         cells_inmask_dict: dict = None,
                         cellsdist_inmask_dict: dict = None,
                         masktype: str = 'Tumor',
                         areaofmask: int = None, 
                         selectedcls_ratio: list = None,
                         selectedcls_dist: list = None) -> dict:
    """
    Calculate and store in a dictionnary all tissue features.

    Works with already defined class, needs change if someone wants to use
    different classes/

    Parameters:
    ----------
    allcells_inWSI_dict: dict, optional
        Dictiannary containing count of cells per cell type in the whole slide image.
    cells_inmask_dict: dict, optional
        Dictionnary containing number of instances from each class contained in "selectedclasses",
        that are inside the mask from maskmap.
        It is the output of cellsratio_insidemask_classjson function
    cellsdist_inmask_dict: list, optional
        List containing average of the closest neighboor distances between all cells from one type and
        all cells from another.
        It is the output of mpcell2celldist_classjson and cell2celldist_classjson functions
    masknature: str, optional
        Define the type of the mask from mask map. Here it is usually Tumor.
    areaofmask: int, optional
        Area in pixel of the mask (tumor region) in the maskmap.
    selectedcls_ratio: list, optional
        List containing the different class from what the user wants the ratio caclulations (inside tumor regions) to be done
    selectedcls_dist: list, optional
        List containing the different class from what the user wants the distance caclulations to be done
    Returns:
    -------
    resultdict, dict
        Dictionnary continaing all the calculated tissue features that will be used for final classification.
    """
    # Initializing all dictionnaries as empty and paper metrics as None in case they are not calculating before
    # because of missing optionnal input for the function

    # Dictionnary of calculation for the whole WSI
    if allcells_in_wsi_dict is None:
        allcells_in_wsi_dict = {}
    # Dictionnary for the calculation of cell type ratios using mask information
    if cells_inmask_dict is None:
        cells_inmask_dict = {}
    # List of the distance between cell types
    if cellsdist_inmask_dict is None:
        cellsdist_inmask_dict = {}

    fractions_wsi_dict = dict()
    ratio_wsi_dict = dict()
    fractions_tumor_dict = dict()
    dist_tumor_dict = dict()
    density_tumor_dict = dict()
    ratio_tumor_dict = dict()
    insidevs_outside_dict = dict()
    ITLR = "Not calculated"
    SCD = "Not calculated"

    ### Calculations linked to WSI cells regardless Tumor Regions

    if allcells_in_wsi_dict:
        # Fraction of cell types (FractionsWSIDict)
        totalnumberofcells = (
                sum(allcells_in_wsi_dict.values()) - allcells_in_wsi_dict["Background"]
        )
        fractions_wsi_dict["Granulocytes_Pourcentage"] = (
                allcells_in_wsi_dict["Granulocyte"] / totalnumberofcells
        )
        fractions_wsi_dict["Lymphocytes_Pourcentage"] = (
                allcells_in_wsi_dict["Lymphocyte"] / totalnumberofcells
        )
        fractions_wsi_dict["PlasmaCells_Pourcentage"] = (
                allcells_in_wsi_dict["Plasma"] / totalnumberofcells
        )
        fractions_wsi_dict["StromaCells_Pourcentage"] = (
                allcells_in_wsi_dict["Stroma"] / totalnumberofcells
        )
        fractions_wsi_dict["TumorCells_Pourcentage"] = (
                allcells_in_wsi_dict["Tumor"] / totalnumberofcells
        )
        fractions_wsi_dict["EpithelialCells_Pourcentage"] = (
                allcells_in_wsi_dict["Epithelial"] / totalnumberofcells
        )


        # Cell Type ratios (RatioWSIDict)
        ratio_wsi_dict["Ratio_Granulocytes_TumorCells"] = (
                allcells_in_wsi_dict["Granulocyte"] / allcells_in_wsi_dict["Tumor"]
        )
        ratio_wsi_dict["Ratio_Lymphocytes_TumorCells"] = (
                allcells_in_wsi_dict["Lymphocyte"] / allcells_in_wsi_dict["Tumor"]
        )
        ratio_wsi_dict["Ratio_PlasmaCells_TumorCells"] = (
                allcells_in_wsi_dict["Plasma"] / allcells_in_wsi_dict["Tumor"]
        )
        ratio_wsi_dict["Ratio_StromaCells_TumorCells"] = (
                allcells_in_wsi_dict["Stroma"] / allcells_in_wsi_dict["Tumor"]
        )
        ratio_wsi_dict["Ratio_EpithelialCells_TumorCells"] = (
                allcells_in_wsi_dict["Epithelial"] / allcells_in_wsi_dict["Tumor"]
        )
        ratio_wsi_dict["Ratio_Granulocytes_Lymphocytes"] = (
                allcells_in_wsi_dict["Granulocyte"] / allcells_in_wsi_dict["Lymphocyte"]
        )
        ratio_wsi_dict["Ratio_PlasmaCells_Lymphocytes"] = (
                allcells_in_wsi_dict["Plasma"] / allcells_in_wsi_dict["Lymphocyte"]
        )
        ratio_wsi_dict["Ratio_StromaCells_Lymphocytes"] = (
                allcells_in_wsi_dict["Stroma"] / allcells_in_wsi_dict["Lymphocyte"]
        )
        ratio_wsi_dict["Ratio_EpithelialCells_Lymphocytes"] = (
                allcells_in_wsi_dict["Epithelial"] / allcells_in_wsi_dict["Lymphocyte"]
        )
        ratio_wsi_dict["Ratio_Granulocytes_PlasmaCells"] = (
                allcells_in_wsi_dict["Granulocyte"] / allcells_in_wsi_dict["Plasma"]
        )
        ratio_wsi_dict["Ratio_StromaCells_PlasmaCells"] = (
                allcells_in_wsi_dict["Stroma"] / allcells_in_wsi_dict["Plasma"]
        )
        ratio_wsi_dict["Ratio_EpithelialCells_PlasmaCells"] = (
                allcells_in_wsi_dict["Epithelial"] / allcells_in_wsi_dict["Plasma"]
        )
        ratio_wsi_dict["Ratio_StromaCells_Granulocytes"] = (
                allcells_in_wsi_dict["Stroma"] / allcells_in_wsi_dict["Granulocyte"]
        )
        ratio_wsi_dict["Ratio_EpithelialCells_Granulocytes"] = (
                allcells_in_wsi_dict["Epithelial"] / allcells_in_wsi_dict["Granulocyte"]
        )
        ratio_wsi_dict["Ratio_EpithelialCells_StromalCells"] = (
                allcells_in_wsi_dict["Epithelial"] / allcells_in_wsi_dict["Stroma"]
        )


    # Create dictionnary for the whole section of calculations linked to WSI cells
    calculations_wsi_dict = {
        "Pourcentages_of_cell_types_in_WSI": fractions_wsi_dict,
        "Ratios_between_cell_types_WSI": ratio_wsi_dict,
    }

    # Number of cells per type inside tumor areas
    # Already calculated, it is an input of the function (cellsratio_inmask_dict)

    ### Calculations linked to ratio of cells inside tumor regions

    if cells_inmask_dict:
        # Fraction of cell types taking into account only cells inside tumor regions (FractionsTumorDict)
        if masktype == "Tumor":
            nummcellsdict = cells_inmask_dict.get(
                "dict_numinstanceperclass", {}
            )  # number of cells inside tumor regions
            numcells = sum(
                nummcellsdict.values()
            )  # No background cell class inside  instmaskdict
                    
            if selectedcls_ratio == [1, 2, 3, 4, 5]:    
                fractions_tumor_dict["Pourcentage_Granulocytes_allcellsinTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Granulocyte"]
                        / numcells
                )
                fractions_tumor_dict["Pourcentage_Lymphocytes_allcellsinTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Lymphocyte"]
                        / numcells
                )
                fractions_tumor_dict["Pourcentage_PlasmaCells_allcellsinTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Plasma"] / numcells
                )
                fractions_tumor_dict["Pourcentage_StromaCells_allcellsinTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Stroma"] / numcells
                )
                fractions_tumor_dict["Pourcentage_TumorCells_allcellsinTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Tumor"] / numcells
                        )
                # Cell Type ratios (RatioTumorDict)
                ratio_tumor_dict["Ratio_Granulocytes_TumorCells_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Granulocyte"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Tumor"]
                )
                ratio_tumor_dict["Ratio_Lymphocytes_TumorCells_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Lymphocyte"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Tumor"]
                )
                ratio_tumor_dict["Ratio_PlasmaCells_TumorCells_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Plasma"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Tumor"]
                )
                ratio_tumor_dict["Ratio_StromaCells_TumorCells_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Stroma"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Tumor"]
                )
                ratio_tumor_dict["Ratio_Granulocytes_Lymphocytes_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Granulocyte"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Lymphocyte"]
                )
                ratio_tumor_dict["Ratio_PlasmaCells_Lymphocytes_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Plasma"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Lymphocyte"]
                )
                ratio_tumor_dict["Ratio_StromaCells_Lymphocytes_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Stroma"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Lymphocyte"]
                )
                ratio_tumor_dict["Ratio_Granulocytes_PlasmaCells_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Granulocyte"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Plasma"]
                )
                ratio_tumor_dict["Ratio_StromaCells_PlasmaCells_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Stroma"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Plasma"]
                )
                ratio_tumor_dict["Ratio_StromaCells_Granulocytes_inTumor"] = (
                        cells_inmask_dict["dict_numinstanceperclass"]["Stroma"]
                        / cells_inmask_dict["dict_numinstanceperclass"]["Granulocyte"]
                )
                if areaofmask:
                    # Number of cells per tumor area
                    density_tumor_dict["Granulocytes_perTumorarea"] = (
                            cells_inmask_dict["dict_numinstanceperclass"]["Granulocyte"]
                            / areaofmask
                    )
                    density_tumor_dict["Lymphocytes_perTumorarea"] = (
                            cells_inmask_dict["dict_numinstanceperclass"]["Lymphocyte"]
                            / areaofmask
                    )
                    density_tumor_dict["PlasmaCells_perTumorarea"] = (
                            cells_inmask_dict["dict_numinstanceperclass"]["Plasma"]
                            / areaofmask
                    )
                    density_tumor_dict["StromaCells_perTumorarea"] = (
                            cells_inmask_dict["dict_numinstanceperclass"]["Stroma"]
                            / areaofmask
                    )
                    density_tumor_dict["TumorCells_perTumorarea"] = (
                            cells_inmask_dict["dict_numinstanceperclass"]["Tumor"]
                            / areaofmask
                    )
                    # Density of cells per tumor area
                    density_tumor_dict["GranulocytesDensity_insideTumorarea"] = (
                            cells_inmask_dict["dict_totareainstanceperclass"][
                                "Granulocyte"
                            ]
                            / areaofmask
                    )
                    density_tumor_dict["LymphocytesDensity_insideTumorarea"] = (
                            cells_inmask_dict["dict_totareainstanceperclass"]["Lymphocyte"]
                            / areaofmask
                    )
                    density_tumor_dict["PlasmaCellsDensity_insideTumorarea"] = (
                            cells_inmask_dict["dict_totareainstanceperclass"]["Plasma"]
                            / areaofmask
                    )
                    density_tumor_dict["StromaCellsDensity_insideTumorarea"] = (
                            cells_inmask_dict["dict_totareainstanceperclass"]["Stroma"]
                            / areaofmask
                    )
                    density_tumor_dict["TumorCellsDensity_insideTumorarea"] = (
                            cells_inmask_dict["dict_totareainstanceperclass"]["Tumor"]
                            / areaofmask
                    )
            else:
                raise ValueError('hvn_outputproperties cannot run with selectedcls_ratio as {}.'
                    'This is a custom class selection for ratio calculations iniside tumors.'
                    'hvn_outputproperties function needs to be updated to fit this selection.'
                    .format(selectedcls_ratio)) 


    # Create dictionnary for the whole section of calculations linked to cells inside tumor regions ratios
    calculations_ratio_tumor_dict = {
        "Pourcentages_of_cell_types_in_Tumor_Regions": fractions_tumor_dict,
        "Density_of_cell_types_inside_Tumor_Regions": density_tumor_dict,
        "Ratios_between_cell_types_Tumor_Regions": ratio_tumor_dict,
    }

    ### Calculations linked to distance between cells inside< tumor regions

    if cellsdist_inmask_dict:
        # Average Distance to closest neighboor
        # What to look for in cellsdist_in_mask is not obvious at all !!! Look doc string of cell2celldist_classjson
        if selectedcls_dist == [1, 2, 3, 4, 5]:
            dist_tumor_dict["DistClosest_Granulocytes_Lymphocytes_inTumor"] = cellsdist_inmask_dict[0][0]    
            dist_tumor_dict["DistClosest_Granulocytes_PlasmaCells_inTumor"] = cellsdist_inmask_dict[0][1]
            dist_tumor_dict["DistClosest_Granulocytes_StromaCells_inTumor"] = cellsdist_inmask_dict[0][2]
            dist_tumor_dict["DistClosest_Granulocytes_TumorCells_inTumor"] = cellsdist_inmask_dict[0][3]
            dist_tumor_dict["DistClosest_Lymphocytes_PlasmaCells_inTumor"] = cellsdist_inmask_dict[1][0]
            dist_tumor_dict["DistClosest_Lymphocytes_StromaCells_inTumor"] = cellsdist_inmask_dict[1][1]
            dist_tumor_dict["DistClosest_Lymphocytes_TumorCells_inTumor"] = cellsdist_inmask_dict[1][2]
            dist_tumor_dict["DistClosest_StromaCells_PlasmaCells_inTumor"] = cellsdist_inmask_dict[2][0]
            dist_tumor_dict["DistClosest_PlasmaCells_TumorCells_inTumor"] = cellsdist_inmask_dict[2][1]
            dist_tumor_dict["DistClosest_StromaCells_TumorCells_inTumor"] = cellsdist_inmask_dict[3][0]
        
        elif selectedcls_dist == [1, 2, 3, 5]:
            dist_tumor_dict["DistClosest_Granulocytes_Lymphocytes_inTumor"] = cellsdist_inmask_dict[0][0]    
            dist_tumor_dict["DistClosest_Granulocytes_PlasmaCells_inTumor"] = cellsdist_inmask_dict[0][1]
            dist_tumor_dict["DistClosest_Granulocytes_TumorCells_inTumor"] = cellsdist_inmask_dict[0][2]
            dist_tumor_dict["DistClosest_Lymphocytes_PlasmaCells_inTumor"] = cellsdist_inmask_dict[1][0]
            dist_tumor_dict["DistClosest_Lymphocytes_TumorCells_inTumor"] = cellsdist_inmask_dict[1][1]
            dist_tumor_dict["DistClosest_PlasmaCells_TumorCells_inTumor"] = cellsdist_inmask_dict[2][0]

        else:
            raise ValueError('hvn_outputproperties cannot run with selectedcls_dist as {}.'
                'This is a custom class selection for distance calculations.'
                'hvn_outputproperties function needs to be updated to fit this selection.'
                .format(selectedcls_dist)) 


    # Create dictionnary for the whole section of calculations linked to cells inside tumor regions distances
    calculations_dist_tumor_dict = {"Distances_of_cells_in_Tumor_Regions": dist_tumor_dict}

    ### Calculations linked to cells inside and outside tumor regions

    if allcells_in_wsi_dict and cells_inmask_dict:
        # Fraction of the cells outside and inside tumor regions per type (InsidevsOutsideDict)
        if masktype == "Tumor":
            insidevs_outside_dict["Pourcentage_Granulocytes_insideTumor"] = (
                    cells_inmask_dict["dict_numinstanceperclass"]["Granulocyte"]
                    / allcells_in_wsi_dict["Granulocyte"]
            )
            insidevs_outside_dict["Pourcentage_Lymphocytes_insideTumor"] = (
                    cells_inmask_dict["dict_numinstanceperclass"]["Lymphocyte"]
                    / allcells_in_wsi_dict["Lymphocyte"]
            )
            insidevs_outside_dict["Pourcentage_PlasmaCells_insideTumor"] = (
                    cells_inmask_dict["dict_numinstanceperclass"]["Plasma"]
                    / allcells_in_wsi_dict["Plasma"]
            )
            insidevs_outside_dict["Pourcentage_StromaCells_insideTumor"] = (
                    cells_inmask_dict["dict_numinstanceperclass"]["Stroma"]
                    / allcells_in_wsi_dict["Stroma"]
            )
            insidevs_outside_dict["Pourcentage_TumorCells_insideTumor"] = (
                    cells_inmask_dict["dict_numinstanceperclass"]["Tumor"]
                    / allcells_in_wsi_dict["Tumor"]
            )

    # Create dictionnary for the whole section of calculations linked to cells inside and outside tumor regions
    calculations_mixed_dict = {
        "Pourcentage_of_Cells_inside_Tumor_regions_for_a_given_cell_type": insidevs_outside_dict
    }

    ### Calculating metrics from papers NO NEED FOR NOW - TO REMOVE

    if cells_inmask_dict:
        # ITLR (intra-tumor lymphocyte ratio) = N intratumor lymphocyte / N cancer cell (inside tumor)
        print(
            "For ITLR calculation, the mask used for the input cells_inmask_dict must be a mask of tumor areas"
        )
        if masktype == "Tumor":
            num_itlymphocytes = cells_inmask_dict.get(
                "dict_numinstanceperclass", {}
            ).get(
                "Lymphocyte"
            )  # safe way of dealing with nested dictionnaries
            num_ittumor = cells_inmask_dict.get(
                "dict_numinstanceperclass", {}
            ).get("Tumor")
            ITLR = num_itlymphocytes / num_ittumor

        # SCD = total number of nucleated cells area/ stromal area
        print(
            "For SCD calculation, the mask used for the input cells_inmask_dict must be a mask of stromal areas"
        )
        if not masktype == "Tumor" and areaofmask:
            areanuclcellsdict = cells_inmask_dict.get(
                "dict_totareainstanceperclass", {}
            )
            areanuclcells = sum(areanuclcellsdict.values())
            SCD = areanuclcells / areaofmask

    # Create dictionnary for the whole section of calculating metrics from papers
    calculations_paper_metrics_dict = {"ITLR": ITLR, "SCD": SCD}

    # Creation of Nested dictionnary including all the informations needed
    resultdict = {
        "CalculationsforWSI": calculations_wsi_dict,
        "CalculationsRatiosinsideTumor": calculations_ratio_tumor_dict,
        "CalculationsDistinsideTumor": calculations_dist_tumor_dict,
        "CalculationsMixed": calculations_mixed_dict,
    }
    # Remark: we no longer keep the instance numbers in the dict :
    # {'InstancesNumberWSI': allcells_inWSI_dict, 'InstancesNumberMaskRegion': cells_inmask_dict}
    return resultdict





########
# UNDER CONSTRUCTIONS FUNCTIONS
########



def multipro_distc2c_test(allnucl_info,
                     sourceclass,
                     targetclass,
                     regions,
                     maskmap,
                     tumorid_map,
                     cellfilter,
                     maskmapdownfactor,
                     queue):
    """
    Function to allow multiprocessing on the distance calculation.
    See mpcell2celldist_classjson function

    Parameters
    ----------

    Returns
    -------

    """
    # In the case we take all the cells inside the tumor region only
    # or we take all the cells inside the tumor region + a margin
    # Keep in mind that the maskmap (tumormap) is a downsampled version of the WSI
    if cellfilter == 'Tumor' or cellfilter == 'TumorMargin':
        # keep only the nucleus of source class inside the tumor region:
        sourceclass_list = [nucl_info for nucl_info in allnucl_info
                            if nucl_info[2] == sourceclass and
                            maskmap[int(nucl_info[1] / maskmapdownfactor),
                                    int(nucl_info[0] / maskmapdownfactor)] == 255]

        # keep only the nucleus of target class inside the tumor region:
        targetclass_list = [nucl_info for nucl_info in allnucl_info
                            if nucl_info[2] == targetclass and
                            maskmap[int(nucl_info[1] / maskmapdownfactor),
                                    int(nucl_info[0] / maskmapdownfactor)] == 255]
    # In the case we take all the cells
    else:
        sourceclass_list = [nucl_info for nucl_info in allnucl_info if
                            nucl_info[2] == sourceclass]  # keep only the nucleus of source class
        targetclass_list = [nucl_info for nucl_info in allnucl_info if
                            nucl_info[2] == targetclass]  # keep only the nucleus of target class

    # maybe create a 2 different reseacrch areas size and if there is no cells in the first area, go to the second,
    # if there is no cells in the second area  go to everything
    # create a chain of this the areea size will be linked to the tumor region area (maybe taking the extreme points)

    # Pick a  nucleus of source class, calculate distance with it and all the other class, keep
    # the lowest and delete the list
    allmindist = list()
    min_dist = []
    for source_info in tqdm(sourceclass_list):
        alldist = list()
        all_trgpoints = list()
        if cellfilter == 'Tumor' or cellfilter == 'TumorMargin':
            for target_info in targetclass_list:
                source_tumor_id = tumorid_map[int(source_info[1] / maskmapdownfactor),
                                              int(source_info[0] / maskmapdownfactor)]
                target_tumor_id = tumorid_map[int(target_info[1] / maskmapdownfactor),
                                              int(target_info[0] / maskmapdownfactor)]
                if source_tumor_id == target_tumor_id:
                    # we calculate the distance only if we are in the same tumor region
                    all_trgpoints.append(
                        [int(target_info[0]), int(target_info[1])])  # /!\ x and y are kept inverted (as in the json)
        else:
            all_trgpoints = [[int(target_info[0]), int(target_info[1])] for target_info in targetclass_list]
        selectedtrg_points = list()
        multfactor = 1
        # define the bounding box of the tumor region, and the length and wide of the bbox
        bboxcoord = [r.bbox for r in regions if r.label == source_tumor_id]
        # Bounding box (min_row, min_col, max_row, max_col).
        bbmin_row, bbmax_row, bbmin_col, bbmax_col = bboxcoord[0][0], bboxcoord[0][2], bboxcoord[0][1], bboxcoord[0][3]
        bboxlength = bbmax_col - bbmin_col
        # bboxcoord mmust be a LIST of ONE TUPLE, this explain the double brackets.
        bboxwide = bbmax_row - bbmin_row
        # bboxcoord mmust be a LIST of ONE TUPLE, this explain the double brackets.
        # BE CAREFUL ALL THESE LENGHT ARE IN THE DOWNSIZE MAP
        # Find all the points belonging to the subset of the bounding box around the source class cell
        # We continue to expand the subset size if we don't find any cell or until the subset is the bounding box itself
        ratioindex = 0
        sizeratios = [0.05, 0.1, 0.25, 0.5, 1, 'stop']
        while len(selectedtrg_points) == 0 and sizeratios[ratioindex] != 'stop':
            sizeratio = int(sizeratios[ratioindex])
            xminthr = source_info[0] - bboxlength * sizeratio * maskmapdownfactor
            xmaxthr = source_info[0] + bboxlength * sizeratio * maskmapdownfactor
            yminthr = source_info[1] - bboxwide * sizeratio * maskmapdownfactor
            ymaxthr = source_info[1] + bboxwide * sizeratio * maskmapdownfactor
            selectedtrg_points = [trgpoint for trgpoint in all_trgpoints if
                                  max(xminthr, bbmin_col * maskmapdownfactor) <= trgpoint[0]
                                  <= min(xmaxthr, bbmax_col * maskmapdownfactor) and
                                  max(yminthr, bbmin_row) <= trgpoint[1]
                                  <= min(ymaxthr, bbmax_row * maskmapdownfactor)]
            ratioindex += 1

        # We calculate all the distances for the points in the subset
        # print('len of selectedtrg_points', len(selectedtrg_points))
        for selectedtrg_point in selectedtrg_points:
            dist = math.sqrt((int(source_info[0]) - selectedtrg_point[0]) ** 2 + (
                    int(source_info[1]) - selectedtrg_point[1]) ** 2)  # distance calculation
            # In alldist there is all the distances between cell of source class and all cells of target class
            alldist.append(dist)
        # We keep only the min distance
        if alldist:  # We need to check if the list is not empty to continue
            # (because it is possible that no target cells are in the same tumor region)
            min_dist = min(alldist)
        del alldist
        # In allmindist there is all the minimum distances between each cell of source class and
        # all cells of target class

        # if not min_dist:  # in case there is no neighbour beetwen the 2 classes
        #     dist_nestedlist.append({})

        if min_dist:  # redundant but better readibility
            allmindist.append(min_dist)

    del sourceclass_list
    del targetclass_list
    if min_dist:
        avgdist = sum(allmindist) / len(allmindist)  # take the average of all closest neighbour distance
    else:
        avgdist = False

    # In sourceclass_allavgdist

    queue.put(avgdist)

    # outlist = [avgdist, min_dist]
    # list(map(queue.put, outlist))