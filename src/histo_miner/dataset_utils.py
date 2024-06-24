#Lucas Sancéré -

import glob
import itertools
import os

import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import json
from scipy.io import loadmat
from typing import Tuple



def count_cells(classmap_folder: str, 
                instancemap_folder: str) -> Tuple[int, list, np.ndarray]:
        
    """
    Count number of cells for each class, for each classmap in a folder. It also needs the corresponding
    instance map. It returns:
    - total number of cells for all files combined
    - total number of cells for each class for all files combined
    - matrix with each row corresponding to number of cells for each class for a given file.
    The first column of the row is the name of the file. 

    Parameters:
    -----------
    instancemap_folder: str
        Path to the .npy InstanceMap files. Each file should contain the instance segmentation map
        where each cell instance is labeled with a unique integer value.
    classmap_folder: str
        Path to the .npy ClassMap files. The names should be the same as the InstanceMap files.
    Returns:
    --------
    nbr_cells: int
        The total number of cells counted across all instance maps in the folder.
    total_count_vect: list
        A list containing the total count of cells for each class across all instance maps in the folder.
    cell_count_matrix: np.ndarray
        A matrix where each row corresponds to an instance map and each column corresponds to a class,
        with the entries representing the count of cells for that class in the respective instance map.
    """
    instance_maps_path = os.path.join(instancemap_folder, '*.npy')
    files_instance_maps = glob.glob(instance_maps_path)
    count_per_file = []
    for fname in tqdm(files_instance_maps):
        #Load the Groundtruth class image
        instancemappath = os.path.splitext(fname)[0] #remove extension from the path
        instancemapname = os.path.split(instancemappath)[1] #keep only name of the image
        class_map_path = classmap_folder + instancemapname  \
                         + os.path.splitext(fname)[1]  # add the extension at the end
        if not os.path.exists(class_map_path): #check if there is an associated classmap, if not continue the loop
            continue
        class_map = np.load(class_map_path)
        #Load all the regions from input Instance Map
        instance_map = np.load(fname)
        regions = regionprops(instance_map)

        # Create a vector that will collect all pixel values for the same location of the object but
        # in Class Groundtruth Image
        class_values = []
        for region_id, region in enumerate(regions):
            coordinates = region.centroid
            if class_map[int(coordinates[0]), int(coordinates[1])] != 0: #Don't keep backround values
                class_values.append(class_map[int(coordinates[0]), int(coordinates[1])])
        class_values = np.asarray(class_values)
 
        # Count the number of each number in a new vector and start the vector by the name of the image
        cell_count = [0 for _ in range(6)]
        cell_count[0] = instancemapname + '.png'
        unique, counts = np.unique(class_values, return_counts=True)
        string_unique = unique.astype(str)
        class_count_dict = dict(zip(string_unique, counts)) 

        for index in range(1,len(cell_count)):
            if str(index) in class_count_dict:
                cell_count[index] = class_count_dict[str(index)]
            else:
                cell_count[index] = 0

        count_per_file.append(cell_count)

    # After looping on all file, we create the matrix for each file but also a global count
    cell_count_matrix = np.row_stack(count_per_file)

    # this vectors exclude the name of the files, it is just the total per class
    total_count_vect = [np.sum(cell_count_matrix[:, column_index].astype(int)) 
                        for column_index in range(1,len(cell_count))]
    print('Number of cells for each class in the folder:', total_count_vect)
    nbr_cells = np.sum(total_count_vect)
    print('Total number of cells in the folder:', nbr_cells)

    return nbr_cells, total_count_vect, cell_count_matrix


def count_cells_csv(classmap_folder: str, 
                    instancemap_folder: str, 
                    pathtosave: str) -> None:

    """
    Save the matrix of cell counts made with count_cell function with a csv file 
    Following the CellVIT/Pannnuke README format. Aslo save a csv with cancer type for each image
    Here it will be filled with "Skin" 

    Parameters:
    -----------
    instancemap_folder: str
        Path to the .npy IcnstanceMap files. Each file should contain the instance segmentation map
        where each cell instance is labeled with a unique integer value.
    classmap_folder: str
        Path to the .npy ClassMap files. The names should be the same as the InstanceMap files.
    pathtosave: str
        Path to the folder where the CSV file containing the count of cells will be saved.
    Returns:
    --------
    """
    count_cell_output = count_cells(classmap_folder, instancemap_folder)
    cell_count_matrix = count_cell_output[2]
    #update format of the cell_count_matrix to make it usable
    cell_count_matrix = cell_count_matrix.astype(str)
    # Headers name for the cell count csv
    headers = ["Image", "Granulocyte", "Lymphocyte", "Plasma", "Stromal", "Tumor"]
    # Save data to CSV with headers
    csvname_cellcnt = pathtosave + 'cell_count.csv'
    np.savetxt(csvname_cellcnt, 
               cell_count_matrix, 
               delimiter=",", 
               header=",".join(headers), 
               comments='', 
               fmt='%s')
    # Headers name for the cancer type names csv
    headers2 = ["img","type"]
    # Save data to CSV with headers
    csvname_cancertype = pathtosave + 'types.csv' 
    column_skin_vector = np.full(((cell_count_matrix[:,0]).shape[0], 1), "Skin")
    #q column_skin_vector = column_skin_vector.astype(str)  
    cancertype_vect = np.vstack((cell_count_matrix[:,0] , column_skin_vector[:,0]))
    cancertype_vect = np.transpose(cancertype_vect)
    np.savetxt(csvname_cancertype, 
               cancertype_vect, 
               delimiter=",", 
               header=",".join(headers2), 
               comments='',
               fmt='%s')


def cellVIT_format(classmap_folder: str, 
                   instancemap_folder: str, 
                   pathtosave: str) -> None:
    """
    Create a file "label" that contains both class maps and instance maps as dictionnary keys inside np file/
    Format needed to train cellVIT on custom dataset. 

    Parameters:
    -----------
    instancemap_folder: str
        Path to the .npy IcnstanceMap files. Each file should contain the instance segmentation map
        where each cell instance is labeled with a unique integer value.
    classmap_folder: str
        Path to the .npy ClassMap files. The names should be the same as the InstanceMap files.
    pathtosave: str
        Path to the folder where the CSV file containing the count of cells will be saved.
    Returns:
    --------    
    """
    instance_maps_path = os.path.join(instancemap_folder, '*.npy')
    files_instance_maps = glob.glob(instance_maps_path)
    for fname in tqdm(files_instance_maps):
        #Load the Groundtruth class image
        instancemappath = os.path.splitext(fname)[0] #remove extension from the path
        instancemapname = os.path.split(instancemappath)[1] #keep only name of the image
        class_map_path = classmap_folder + instancemapname  \
                         + os.path.splitext(fname)[1]  # add the extension at the end
        if not os.path.exists(class_map_path): #check if there is an associated classmap, if not continue the loop
            continue
        type_map = np.load(class_map_path)
        type_map = type_map.astype(np.int32)
        #Load all the regions from input Instance Map
        inst_map = np.load(fname)
        inst_map = inst_map.astype(np.int32)

        outname = instancemapname + '.npy'
        outdict = {"inst_map": inst_map, "type_map": type_map}

        np.save(pathtosave + "/labels/" + outname, outdict)




# def main():

#     pathtofolder = '/data/lsancere/Data_General/Predictions/HE-ICH-external-validation-inference/281_18_HE/Classmaps/'
#     savefoldername = 'images/'
#     npystring = False
#     makergbimage = True
#     more_than_8bits = True
#     npydim = 3

#     fileslist = os.path.join(pathtofolder, '*.npy')
#     fileslist = glob.glob(fileslist)
#     os.makedirs(pathtofolder + savefoldername, exist_ok=True)

#     # for fname in tqdm(fileslist):
#     #     if os.path.exists(fname):
#     #         path, extension = os.path.splitext(fname)  # split the file name and extension
#     #         pathtofolder, filename = os.path.split(path)  # split the path and the file name
#     #         # create the name of the file to save in the same place as the original file:
#     #         savename = savefoldername + filename

#     #         # Load the saved numpy file
#     #         loaded_dict = np.load(fname, allow_pickle=True).item()

#     #         # Extract the type_map from the loaded dictionary
#     #         type_map = loaded_dict['type_map']

#     #         maxval = np.max(type_map)

#     #         if maxval > 5:
                
#     #             print("finame",filename)
#     #             print("max value", maxval)


#     if npystring:
#         for fname in tqdm(fileslist):
#             if os.path.exists(fname):
#                 path, extension = os.path.splitext(fname)  # split the file name and extension
#                 pathtofolder, filename = os.path.split(path)  # split the path and the file name
#                 # create the name of the file to save in the same place as the original file:
#                 savename = savefoldername + filename

#                 # Apply function only if the output file doesn't exist
#                 if not os.path.exists(pathtofolder + '/' + savename + '.png'):
#                     # already (be careful Extension variable cannot be used here)
#                     save2dnpy_2png(fname, savename)

#     else:
#         if npydim == 2:
#             for fname in tqdm(fileslist):
#                 if os.path.exists(fname):
#                     path, extension = os.path.splitext(fname)  # split the file name and extension
#                     pathtofolder, filename = os.path.split(path)  # split the path and the file name
#                     # create the name of the file to save in the same place as the original file:
#                     savename = savefoldername + filename

#                     # Apply function only if the output file doesn't exist
#                     if not os.path.exists(pathtofolder + '/' + savename + '.png'):
#                         # already (be careful Extension variable cannot be used here)
#                         save2dnpy_2png(fname, savename, more_than_8bits=more_than_8bits)

#         if npydim == 3:
#             if not makergbimage:
#                 for k in range(0, 3):
#                     # Index is related to the function, it is to extract the right dimensions from the 3D npy
#                     index = k
#                     for fname in tqdm(fileslist):
#                         path, extension = os.path.splitext(fname)  # split the file name and extension
#                         pathtofolder, filename = os.path.split(path)  # split the path and the file name
#                         # create the name of the file to save in the same place as the original file:
#                         savename = savefoldername + filename
#                         # Apply function only if the output file doesn't exist:
#                         if not os.path.exists(pathtofolder + '/' + savename + '.png'):
#                                 # already (be careful Extension variable cannot be used here)
#                                 save3dnpy_2png(fname, savename, index, make_rgbimage=makergbimage)

#             else:
#                 for fname in tqdm(fileslist):
#                     if os.path.exists(fname):
#                         path, extension = os.path.splitext(fname)  # split the file name and extension
#                         pathtofolder, filename = os.path.split(path)  # split the path and the file name
#                         # create the name of the file to save in the same place as the original file:
#                         savename = savefoldername + filename

#                         # Apply function only if the output file doesn't exist:
#                         if not os.path.exists(pathtofolder + '/' + savename + '.png'):
#                             # already (be careful Extension variable cannot be used here)
#                             save3dnpy_2png(fname, savename, None, make_rgbimage=makergbimage)


#         if npydim == 4:
#             print('Use of save4Dnpy_2png not written yet')

#     print('Done')

# if __name__ == "__main__":
#     main()



def main():

    instancemap_folder = '/data/lsancere/Data_General/Predictions/HE-ICH-external-validation-inference/4811_18_HE/instancemaps/'
    classmap_folder = '/data/lsancere/Data_General/Predictions/HE-ICH-external-validation-inference/4811_18_HE/ClassMaps/'
    pathtosave = '/data/lsancere/Data_General/Predictions/HE-ICH-external-validation-inference/4811_18_HE/cellcount/'

    count_cells_csv(classmap_folder, instancemap_folder, pathtosave)
    
    print('Done')


if __name__ == "__main__":
    main()









