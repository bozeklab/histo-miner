#Lucas Sancéré -

import json
import os
import shutil
import numpy as np

import glob
from tqdm import tqdm
from PIL import Image as im

### Utils Functions

# Check this function as it can just REMOVE files if not used properly
# Maybe add GUI with users to understand fully want is happening


def anaylser2featselect(folderpath: str, recnaming: list = ('no_recurrence','recurrence')):
    """
    Move all the output files from the tissue analyses to 2 folders, recurrence and no_recurrence
    to perform the feature selection    

    Dangerous to use as it can remove json files.
    Maybe update it to be clearer.

    Parameters
    ----------
    folderpath: str
        Path to the folder containing all the json output of the tissue analyses.
        The json files could be in subdirectories as well
    recnaming: list
        List of names of the 2 classes

    Returns
    -------
    """
    tissueanalyser_folder = folderpath + '/' + 'tissue_analyses_sorted'
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
                namewoext, extension = os.path.splitext(file)
                filepath = root + '/' + file
                if extension == '.json' and 'analysed' in namewoext:
                    if norecurrencestr in filepath:
                        #Move the json file to sort it with the other no recurrence WSIs and rename the file as well to make
                        # no_recurrence strings appear in the name
                        shutil.move(filepath, norec_analyse_folder + '/' + file.replace('analysed', 'no_recurrence_analysed'))
                    # Be careful if one of the two string is also included in the other one. So to avoid issue we do:
                    elif not norecurrencestr in filepath and recurrencestr in filepath: 
                        shutil.move(filepath, rec_analyse_folder + '/' + file.replace('analysed', 'recurrence_analysed'))
                    else: 
                        raise ValueError('Some features are not associated to a recurrence or norecurrence WSI classification.'
                                          'User need a way to discriminate between the two cases, using folder naming.'
                                          'For now, the strings choosen in recnaming argument are: {}'
                                          'User can change this argument and check recnaming docstring to have more information'
                                          .format(recnaming))


def savenpy_2txt(file: str, savename: str) -> None:
    """
    Save .npy containing text into text file

    Parameters:
    -----------
    file: str
        path to the .npy file
    savename: str
        name of the file to save. By default it will be saved in the same folder.
        To change it add a path prefix to the file name. For example: 'Output/filename'
    Returns:
    --------
    """
    npyarray = np.load(file, allow_pickle=True)
    pathtofolder, filename = os.path.split(file)
    np.savetxt(pathtofolder + '/' + savename + '.txt', npyarray, fmt='%s')


def save2dnpy_2png(file: str, savename: str, more_than_8bits: bool = False) -> None:
    """
    Save .npy containing 2D matrix into .png

    Parameters:
    -----------
     file: str
        path to the .npy file
    savename: str
        name of the file to save. By default it will be saved in the same folder.
        To change it add a path prefix to the file name. For example: 'Output/filename'
    more_than_8bits: bool, optional
        Precise if the image contains pixel coded with more than 8 bits or not
    Returns:
    --------
    """
    npyarray = np.load(file)
    npyarray = npyarray.astype(int)
    if more_than_8bits:
        npyarray = npyarray.astype('uint32')
    else:
        npyarray = npyarray.astype('uint8')
    data = im.fromarray(npyarray)
    pathtofolder, filename = os.path.split(file)
    data.save((pathtofolder + '/' + savename + '.png'))


def save3dnpy_2png(file: str,
                   savename: str,
                   indextoextract: int = 0,
                   make_rgbimage: bool = False,
                   more_than_8bits: bool = False) -> None:
    """
    Save .npy containing 3D matrices into .png

    Parameters:
    -----------
    file: str
        path to the .npy file
    savename: str
        name of the file to save. By default it will be saved in the same folder.
        To change it add a path prefix to the file name. For example: 'Output/filename'
    indextoextract: int, optional
        dimension to set to extract png from 3D npy
    make_rgbimage: bool, optional
        precise if the image contains RGB channel or not
    more_than_8bits: bool, optional
        Precise if the image contains pixel coded with more than 8 bits or not
    Returns:
    --------
    """
    if indextoextract is None:
        indextoextract = 0
    npyarray = np.load(file)
    npyarray = npyarray[:, :, :3]
    npyarray = npyarray.astype(int)
    if not make_rgbimage:
        if more_than_8bits:
            npyarray = npyarray.astype('uint32')
        else:
            npyarray = npyarray.astype('uint8')
        data = im.fromarray(npyarray[:, :, indextoextract])  # Choose what to extract accordingly
        pathtofolder, filename = os.path.split(file)
        data.save((pathtofolder + '/' + savename + '_' + str(indextoextract) + '.png'))
    else:
        npyarray = npyarray.astype('uint8')
        data = im.fromarray(npyarray) #We want all the dimension for a PNG RGB image
        pathtofolder, filename = os.path.split(file)
        data.save((pathtofolder + '/' + savename + '.png'))


def save4dnpy_2png(file: str,
                   savename: str,
                   indexestoextract=None,
                   more_than_8bits: bool = False) -> None:
    """
    Save .npy containing 4D matrices into png

    Parameters:
    -----------
    file: str
        path to the .npy file
    savename: str
        name of the file to save. By default it will be saved in the same folder.
        To change it add a path prefix to the file name. For example: 'Output/filename'
    indextoextract: array
        Dimension to set to extract png, indexestoextract.shape must be 2
    more_than_8bits: bool, optional
        Precise if the image contains pixel coded with more than 8 bits or not
    Returns:
    --------
    """
    if indexestoextract is None:
        indexestoextract = [0, 0]

    if indexestoextract.shape != 2:
        raise ValueError('Error, for a 4D matrix you need to set 2 dimensions with provided vector, '
                         'then indexestoextract.shape must be 2')

    npyarray = np.load(file)
    npyarray = npyarray.astype(int)
    if more_than_8bits:
        npyarray = npyarray.astype('uint32')
    else:
        npyarray = npyarray.astype('uint8')
    data = im.fromarray(npyarray[indexestoextract[0], :, :, indexestoextract[1]])  # Choose what to extract accordingly
    # to the parameter indexestoextract, in the context of a 4D matrix. Of course to end up with a PNG, 2 dimension has
    #to be set
    pathtofolder, filename = os.path.split(file)
    data.save((pathtofolder + '/' + savename + '.png'))



## Use

def main():

    pathtofolder = "/data/lsancere/Data_General/TrainingSets/Hovernet/Carina-Corinna-Johannes-Data/ChrisSeg-LucasJohannesUpdatesClass/Hvn-Mc-annotations/NapariClassCorrection/TrainingDataGeneration/" + "/InfonVal_output/mat/InstancesTypes/ClassMaps/"
    savefoldername = '/images/'
    npystring = False
    makergbimage = False
    more_than_8bits = False
    npydim = 2

    fileslist = os.path.join(pathtofolder, '*.npy')
    fileslist = glob.glob(fileslist)
    os.makedirs(pathtofolder + savefoldername, exist_ok=True)

    if npystring:
        for fname in tqdm(fileslist):
            if os.path.exists(fname):
                path, extension = os.path.splitext(fname)  # split the file name and extension
                pathtofolder, filename = os.path.split(path)  # split the path and the file name
                # create the name of the file to save in the same place as the original file:
                savename = savefoldername + filename

                # Apply function only if the output file doesn't exist
                if not os.path.exists(pathtofolder + '/' + savename + '.png'):
                    # already (be careful Extension variable cannot be used here)
                    save2dnpy_2png(fname, savename)

    else:
        if npydim == 2:
            for fname in tqdm(fileslist):
                if os.path.exists(fname):
                    path, extension = os.path.splitext(fname)  # split the file name and extension
                    pathtofolder, filename = os.path.split(path)  # split the path and the file name
                    # create the name of the file to save in the same place as the original file:
                    savename = savefoldername + filename

                    # Apply function only if the output file doesn't exist
                    if not os.path.exists(pathtofolder + '/' + savename + '.png'):
                        # already (be careful Extension variable cannot be used here)
                        save2dnpy_2png(fname, savename, more_than_8bits=more_than_8bits)

        if npydim == 3:
            if not makergbimage:
                for k in range(0, 3):
                    # Index is related to the function, it is to extract the right dimensions from the 3D npy
                    index = k
                    for fname in tqdm(fileslist):
                        path, extension = os.path.splitext(fname)  # split the file name and extension
                        pathtofolder, filename = os.path.split(path)  # split the path and the file name
                        # create the name of the file to save in the same place as the original file:
                        savename = savefoldername + filename
                        # Apply function only if the output file doesn't exist:
                        if not os.path.exists(pathtofolder + '/' + savename + '.png'):
                                # already (be careful Extension variable cannot be used here)
                                save3dnpy_2png(fname, savename, index, make_rgbimage=makergbimage)

            else:
                for fname in tqdm(fileslist):
                    if os.path.exists(fname):
                        path, extension = os.path.splitext(fname)  # split the file name and extension
                        pathtofolder, filename = os.path.split(path)  # split the path and the file name
                        # create the name of the file to save in the same place as the original file:
                        savename = savefoldername + filename

                        # Apply function only if the output file doesn't exist:
                        if not os.path.exists(pathtofolder + '/' + savename + '.png'):
                            # already (be careful Extension variable cannot be used here)
                            save3dnpy_2png(fname, savename, None, make_rgbimage=makergbimage)


        if npydim == 4:
            print('Use of save4Dnpy_2png not written yet')

    print('Done')


if __name__ == "__main__":
    main()


