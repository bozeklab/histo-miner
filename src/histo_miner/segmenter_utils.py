#Lucas Sancéré -

from PIL import Image
import numpy as np
from itertools import product
import copy


##### SEGMENTER OUTPUT CONVERSIONS AND PROCESSING


def change_pix_values(file: str, valuestochange: list,
                      newvalues: list):
    """
    Change given pixel values into given newvalues.
    Follow the valuestochange list and newvalues list in index order.
    Needs images with one channel as input.
    Overwritte the previous image with by new one. Not working with RGB images,
    or images with more than 1 channel.

    Parameters:
    -----------
    file: str
        path to the image. The extension of the image can be any PILLOW supported format
    valuestochange: list
        List of values user wants to change into newvalues (integer numbers)
    newvalues: list
        List of Newvalues of the pixel (integer numbers)
    Returns:
    --------
    """
    image = Image.open(file)  # PIL support lots of formats
    array = np.array(image)  # Transform PIL Image format into numpy array
    newarray = copy.deepcopy(array)
    if len(newarray.shape) != 2:
        raise ValueError('The image must contain only 1 channel. '
                         'For instance RGB images cannot be processed. '
                         'Use ChangePixelValueRGB function instead')
    # Case where there is only one value to change and the input is a int
    if isinstance(valuestochange, int):
        for x, y in product(range(array.shape[0]), range(array.shape[1])):
            if array[x, y] == valuestochange:
                newarray[x, y] = newvalues
    # Case wherre there is one or more values to change and the input is a list
    else:
        for index in range(len(valuestochange)):
            for x, y in product(range(array.shape[0]), range(array.shape[1])):
                if array[x, y] == int(valuestochange[index]):
                    newarray[x, y] = int(newvalues[index])
    new_image = Image.fromarray(newarray)
    new_image.save(file)