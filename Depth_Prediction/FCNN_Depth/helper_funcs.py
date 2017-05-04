import skimage.io as io
from PIL import Image
import numpy as np 



##### VISUALIZERS #######
def scale(array):
#   """ Useful for grayscale visualization """
    a_max= array.max()
    copy=(array*255/a_max).astype(np.uint8)
    print(type(copy))
    print(copy.shape)
    return(copy)

def save_picture(array,name='test.jpeg'):
    im =Image.fromarray(array)
    im.save(name)
    return()

def show_picture(array,name='test.jpeg'):
    im =Image.fromarray(array)
    im.show(name)
    return()