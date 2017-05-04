# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:22:42 2017

@author: Crispy
"""

file = 'nyu_v2.mat'
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE.x = 640
IMAGE_SIZE.y = 480
IMAGE_SIZE.c = 3

# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100





#f = h5py.File(file,'r') 
#data = f.get('names') 
#data = np.array(data) # For converting to numpy array
#print(data)

#mat = scipy.io.loadmat(file)
depths = np.load('depths.npy')
images = np.load('images.npy')
labels = np.load('labels.npy')


def scale(array):
    a_max= array.max()
    return(array*255/a_max)

def picture(array):
    plt.imshow(array) 
    return()

picture(scale(depths[:,:,1]))



#def false_color(array):
#    a_max= array.max()
#    scaled_array = array*3*255/a_max
#    for i in np.nditer(scaled_array, op_flags=['readwrite']):
#        if i<256:
            
#images are 480x640