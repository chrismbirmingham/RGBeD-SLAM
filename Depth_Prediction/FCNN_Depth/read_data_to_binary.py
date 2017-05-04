# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:22:42 2017

@author: Crispy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm



MAT_FILE = 'nyu_v2.mat'
DEPTH_FILE = 'train_depths.npy'
IMAGE_FILE = 'train_images.npy'
LABELS_FILE = 'labels.npy'
LOAD_DATA = True
TOTAL_DATA = 1449
TRAIN_DATA = 1000
TEST_DATA = 200
VALIDATE_DATA = TOTAL_DATA - TRAIN_DATA - TEST_DATA


class NYUv2(object):
    def __init__(self):
        im_x_dim = 640
        im_y_dim = 480
        im_channels = 3
        self.meta_const = dict(
                image=dict(
                        shape = (im_x_dim,im_y_dim,im_channels),
                        dtype = 'float32'),
                depth=dict(
                        shape = (im_x_dim,im_y_dim),
                        dtype = 'float32'))
    def read_data(self):
#        self.train_ids = np.arange(TRAIN_DATA)
#        self.test_ids = np.arange(TRAIN_DATA,TEST_DATA)
        self.test_images = np.load(IMAGE_FILE)
        self.test_depths = np.load(DEPTH_FILE)
        
#        print('readingi')
#        self.all_images = np.load(IMAGE_FILE)
#        print('seperatei1')
#        self.training_set_images = self.all_images[self.train_ids]
#        print('seperatei2')
#        self.test_set_images = self.all_images[self.test_ids]
#        self.all_images = []

#        print('readingd')
#        self.all_depths = np.load(DEPTH_FILE)
#        print('seperated1')
#        self.training_set_depths = self.all_depths[self.train_ids]
#        print('seperated2')
#        self.test_set_depths = self.all_depths[self.test_ids]
#        self.all_depths = []
#        print('done')
        return()
                        
        
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images, depths, name):
  """Converts a dataset to tfrecords."""
#  images = data_set['images']
#  depths = data_set['depths']
  num_examples = depths.shape[0]
  print('Num examples:',num_examples)
  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = name + '.tfrecords'
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  Idx = np.arange(num_examples)
  np.random.shuffle(Idx)
  print('idx:',Idx)
  for index in Idx:
    image_raw = images[index].tostring()
    label_raw = depths[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label_raw': _bytes_feature(label_raw),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def main():
    print('start')
    data_sets = NYUv2()
    print('load data:')
    data_sets.read_data()
    print('Convert:')
#    print('start convertion')
#    convert_to(data_sets.training_set,'training')
#    print('done with traning set')
    convert_to(data_sets.test_images,data_sets.test_depths,'train')
    print('done')


def scale(array):
    a_max= array.max()
    return(array*255/a_max)

def picture(array):
    plt.imshow(array) 
    return()

def show(array):
    picture(scale(array))
    return()

main()