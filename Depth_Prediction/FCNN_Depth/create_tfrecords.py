"""
@Author: Chris Birmingham

This file creates tf records from numpy ndarrays

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
import tensorflow as tf
from PIL import Image
import numpy as np 
import skimage.io as io
import matplotlib.pyplot as plt
import helper_funcs as helpers

# Global Variables 
DEPTH_FILE = 'depths2.npy'
IMAGE_FILE = 'images2.npy'

tfrecords_filename = 'nyu_v2.tfrecords'

TOTAL_DATA = 1449
TRAIN_DATA = 1000
TEST_DATA = 200

####### LOADING ######### 
def load_images(verbose=False):
	all_images = np.load(IMAGE_FILE)
	all_depths = np.load(DEPTH_FILE)
	if verbose:
		print('images.npy shape:',all_images.shape)
		#print(type(all_images[1,1,1,1])) #uint8
		print('depths.npy shape:',all_depths.shape)
		print(type(all_depths[1,1,1]))
	return(all_images,all_depths)

###### FEATURES ######
def _int64_feature(value):
	# For labels
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	# For images
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

###### RECORDING #######
def write_record(name, images, annotations):
	writer = tf.python_io.TFRecordWriter(tfrecords_filename)

	files = images.shape[0]
	height = images.shape[1]
	width = images.shape[2]
	channels = images.shape[3]
	for file in range(files): # images are all the same size or shape info would loop
		img_raw = images[file].tostring()
		annotation_raw = annotations[file].tostring()

		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'image_raw': _bytes_feature(img_raw),
			'mask_raw': _bytes_feature(annotation_raw)}))

		writer.write(example.SerializeToString())
	writer.close()
	return()

def read_record(test=False,pic=0):
	""" Read record to test correct reconstruction """
	reconstructed_images=[]
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
	for string_record in record_iterator:
		example = tf.train.Example()	
		example.ParseFromString(string_record)

		height = int(example.features.feature['height']
	                                 .int64_list
	                                 .value[0])
	    
		width = int(example.features.feature['width']
	                                .int64_list
	                                .value[0])
	    
		img_string = (example.features.feature['image_raw']
	                                  .bytes_list
	                                  .value[0])
	    
		annotation_string = (example.features.feature['mask_raw']
	                                .bytes_list
	                                .value[0])
	    
		img_1d = np.fromstring(img_string, dtype=np.uint8)
		reconstructed_img = img_1d.reshape((height, width, -1))
	    
		annotation_1d = np.fromstring(annotation_string, dtype=np.float32) # depth values are float 32
		reconstructed_annotation = annotation_1d.reshape((height, width))

		reconstructed_images.append((reconstructed_img,reconstructed_annotation))
	if test:
		print('reconstructed_images shape', reconstructed_images[pic][0].shape)
		print('reconstructed_depths shape', reconstructed_images[pic][1].shape)
		helpers.save_picture(np.array(reconstructed_images[pic][0]),'reconstructed.jpeg')
		helpers.save_picture(scale(np.array(reconstructed_images[pic][1])),'reconstructedDepth.jpeg')
	return()


def main():
	all_images,all_depths = load_images(True)
	# TODO split into train eval and test
	pic=3
	helpers.save_picture(all_images[pic],'original.jpeg')
	write_record(tfrecords_filename,all_images,all_depths)
	read_record(test=False,pic=pic)
	return()




main()


