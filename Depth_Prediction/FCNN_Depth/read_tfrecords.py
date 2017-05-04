"""
@Author: Chris Birmingham

This file reads and batches tfrecords

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
import tensorflow as tf
import skimage.io as io
from PIL import Image
import numpy as np 
import tensorflow.contrib.slim as slim
import helper_funcs as helpers

# For this network both the input and output will be half sized
IMAGE_HEIGHT = 480  #480/2
IMAGE_WIDTH = 640   #640/2

tfrecords_filename = 'nyu_v2.tfrecords'


def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image,tf.float32)
    annotation = tf.decode_raw(features['mask_raw'], tf.float32)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    annotation_shape = tf.stack([height, width, 1])
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.uint8)
    annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)
    
    # TODO Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)
    
    resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)
    
    images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                 batch_size=10,
                                                 capacity=100,
                                                 num_threads=2,
                                                 allow_smaller_final_batch=True,
                                                 min_after_dequeue=10)         
    return(images, annotations)

def architecture(inputs):
    print(inputs.dtype,inputs.shape)
    with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      padding='SAME',
                      stride=1,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(inputs, 5, [2, 2], scope='conv1')
        for i in range(3,6):
            net = slim.conv2d(net, i*2, [i, i], scope='conv%d'% (i-1))
        net = slim.conv2d(net, 3, [4, 4], scope='conv5')
        net = slim.conv2d(net, 2, [3, 3], scope='conv6')
        net = slim.conv2d(net, 1, [2, 2], scope='conv7')

    return(net)


def input():
    # Create a filename queue
    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, annotation = read_and_decode(filename_queue)
    return(image, annotation)


def main():
    print('start')
    with tf.Graph().as_default():
        # Data loading
        print('input')
        image, annotation = input()
        # Define model
        print('model')
        predictions = architecture(image)
        # Specify loss function
        print('loss')
        loss = tf.losses.mean_squared_error(predictions, annotation)
        tf.summary.scalar('losses/loss', loss)
        # Specify optimization scheme
        print('optimizer')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
        # Create the training op
        print('training')
        train_op = slim.learning.create_train_op(loss, optimizer)
        print('GO')
        logdir = './log' # Where checkpoints are stored.
        slim.learning.train(
            train_op,
            logdir,
            number_of_steps=1000,
            save_summaries_secs=300,
            save_interval_secs=600)
    # The op for initializing the variables.

main()
def not_run():
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:
        
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # Let's read off 3 batches just for example
        for i in range(3):
        
            img, anno = sess.run([image, annotation])
            print(img[0, :, :, :].shape)
            print(anno[0].shape)
            
            print('current batch')
            
            # We selected the batch size of two
            # So we should get two image pairs in each batch
            # Let's make sure it is random

            helpers.show_picture(img[0, :, :, :])
            helpers.show_picture(helpers.scale(anno[0]))
            
        
        coord.request_stop()
        coord.join(threads)

