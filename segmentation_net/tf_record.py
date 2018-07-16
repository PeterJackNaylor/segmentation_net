#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package file tf_record

This module provides two main functions that are create_tfrecord and
compute_mean. Given a list of data generator object with the attributes 
of 'give_length' and 'next'.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(outname, dg_list):
    """
    Takes a list of data generator object  who has two methods: next and length 
    and creates an associated TFRecord file. 
    """
    writer = tf.python_io.TFRecordWriter(outname)
    for datagen in dg_list:
        n_iter_max = datagen.give_length()
        for _ in range(n_iter_max):
            img, annotation = datagen.next()
            annotation = annotation.astype(np.uint8)
            height = img.shape[0]
            width = img.shape[1]
            height_a = annotation.shape[0]
            width_a = annotation.shape[1]
            img_raw = img.tostring()
            annotation_raw = annotation.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height_img': _int64_feature(height),
                'width_img': _int64_feature(width),
                'height_mask': _int64_feature(height_a),
                'width_mask': _int64_feature(width_a),
                'image_raw': _bytes_feature(img_raw),
                'mask_raw': _bytes_feature(annotation_raw)}))
              
            writer.write(example.SerializeToString())
    writer.close()

def compute_mean(outname, dg_list):
    """
    Takes a list of data generator object  who has two methods: next and length 
    and creates an associated mean_file for RGB images.
    """
    mean_per_image = []
    for datagen in dg_list:
        n_iter_max = datagen.give_length()
        for _ in range(n_iter_max):
            img, _ = datagen.next()
            mean_per_image.append(np.mean(img, axis=(0, 1)))
    mean_array = np.mean(mean_per_image, axis=0)
    np.save(outname, mean_array)
    return mean_array
