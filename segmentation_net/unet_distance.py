#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DistanceUnet


In this module we implement a object defined as SegNet.
SegNet is an abstract class from which we derive many
of the more complex networks. Pang-net, U-net, D-net,
and possibly many others. SegNet is the core of many 
of the structures used and implements many basic functions.
Such as a convolutional layer, ordering of the initialization
procedure and creation of the graph needed by tensorflow.

This module is not intented to be run as the class is abstract.

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
from tqdm import tqdm

from .unet_batchnorm import BatchNormedUnet

class DistanceUnet(BatchNormedUnet):
    """
    UNet object with batch normlisation after each convolution.
    """
    def __init__(self, image_size=(212, 212), log="/tmp/unet",
                 num_channels=3, tensorboard=True, seed=42,
                 verbose=0, n_features=2):
        super(DistanceUnet, self).__init__(image_size, log, num_channels,
                                           1, tensorboard, seed, verbose, 
                                           n_features)

    def add_summary_images(self):
        """
        Croping UNet image so that it matches with GT.
        Also clipping distance map to check in tensorboard.
        """
        size1 = tf.shape(self.input_node)[1]
        size_to_be = tf.cast(size1, tf.int32) - 2*self.displacement
        slicing = [0, self.displacement, self.displacement, 0]
        sliced_axis = [-1, size_to_be, size_to_be, -1]

        crop_input_node = tf.slice(self.input_node, slicing, sliced_axis)
 
        input_s = tf.summary.image("input", crop_input_node, max_outputs=4)
        label_s = tf.summary.image("label", self.label_node, max_outputs=4)
 
        input_s = tf.summary.image("input", crop_input_node, max_outputs=4)
        label_s = tf.summary.image("label_dist", self.label_node, max_outputs=4)
        pred_dist_s = tf.summary.image("pred_dist", tf.cast(self.predictions, tf.float32), 
                                       max_outputs=4)

        clip_round_pred = tf.clip_by_value(tf.round(self.predictions), 0, 1)
        label_pred = tf.cast(clip_round_pred, tf.uint8) * 255

        clip_round_label = tf.clip_by_value(tf.round(self.label_node), 0, 1)
        label_label = tf.cast(clip_round_label, tf.uint8) * 255

        labelbin_s = tf.summary.image("label_bin", label_label, max_outputs=4)
        predbin_s = tf.summary.image("pred_bin", label_pred, max_outputs=4)

        list_s = [input_s, label_s, pred_dist_s, labelbin_s, predbin_s]
        for __s in list_s:
            self.additionnal_summaries.append(__s)
            self.test_summaries.append(__s)



    def evaluation_graph(self, verbose=0):
        """
        Graph optimization part, here we define the loss and how the model is evaluated
        """
        with tf.name_scope('evaluation'):
            self.probability = self.last #Correcting what is in the init_architecture method
            self.predictions = self.last

            with tf.name_scope('loss'):
                mse_ = tf.losses.mean_squared_error(self.last, self.label_node)
                self.loss = tf.reduce_mean(mse_)
                loss_sum = tf.summary.scalar("mean_squared_error", self.loss)


            round_pred = tf.clip_by_value(tf.round(self.last), 0, 1)
            label_pred = tf.squeeze(tf.cast(round_pred, tf.int64), squeeze_dims=[3])

            round_label = tf.clip_by_value(tf.round(self.label_node), 0, 1)
            label_label = tf.squeeze(tf.cast(round_label, tf.int64), squeeze_dims=[3])

            if self.tensorboard:
                self.additionnal_summaries.append(loss_sum)
                # Disabled as we need several steps averaged
                # self.test_summaries.append(loss_sum)
            self.tf_compute_metrics(label_pred, label_label)
        
        #tf.global_variables_initializer().run()
        if verbose > 1:
            tqdm.write('computational graph initialised')