#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DistanceUnet


In this module we implement an other child of the
SegmentationNet class which is the distance u-net.
We modify the graph evaluation as we now optimize
a least squared error.
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
 
        input_s = tf.summary.image("input", crop_input_node, max_outputs=3)
        label_s = tf.summary.image("label", self.label_node, max_outputs=3)
 
        input_s = tf.summary.image("input", crop_input_node, max_outputs=3)
        label_s = tf.summary.image("label_dist", self.label_node, max_outputs=3)
        pred_dist_s = tf.summary.image("pred_dist", tf.cast(self.predictions, tf.float32), 
                                       max_outputs=4)

        clip_round_pred = tf.clip_by_value(tf.round(self.predictions), 0, 1)
        label_pred = tf.cast(clip_round_pred, tf.uint8) * 255

        clip_round_label = tf.clip_by_value(tf.round(self.label_node), 0, 1)
        label_label = tf.cast(clip_round_label, tf.uint8) * 255

        labelbin_s = tf.summary.image("label_bin", label_label, max_outputs=3)
        predbin_s = tf.summary.image("pred_bin", label_pred, max_outputs=3)

        list_s = [input_s, label_s, pred_dist_s, labelbin_s, predbin_s]
        for __s in list_s:
            self.additionnal_summaries.append(__s)
            self.test_summaries.append(__s)



    def evaluation_graph(self, list_metrics, verbose=0):
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
            if list_metrics:
                self.tf_compute_metrics(label_label, label_pred, list_metrics)
        
        #tf.global_variables_initializer().run()
        if verbose > 1:
            tqdm.write('computational graph initialised')

    def logit_layer(self, input, feat, num_labels, name="regression"):
        strides = [1, 1, 1, 1]
        with tf.name_scope(name):
            self.last = self.conv_layer_f(input, feat,
                                          num_labels, 1, 
                                          strides=strides,
                                          scope_name="logit/")
