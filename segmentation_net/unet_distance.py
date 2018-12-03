#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DistanceUnet


In this module we implement an other child of the
SegmentationNet class which is the distance u-net.
We modify the graph evaluation as we now optimize
a least squared error.
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .unet_batchnorm import BatchNormedUnet

class DistanceUnet(BatchNormedUnet):
    """
    UNet object with batch normlisation after each convolution.
    """
    def __init__(self, image_size=(212, 212), log="/tmp/unet",
                 mean_array=None, num_channels=3, 
                 seed=42, verbose=0, n_features=2, fake_batch=1):

        self.label_int = None
        super(DistanceUnet, self).__init__(image_size, log, mean_array,
                                           num_channels, 1, seed, verbose, 
                                           n_features, fake_batch)

    def add_summary_images(self):
        """
        Croping UNet image so that it matches with GT.
        Also clipping distance map to check in tensorboard.
        """
        size1 = tf.shape(self.rgb_ph)[1]
        size_to_be = tf.cast(size1, tf.int32) - 2*self.displacement
        slicing = [0, self.displacement, self.displacement, 0]
        sliced_axis = [-1, size_to_be, size_to_be, -1]

        crop_input_node = tf.slice(self.rgb_ph, slicing, sliced_axis)

        input_s = tf.summary.image("input", crop_input_node, max_outputs=3)
        label_s = tf.summary.image("label_dist", self.lbl_ph, 
                                   max_outputs=3)
        lblint_s = tf.summary.image("label_int", tf.cast(self.lbl_int, tf.float32), 
                                    max_outputs=3)

        pred_dist = tf.clip_by_value(tf.cast(self.last, tf.float32), 0, 255)
        pred_dist_s = tf.summary.image("pred_dist", pred_dist, max_outputs=3)

        pred_s = tf.summary.image("pred", tf.cast(self.predictions, tf.float32),
                                  max_outputs=3)
        # label_pred = tf.cast(self.predictions, tf.uint8) * 255
        # label_label = tf.cast(self.label_int, tf.uint8) * 255

        # labelbin_s = tf.summary.image("label_bin", label_label, max_outputs=3)
        # predbin_s = tf.summary.image("pred_bin", label_pred, max_outputs=3)

        list_s = [input_s, label_s, pred_dist_s, lblint_s, pred_s]
        for __s in list_s:
            self.additionnal_summaries.append(__s)
            self.test_summaries.append(__s)

    def evaluation_graph(self, last, lbl_node):
        """
        Graph optimization part, here we define the loss and how the model is evaluated
        """
        with tf.name_scope('evaluation'):
            predictions = tf.cast(tf.clip_by_value(tf.round(last), 0, 1), tf.uint8)
            binary = tf.clip_by_value(tf.round(lbl_node), 0, 1)
            self.lbl_int = tf.cast(binary, tf.uint8)

            with tf.name_scope('loss'):
                mse_ = tf.losses.mean_squared_error(last, lbl_node)
                loss = tf.reduce_mean(mse_)
                # loss_sum = tf.summary.scalar("mean_squared_error", loss)
                
                # This trick ensures that self.label_int and self.label_node followed the same path
                # through the tensorflow graph.
                # dummy_tf = tf.Variable(0., validate_shape=False, name="dummy_tf")
                # ensuring_label_int = tf.assign(dummy_tf, self.loss + 0)
                # with tf.control_dependencies([ensuring_label_int]):
                #     binary = tf.clip_by_value(tf.round(self.label_node), 0, 1)
                #     self.label_int = tf.cast(binary, tf.int64)


            # if self.tensorboard:
            #     self.additionnal_summaries.append(loss_sum)
                # Disabled as we need several steps averaged
                # self.test_summaries.append(loss_sum)

            if self.list_metrics:
                lbl_lbl = tf.squeeze(self.lbl_int, axis=[3])
                lbl_pred = tf.squeeze(predictions, axis=[3])
                self.metric_handler = self.tf_compute_metrics(lbl_lbl, lbl_pred, 
                                                              self.list_metrics)
        
        if self.verbose > 1:
            tqdm.write('computational graph initialised')

        return loss, predictions

    def logit_layer(self, input_, feat, num_labels, name="regression"):
        strides = [1, 1, 1, 1]
        with tf.name_scope(name):
            last = self.conv_layer_f(input_, feat,
                                     num_labels, 1, 
                                     strides=strides,
                                     scope_name="logit/")
            probability = last
        return probability, last
