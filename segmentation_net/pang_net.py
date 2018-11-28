#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package mother class is defined here: SegNet


In this module we implement a object defined as SegNet.
SegNet is a generic class from which we derive many
of the more complex networks. Pang-net, U-net, D-net,
and possibly many others. SegmentationNet is the core of many 
of the structures used and implements many basic functions.
Such as a convolutional layer, ordering of the initialization
procedure and creation of the graph needed by tensorflow.

This module is not intented to be run as but can be.
It implements PangNet

"""

# import sys
# from datetime import datetime

# import numpy as np
# import tensorflow as tf
# from tqdm import tqdm, trange

# from . import utils_tensorflow as ut
# from .data_read_decode import read_and_decode
# from .net_utils import ScoreRecorder
# from .tensorflow_metrics import TFMetricsHandler, metric_bundles_function
# from .utils import check_or_create, element_wise, merge_dictionnary

# MAX_INT = sys.maxsize

from .segmentation_class.segmentation_train import *



softmax = tf.nn.sparse_softmax_cross_entropy_with_logits

class PangNet(SegmentationTrain):

    def evaluation_graph(self, last, lbl_node):
        """
        Graph optimization part, here we define the loss and how the model is evaluated
        """

        with tf.name_scope('evaluation'):
            predictions = tf.argmax(last, axis=3, output_type=tf.int32)
            with tf.name_scope('loss'):
                lbl_int32 = tf.cast(lbl_node, tf.int32)
                labels = tf.squeeze(lbl_int32, axis=[3])
                loss = tf.reduce_mean(softmax(logits=last,
                                              labels=labels, name="entropy"))
            # if self.tensorboard:
            #     loss_sum = tf.summary.scalar("entropy", loss)
            #     self.additionnal_summaries.append(loss_sum)
                # Disabled as we need several steps averaged
                # self.test_summaries.append(loss_sum)
            if self.list_metrics:
                self.metric_handler = self.tf_compute_metrics(labels, predictions, 
                                                              self.list_metrics)
        
        #tf.global_variables_initializer().run()
        if self.verbose > 1:
            tqdm.write('computational graph initialised')

        return loss, predictions

    def init_architecture(self, rgb_node):
        """
        Initialises variables for the graph
        """
        strides = [1, 1, 1, 1]

        input_network = self.subtract_mean(rgb_node)
        self.conv1 = self.conv_layer_f(input_network,
                                       self.num_channels,
                                       8, 3, strides=strides, 
                                       scope_name="conv1/")
        self.conv2 = self.conv_layer_f(self.conv1, 8,
                                       8, 3, strides=strides,
                                       scope_name="conv2/")
        self.conv3 = self.conv_layer_f(self.conv2, 8,
                                       8, 3, strides=strides,
                                       scope_name="conv3/")
        self.logit = self.conv_layer_f(self.conv3, 8,
                                       self.num_labels, 1, 
                                       strides=strides,
                                       scope_name="logit/")

        with tf.name_scope('final_layers'): 
            probability = tf.nn.softmax(self.logit, axis=-1)
            last = self.logit

        if self.verbose > 1:
            tqdm.write('model variables initialised')

        return probability, last

