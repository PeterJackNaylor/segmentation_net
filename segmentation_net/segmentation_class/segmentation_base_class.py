#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package file tf_record

Segmentation_base_class ->  SegmentationInput -> SegmentationCompile -> SegmentationSummaries

"""

import sys
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from numpy import random, zeros
from tqdm import tqdm

from .. import utils_tensorflow as ut
from ..tensorflow_metrics import metric_bundles_function
from ..utils import check_or_create

MAX_INT = sys.maxsize

class SegmentationBaseClass:

    __metaclass__ = ABCMeta
    def __init__(self, image_size=(224, 224), log="/tmp/net",
                 mean_array=None, num_channels=3, num_labels=2,
                 seed=None, verbose=0, displacement=0, fake_batch=1):
        ## this doesn't work yet... 
        ## when the graph is initialized it is completly equal everywhere.
        ## https://github.com/tensorflow/tensorflow/issues/9171
        random.seed(seed)
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        # Parameters
        self.image_size = image_size
        self.log = log
        self.mean_array = mean_array
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.seed = seed
        self.verbose = verbose
        self.displacement = displacement #Variable important for the init queues

        check_or_create(log)

        # Preparing inner variables
        self.sess = tf.InteractiveSession()
        self.sess.as_default()
        
        self.tensorboard = True
        self.gradient_to_summarise = []
        self.test_summaries = []
        self.additionnal_summaries = []
        self.training_variables = []
        self.var_activation_to_summarise = []
        self.var_to_reg = []
        self.var_to_summarise = []
        self.list_metrics = metric_bundles_function(num_labels)

        tqdm.write("FAKE BATCH is only for the ResourceVariable that have the validate_shape ignored")
        self.fake_batch = fake_batch

        # Preparing tensorflow variables
        self.setup_mean()
        self.rgb_v, self.lbl_v, self.rgb_ph, self.lbl_ph = self.setup_input_network()
        self.probability, self.last = self.init_architecture(self.rgb_ph)
        self.loss, self.predictions = self.evaluation_graph(self.last, self.lbl_ph)

        self.saver = self.saver_object(self.log, 3, restore=True)

    def conv_layer_f(self, i_layer, s_input_channel, size_output_channel,
                     ks, strides=[1, 1, 1, 1], scope_name="conv", random_init=0.1, 
                     padding="SAME"):
        """
        Defining convolution layer
        """
        with tf.name_scope(scope_name):
            weights = self.weight_xavier(ks, s_input_channel, 
                                         size_output_channel)
            biases = ut.biases_const_f(random_init, size_output_channel)
            conv_layer = tf.nn.conv2d(i_layer, weights, 
                                      strides=strides, 
                                      padding=padding)
            conv_with_bias = tf.nn.bias_add(conv_layer, biases)
            act = self.non_activation_f(conv_with_bias)

            self.training_variables.append(weights)
            self.training_variables.append(biases)
            self.var_to_reg.append(weights)

            if self.tensorboard:
                self.var_to_summarise.append(weights)
                self.var_to_summarise.append(biases)
                self.var_to_summarise.append(conv_layer)
                self.var_activation_to_summarise.append(act)
            return act

    def drop_out(self, i_layer, keep_prob=0.5, scope="drop_out"):
        """
        Performs drop out layer and checks if the seed is set
        """
        if self.seed:
            seed = random.randint(MAX_INT)
        else: 
            seed = self.seed
        return ut.drop_out_layer(i_layer, keep_prob, scope, seed)


    def non_activation_f(self, i_layer):
        """
        Defining relu layer.
        The default is relu. Feel free to redefine non activation
        function for the standard convolution_
        """
        act = tf.nn.relu(i_layer)
        return act

    def weight_xavier(self, kernel, s_input_channel, size_output_channel):
        """
        xavier initialization with 'random seed' if seed is set.
        """
        if self.seed:
            seed = random.randint(MAX_INT)
        else:
            seed = self.seed
        weights = ut.weight_xavier(kernel, s_input_channel, 
                                   size_output_channel, seed=seed)
        return weights

    def restore(self, saver, log):
        """
        restore the model
        if no log files exit, it re initializes the architecture variables
        """
        ckpt = tf.train.get_checkpoint_state(log)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            if self.verbose:
                tqdm.write("model restored...")
        else:
            if self.verbose:
                tqdm.write("values have been initialized")
            self.init_uninit([])
            # self.initialize_all()

    def saver_object(self, log, keep=3, restore=False):
        """
        Defining the saver, it will load if possible.
        """
        if self.verbose > 1:
            tqdm.write("setting up saver...")
        saver = tf.train.Saver(self.training_variables, max_to_keep=keep)
        if log and restore:
            self.restore(saver, log)
        return saver

    # def setup_init(self):
    #     """
    #     Initialises everything that has not been initialized.
    #     TODO: better initialization for data
    #     """
    #     with tf.name_scope("initialisation"):
    #         extra_variables = []

    #         test_init = self.data_init_test is not None
    #         train_init = self.data_init is not None

    #         if train_init and test_init: 
    #             init_op = tf.group(self.data_init, self.data_init_test)
    #         elif train_init:
    #             init_op = tf.group(self.data_init)
    #         elif test_init:
    #             init_op = tf.group(self.data_init_test)
    #         else:
    #             if self.verbose:
    #                 tqdm.write("No data initialization possible")
    #     # self.init_uninit(extra_variables)
    #     self.sess.run(init_op)

    # def initialize_all(self):
    #     """
    #     initialize all variables
    #     """
    #     self.sess.run(tf.global_variables_initializer())

    def init_uninit(self, extra_variables):
        ut.initialize_uninitialized_vars(self.sess, {self.is_training: False},
                                         extra_variables)

    @abstractmethod
    def init_architecture(self):
        """
        Sets the tensorflow graph associated to the network architecture.
        """
        pass

    @abstractmethod
    def evaluation_graph(self, metrics):
        """
        Sets the tensorflow graph associated to the evaluation
        """
        pass
