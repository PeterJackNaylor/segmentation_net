#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" implementation of the unet with batch norm


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


from . import utils_tensorflow as ut
from .unet import UnetPadded

class BatchNormedUnet(UnetPadded):
    """
    UNet object with batch normlisation after each convolution.
    """
    def conv_layer_f(self, i_layer, s_input_channel, size_output_channel,
                     ks, strides=[1, 1, 1, 1], scope_name="conv", random_init=0.1, 
                     padding="VALID"):
        """
        Defining convolution layer
        """
        with tf.name_scope(scope_name):
            weights = ut.weight_xavier(ks, s_input_channel, 
                                       size_output_channel)
            biases = ut.biases_const_f(random_init, size_output_channel)
            conv_layer = tf.nn.conv2d(i_layer, weights, 
                                      strides=strides, 
                                      padding=padding)
            n_out = weights.shape[3].value
            batch_normed, beta, gamma, others = ut.batch_normalization(conv_layer, n_out, self.is_training)
            conv_with_bias = tf.nn.bias_add(batch_normed, biases)
            act = self.non_activation_f(conv_with_bias)

            #self.training_variables.append(weights)
            #self.training_variables.append(biases)
            #self.training_variables.append(beta)
            #self.training_variables.append(gamma)
            
            self.var_to_reg.append(weights)
            for _, value in others.items():
                self.training_variables.append(value)
            if self.tensorboard:
                self.var_to_summarise.append(weights)
                self.var_to_summarise.append(biases)
                self.var_to_summarise.append(conv_layer)
                self.var_to_summarise.append(beta)
                self.var_to_summarise.append(gamma)
                self.var_activation_to_summarise.append(act)
            return act
