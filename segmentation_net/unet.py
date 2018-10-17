#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" implementation of the unet


In this module we define two childs of the main class
SegmentationNet. We modify the init_architecture method 
of the class to represent the computationnal graph of the 
original U-net implementation. We propose two methods,
one with valid padding and the one without.
"""


import tensorflow as tf
from tqdm import tqdm


from . import utils_tensorflow as ut
from .segmentation_net import SegmentationNet


class Unet(SegmentationNet):
    """
    U-net implementation with same padding.
    """
    def __init__(self, image_size=(256, 256), log="/tmp/unet",
                 num_channels=3, num_labels=2, tensorboard=True,
                 seed=42, verbose=0, n_features=2):
        self.n_features = n_features
        self.padding_for_conv = "SAME"
        super(Unet, self).__init__(image_size, log, num_channels,
                                   num_labels, tensorboard, seed, verbose, 0)

    def conv_layer_f(self, i_layer, s_input_channel, size_output_channel,
                     ks, strides=[1, 1, 1, 1], scope_name="conv", random_init=0.1, 
                     padding="SAME"):
        """
        Changed default padding to Valid
        """
        # return SegNet.conv_layer_f(self, i_layer, s_input_channel, size_output_channel,
        #                            ks, strides, scope_name, random_init, padding)
        return super(Unet, self).conv_layer_f(i_layer, s_input_channel, size_output_channel,
                                              ks, strides, scope_name, random_init, 
                                              padding=self.padding_for_conv)

    def trans_merge_layer_f(self, to_transpose_layer, to_concatenate_layer, 
                            in_channel, out_channel, kernel_size=2, 
                            random_init=0.1, scope_name="transpose_crop_merge"):
        """
        Transpose one layer, concatenate it to another one
        """
        with tf.name_scope(scope_name):
            tconv_weights = self.weight_xavier(kernel_size, out_channel, in_channel)
            tconv_biases = ut.biases_const_f(random_init, out_channel)
            transposed_layer = ut.transposeconv_layer_f(to_transpose_layer, tconv_weights)
            tconv_with_bias = tf.nn.bias_add(transposed_layer, tconv_biases)
            act = self.non_activation_f(tconv_with_bias)
            output = ut.crop_and_merge(to_concatenate_layer, act)

            self.training_variables.append(tconv_weights)
            self.training_variables.append(tconv_biases)
            self.var_to_reg.append(tconv_weights)


            if self.tensorboard:
                self.var_to_summarise.append(tconv_weights)
                self.var_to_summarise.append(tconv_biases)
                self.var_to_summarise.append(transposed_layer)
                self.var_activation_to_summarise.append(act)
            return output

    def add_summary_images(self):
        """
        Croping Unet image so that it matches with GT.
        Image summary to add to the summary
        TODO Does not work with the cond in the network flow...
        """
        size1 = tf.shape(self.input_node)[1]
        size_to_be = tf.cast(size1, tf.int32) - 2*self.displacement
        slicing = [0, self.displacement, self.displacement, 0]
        sliced_axis = [-1, size_to_be, size_to_be, -1]
        crop_input_node = tf.slice(self.input_node, slicing, sliced_axis)

        input_s = tf.summary.image("input", crop_input_node, max_outputs=4)
        label_s = tf.summary.image("label", self.label_node, max_outputs=4)
        pred = tf.expand_dims(tf.cast(self.predictions, tf.float32), dim=3)
        predi_s = tf.summary.image("pred", pred, max_outputs=4)
        for __s in [input_s, label_s, predi_s]:
            self.additionnal_summaries.append(__s)
            self.test_summaries.append(__s)

    def max_pool(self, i_layer, padding="VALID", scope_name="max_pool/"):
        """
        Perform max pooling operation
        """
        with tf.name_scope(scope_name):
            return ut.max_pool(i_layer, padding=padding)


    def init_architecture(self, verbose):
        """
        Initialises variables for the graph
        """
        with tf.name_scope('architecture'): 
            strides = [1, 1, 1, 1]
            self.conv11 = self.conv_layer_f(self.input_node,
                                            self.num_channels,
                                            self.n_features, 
                                            3, scope_name="conv11/")
            self.conv12 = self.conv_layer_f(self.conv11,
                                            self.n_features,
                                            self.n_features, 
                                            3, scope_name="conv12/")
            self.max1 = self.max_pool(self.conv12, scope_name="max1/")
            self.conv21 = self.conv_layer_f(self.max1,
                                            self.n_features,
                                            2*self.n_features, 
                                            3, scope_name="conv21/")
            self.conv22 = self.conv_layer_f(self.conv21,
                                            2*self.n_features,
                                            2*self.n_features, 
                                            3, scope_name="conv22/")
            self.max2 = self.max_pool(self.conv22, scope_name="max2/")
            self.conv31 = self.conv_layer_f(self.max2,
                                            2*self.n_features,
                                            4*self.n_features, 
                                            3, scope_name="conv31/")
            self.conv32 = self.conv_layer_f(self.conv31,
                                            4*self.n_features,
                                            4*self.n_features, 
                                            3, scope_name="conv32/")
            self.max3 = self.max_pool(self.conv32, scope_name="max3/")
            self.conv41 = self.conv_layer_f(self.max3,
                                            4*self.n_features,
                                            8*self.n_features, 
                                            3, scope_name="conv41/")
            self.conv42 = self.conv_layer_f(self.conv41,
                                            8*self.n_features,
                                            8*self.n_features, 
                                            3, scope_name="conv42/")
            self.max4 = self.max_pool(self.conv42, scope_name="max4/")
            self.conv51 = self.conv_layer_f(self.max4,
                                            8*self.n_features,
                                            16*self.n_features, 
                                            3, scope_name="conv51/")
            self.conv52 = self.conv_layer_f(self.conv51,
                                            16*self.n_features,
                                            16*self.n_features, 
                                            3, scope_name="conv52/")
            self.merged4 = self.trans_merge_layer_f(self.conv52, 
                                                    self.conv42,
                                                    16*self.n_features,
                                                    8*self.n_features,
                                                    scope_name='trans_merge_merge4/')
            self.conv43 = self.conv_layer_f(self.merged4,
                                            16*self.n_features,
                                            8*self.n_features, 
                                            3, scope_name="conv43/")
            self.conv44 = self.conv_layer_f(self.conv43,
                                            8*self.n_features,
                                            8*self.n_features, 
                                            3, scope_name="conv44/")
            self.merged3 = self.trans_merge_layer_f(self.conv44, 
                                                    self.conv32,
                                                    8*self.n_features,
                                                    4*self.n_features,
                                                    scope_name='trans_merge_merge3/')
            self.conv33 = self.conv_layer_f(self.merged3,
                                            8*self.n_features,
                                            4*self.n_features, 
                                            3, scope_name="conv33/")
            self.conv34 = self.conv_layer_f(self.conv33,
                                            4*self.n_features,
                                            4*self.n_features, 
                                            3, scope_name="conv34/")
            self.merged2 = self.trans_merge_layer_f(self.conv34, 
                                                    self.conv22,
                                                    4*self.n_features,
                                                    2*self.n_features,
                                                    scope_name='trans_merge_merge2/')
            self.conv23 = self.conv_layer_f(self.merged2,
                                            4*self.n_features,
                                            2*self.n_features, 
                                            3, scope_name="conv23/")
            self.conv24 = self.conv_layer_f(self.conv23,
                                            2*self.n_features,
                                            2*self.n_features, 
                                            3, scope_name="conv24/")
            self.merged1 = self.trans_merge_layer_f(self.conv24, 
                                                    self.conv12,
                                                    2*self.n_features,
                                                    self.n_features,
                                                    scope_name='trans_merge_merge1/')
            self.conv13 = self.conv_layer_f(self.merged1,
                                            2*self.n_features,
                                            self.n_features, 
                                            3, scope_name="conv13/")
            self.conv14 = self.conv_layer_f(self.conv13,
                                            self.n_features,
                                            self.n_features, 
                                            3, scope_name="conv14/")

            self.logit = self.conv_layer_f(self.conv14, self.n_features,
                                           self.num_labels, 1, 
                                           strides=strides,
                                           scope_name="logit/")
            self.probability = tf.nn.softmax(self.logit, axis=-1)
            self.last = self.logit
        if verbose > 1:
            tqdm.write('model variables initialised')

class UnetPadded(Unet):
    """
    U-net implementation with valid padding.
    """
    def __init__(self, image_size=(212, 212), log="/tmp/unet",
                 num_channels=3, num_labels=2, tensorboard=True,
                 seed=42, verbose=0, n_features=2):
        self.n_features = n_features
        self.padding_for_conv = "VALID"
        super(Unet, self).__init__(image_size, log, num_channels,
                                   num_labels, tensorboard, seed, 
                                   verbose, 92)
