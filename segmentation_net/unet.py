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

    def max_double_conv(self, input, feat1, feat2, name):
        """
        A block constituted of 2 conv and
        """
        with tf.name_scope(name):
            max = self.max_pool(input, scope_name="{}/max/".format(name))
            double_conv = self.double_conv(max, feat1, feat2, name)

            return [max, *double_conv]

    def double_conv(self, input, feat1, feat2, name):
        with tf.name_scope(name):
            conv1 = self.conv_layer_f(input,
                                      feat1,
                                      feat2, 
                                      3, 
                                      scope_name="{}/conv1/".format(name))
            conv2 = self.conv_layer_f(conv1,
                                      feat2,
                                      feat2, 
                                      3, scope_name="{}/conv2/".format(name))
            return [conv1, conv2]

    def upsample_concat_double_conv(self, upsample_l, concat_l, feat1, feat2, name):
        with tf.name_scope(name):
            merged = self.trans_merge_layer_f(upsample_l, 
                                              concat_l,
                                              feat1,
                                              feat2,
                                              scope_name='{}/up_trans_merge/'.format(name))
            double_conv = self.double_conv(merged, feat1, feat2, name)
            return [merged, *double_conv]

    def logit_layer(self, input, feat, num_labels, name="classification"):
        strides = [1, 1, 1, 1]
        with tf.name_scope(name):
            self.last = self.conv_layer_f(input, feat,
                                          num_labels, 1, 
                                          strides=strides,
                                          scope_name="logit/")
            self.probability = tf.nn.softmax(self.last, axis=-1)

    def init_architecture(self, verbose):
        """
        Initialises variables for the graph
        """

        self.block11 = self.double_conv(self.input_node, self.num_channels, 
                                        self.n_features, "block11")
        self.block21 = self.max_double_conv(self.block11[-1], self.n_features, 
                                            2*self.n_features, "block21")
        self.block31 = self.max_double_conv(self.block21[-1], 2*self.n_features, 
                                            4*self.n_features, "block31")
        self.block41 = self.max_double_conv(self.block31[-1], 4*self.n_features, 
                                            8*self.n_features, "block41")
        self.block5 = self.max_double_conv(self.block41[-1], 8*self.n_features, 
                                            16*self.n_features, "block5")
        self.block42 = self.upsample_concat_double_conv(self.block5[-1], self.block41[-1],
                                                        16*self.n_features, 8*self.n_features, 
                                                        "block42")
        self.block32 = self.upsample_concat_double_conv(self.block42[-1], self.block31[-1], 
                                                        8*self.n_features, 4*self.n_features, 
                                                        "block32")
        self.block22 = self.upsample_concat_double_conv(self.block32[-1], self.block21[-1],  
                                                        4*self.n_features, 2*self.n_features, 
                                                        "block22")
        self.block12 = self.upsample_concat_double_conv(self.block22[-1], self.block11[-1], 
                                                        2*self.n_features, self.n_features,
                                                        "block12")
        self.logit_layer(self.block12[-1], self.n_features, self.num_labels)

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
