#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package 

Helper tensorflow functions for the package.
We implement useful functions to deal with tensorboard,
weight initialization function (glorot) and specific layers.

"""

from itertools import compress

import tensorflow as tf

def add_activation_summary(var):
    """
    Add activation summary with information about sparsity
    """
    hist_s, scal_s = None, None
    if var is not None:
        hist_s = tf.summary.histogram(var.op.name + "/activation", var)
        scal_s = tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))
    return [hist_s, scal_s]
def add_gradient_summary(grad, var):
    """
    Add gradiant summary to summary
    """
    summary = None
    if grad is not None:
        summary = tf.summary.histogram(var.op.name + "/gradient", grad)
    return summary
def add_to_summary(var):
    """
    Adds histogram for each parameter in var
    """
    summary = None
    if var is not None:
        summary = tf.summary.histogram(var.op.name, var)
    return summary
def add_value_to_summary(score, name, summary_writer, step, tag="test"):
    """
    Add list of values (with corresponding names) to a given summary
    writer. It will be saved with the name step.
    """
    summary = tf.Summary()
    #for value, name in zip(scores, names):
    summary.value.add(tag="{}/{}".format(tag, name), simple_value=score)
    summary_writer.add_summary(summary, step)

def batch_normalization(input, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5, seed=None):
    """
    Performs batch normalisation.
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.name_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name="beta")
        gamma = tf.Variable(tf.random_normal([n_out], 1.0, 0.02, seed=seed))
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            """
            Update mean variable of ema.
            """
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, eps)
    return normed, beta, gamma, ema.variables_to_restore()


def biases_const_f(const, shape, name="b"):
    """
    Initialises biais
    """
    biais = tf.Variable(tf.constant(const, shape=[shape]), name=name)
    return biais

def drop_out_layer(input_layer, keep_prob=0.5, scope="drop_out", seed=None):
    """
    Performs drop out on the input layer
    """
    with tf.name_scope(scope):
        return tf.nn.dropout(input_layer, keep_prob, seed)

def crop_and_merge(input1, input2, name="bridge"):
    """
    Crop input1 so that it matches input2 and then
    return the concatenation of both channels.
    """
    size1_x = tf.shape(input1)[1]
    size2_x = tf.shape(input2)[1]

    size1_y = tf.shape(input1)[2]
    size2_y = tf.shape(input2)[2]
    with tf.name_scope(name):
        diff_x = tf.divide(tf.subtract(size1_x, size2_x), 2)
        diff_y = tf.divide(tf.subtract(size1_y, size2_y), 2)
        diff_x = tf.cast(diff_x, tf.int32)
        size2_x = tf.cast(size2_x, tf.int32)
        diff_y = tf.cast(diff_y, tf.int32)
        size2_y = tf.cast(size2_y, tf.int32)
        crop = tf.slice(input1, [0, diff_x, diff_y, 0], [-1, size2_x, size2_y, -1])
        concat = tf.concat([crop, input2], axis=3)

        return concat

def initialize_uninitialized_vars(sess, feed_dict, extra_variables=[]):
    """
    initialize all uninitialized variables in tensorflow graph.
    """
    #import pdb; pdb.set_trace()
    global_vars = tf.global_variables() + tf.local_variables() + extra_variables
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var))
                                   for var in global_vars],
                                  feed_dict=feed_dict)
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if not_initialized_vars:
        sess.run(tf.variables_initializer(not_initialized_vars), feed_dict=feed_dict)

WINDOW_22 = [1, 2, 2, 1]
def max_pool(i_layer, ksize=WINDOW_22, strides=WINDOW_22,
             padding="SAME", name="max_pool"):
    """
    Performs max pool operation
    """
    return tf.nn.max_pool(i_layer, ksize=ksize, strides=strides, 
                          padding=padding, name=name)

def record_size(record):
    """
    return length of a given record.
    """
    return len([el for el in tf.python_io.tf_record_iterator(record)])


def weight_const_f(kernel_size, inchannels, outchannels, stddev, name="W", seed=None):
    """
    Defining parameter to give to a convolution layer
    """
    shape = [kernel_size, kernel_size, inchannels, outchannels]
    init_tn = tf.truncated_normal(shape, stddev=stddev, seed=seed)
    weight_matrix = tf.Variable(init_tn, name=name)
    return weight_matrix

def weight_xavier(kernel_size, in_channels, out_channels, name="W", seed=None):
    """
    Initialises a convolution kernel for a convolution layer with Xavier initialising
    """
    xavier_std = (1. / float(kernel_size * kernel_size * in_channels)) ** 0.5
    return weight_const_f(kernel_size, in_channels, out_channels, 
                          xavier_std, name=name, seed=seed)

def transposeconv_layer_f(i_layer, weight, scope_name="tconv", padding="VALID"):
    """
    Transpose convolution layer
    """
    with tf.name_scope(scope_name):
        i_shape = tf.shape(i_layer)
        o_shape = tf.stack([i_shape[0], i_shape[1]*2, i_shape[2]*2, i_shape[3]//2])
        return tf.nn.conv2d_transpose(i_layer, weight, output_shape=o_shape,
                                      strides=WINDOW_22, padding=padding)

def transpose_crop_merge(to_transpose_layer, to_concatenate_layer, in_channel,
                         out_channel, kernel_size=2, random_init=0.1, 
                         scope_name="transpose_crop_merge", seed=None):
    """

    """
    with tf.name_scope(scope_name):
        tconv_weights = weight_xavier(kernel_size, out_channel, in_channel, "tconv_weights/", seed)
        tconv_biases = biases_const_f(random_init, out_channel, "tconv_biases/")
        transposed_layer = transposeconv_layer_f(to_transpose_layer, tconv_weights)
        tconv_with_bias = tf.nn.bias_add(transposed_layer, tconv_biases)
        relu_out = tf.nn.relu(tconv_with_bias)
        output = crop_and_merge(relu_out, to_concatenate_layer)
        return output


