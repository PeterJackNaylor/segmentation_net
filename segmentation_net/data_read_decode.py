#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" input pipeline to tensorflow


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
import numpy as np
import scipy.stats as st

PI = 3.14159

def coin_flip(p):
    """
    Perform coin flip with probability of p.
    """
    with tf.name_scope("coin_flip"):
        return tf.reshape(tf.less(tf.random_uniform([1],0, 1.0), p), [])

def return_shape(img):
    """
    Return shape of tensor, same style as numpy.
    """
    shape = tf.shape(img)
    x = shape[0]
    y = shape[1]
    z = shape[2]
    return x, y, z

def expend(img, marge):
    """
    Expend image by mirroring by a size of margin.
    """
    with tf.name_scope("expend"):    
        border_left = tf.image.flip_left_right(img[:, :marge])
        border_right= tf.image.flip_left_right(img[:, -marge:])
        width_ok = tf.concat([border_left, img, border_right], 1)
        height, width, depth = return_shape(width_ok)
        border_low = tf.image.flip_up_down(width_ok[:marge, :])
        border_up  = tf.image.flip_up_down(width_ok[-marge:, :])
        final_img = tf.concat([border_low, width_ok, border_up], 0)
        return final_img

def rotate_rgb(img, angles, marge):
    """
    Expend if necessary the image and rotate it
    """
    with tf.name_scope("rotate_rgb"):
        ext_img = expend(img, marge)
        height, width, depth = return_shape(ext_img)
        rot_img = tf.contrib.image.rotate(ext_img, angles, interpolation='BILINEAR')
        reslice = rot_img[marge:-marge, marge:-marge]
        return reslice

def rotate_label(img, angles, marge):
    with tf.name_scope("rotate_lbl"):
        ext_img = expend(img, marge)
        height, width, depth = return_shape(ext_img)
        rot_img = tf.contrib.image.rotate(ext_img, angles, interpolation='NEAREST')
        reslice = rot_img[marge:-marge, marge:-marge]
        return reslice

def flip_left_right(ss, coin):
    with tf.name_scope("flip_left_right"):   
        return tf.cond(coin,
                    lambda: tf.image.flip_left_right(ss),
                    lambda: ss)

def flip_up_down(ss, coin):
    with tf.name_scope("flip_up_down"):   
        return tf.cond(coin,
                    lambda: tf.image.flip_up_down(ss),
                    lambda: ss)

def random_flip(image, label, p):
    with tf.name_scope("random_flip"):
        coin_up_down = coin_flip(p)
        image = flip_up_down(image, coin_up_down)
        label = flip_up_down(label, coin_up_down)
        coin_left_right = coin_flip(p)
        image = flip_left_right(image, coin_left_right)
        label = flip_left_right(label, coin_left_right)
        return image, label

def random_rotate(image, label, p):
    with tf.name_scope("random_rotate"):
        coin_rotate = coin_flip(p)
        angle_rad = 0.5*PI
        angles = tf.random_uniform([1], 0, angle_rad)
        h_i, w_i, d_i = 396, 396, 3
        marge_rgb = int(max(h_i, w_i) * (2 - 1.414213) / 1.414213)
        image = tf.cond(coin_rotate, lambda: rotate_rgb(image, angles, marge_rgb)
                                   , lambda: image)
        h_l, w_l, d_l = 212, 212, 1
        marge_l = int(max(h_l, w_l) * (2 - 1.414213) / 1.414213)
        label = tf.cond(coin_rotate, lambda: rotate_label(label, angles, marge_l)
                                   , lambda: label)
        return image, label

def gaussian_kernel(kernlen=21, nsig=3, channels=1, sigma=1.):
    interval = (2 * nsig + 1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x, scale=sigma))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    kernel_mat = np.repeat(out_filter, channels, axis = 2)
    var = tf.constant(kernel_mat.flatten(), shape=kernel_mat.shape, dtype=tf.float32)
    return var


def gaussian_filter(image, nsig=3, channels=1., sigma=1.):
    kernel = gaussian_kernel(nsig=nsig, channels=channels, sigma=sigma)
    height, width, depth = return_shape(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.nn.depthwise_conv2d(image, kernel, strides=[1,1,1,1], padding='SAME')
    out = tf.squeeze(image)
    conv_image = tf.reshape(image, shape=[height, width, channels])
    return conv_image


def blur(image, sigma, channels=1):
    with tf.name_scope("blur"):
        out = gaussian_filter(image, nsig=3, channels=channels, sigma=sigma)
        return out


def random_blur(image, label, p, channels=3):
    with tf.name_scope("random_blur"):
        coin_blur = coin_flip(p)
        start, end = 0, 3
        sigma_list = [0.1,0.2,0.3]
        elems = tf.convert_to_tensor(sigma_list)
        samples = tf.multinomial(tf.log([[0.60, 0.30, 0.10]]), 1)
        nsig = tf.cast(samples[0][0], tf.int32)
        sigma = elems[nsig]
        for i in range(start, end):
            sigma_try = tf.constant([i])
            check = tf.reshape(tf.equal(nsig, sigma_try), [])
            apply_f = tf.logical_and(coin_blur,check)
            image = tf.cond(apply_f, lambda: blur(image, sigma_list[i], channels=channels)
                                   , lambda: image)
        return image, label

def change_brightness(image, delta, channels=3):
    with tf.name_scope("change_brightness"):
        x, y, z = return_shape(image)
        if channels == 4:
            channel4 = image[:,:,3]
            image = image[:,:,0:3]
        hsv = tf.image.rgb_to_hsv(image)
        first_channel = hsv[:,:,0:2]
        hsv2 = hsv[:,:,2]
        num = tf.multiply(delta, 255.)
        hsv2_aug = tf.clip_by_value(hsv2 + num, 0., 255.)
        hsv2_exp = tf.expand_dims(hsv2_aug, 2)
        hsv_new = tf.concat([first_channel, hsv2_exp], 2)
        new_rgb = tf.image.hsv_to_rgb(hsv_new)
        if channels == 4:
            new_rgb = tf.concat([new_rgb, tf.expand_dims(channel4, -1)], axis=2)
        new_rgb = tf.reshape(new_rgb, [x, y, channels])
        return new_rgb

def random_brightness(image, label, p, channels=3):
    with tf.name_scope("random_brightness"):
        coin_hue = coin_flip(p)
        delta = tf.truncated_normal([1], mean=0.0, stddev=0.1,
                                  dtype=tf.float32)
        image = tf.cond(coin_hue, lambda: change_brightness(image, delta, channels)
                                , lambda: image)
        return image, label

def generate(img, alpha_affine):
    x, y, z = return_shape(img)
    cent = tf.stack([x, y]) / 2
    sq_size = tf.minimum(x, y) / 3
    second = tf.stack([x / 2 + sq_size, y / 2 - sq_size])
    pts1 = tf.cast(tf.stack([cent + sq_size, second
                                   , cent - sq_size]), tf.float32)
    pts2 = pts1 + tf.reshape(tf.random_uniform([6], -alpha_affine, alpha_affine), [3,2])
    transform_matrix = get_affine_transform(pts1, pts2)
    return transform_matrix

def map_coordinates(img, indices, interpolation="BILINEAR", channels=3):
    x, y, z = return_shape(img)
    img = tf.expand_dims(img, 0)
    indices = tf.reshape(indices, [1, x, y, 2])
    if channels == 3:
        img_0 = tf.contrib.resampler.resampler(tf.expand_dims(img[:,:,:,0], -1), indices)
        img_1 = tf.contrib.resampler.resampler(tf.expand_dims(img[:,:,:,1], -1), indices)
        img_2 = tf.contrib.resampler.resampler(tf.expand_dims(img[:,:,:,2], -1), indices)
        res = tf.squeeze(tf.concat([img_0, img_1, img_2], axis=3))
    else:
        # image has to be 255
        maximum = tf.reduce_max(img)
        scale = tf.equal(maximum, tf.constant(255.))
        img = tf.cond(scale, lambda: img, lambda: img * 255.)
        res = tf.contrib.resampler.resampler(img, indices)
    if interpolation == "NEAREST":

        res = res / 255.
        res = tf.round(res)
        res = tf.squeeze(res, [0])
    return res

def elastic_deformation(image, annotation, alpha, alpha_affine, sigma, next_):
    with tf.name_scope("elastic_deformation"):
        image = expend(image, next)
        annotation = expend(annotation, next)
        x, y, _ = return_shape(image)
        x_int = tf.cast(x, dtype="float32")
        alpha = tf.multiply(x_int, alpha)
        alpha_affine = tf.multiply(x_int, alpha_affine)
        transform_m = generate(image, alpha_affine)
        dx = tf.squeeze(gaussian_filter(tf.random_uniform([x, y, 1], -1, 1), sigma=sigma, channels=1)) * alpha
        dy = tf.squeeze(gaussian_filter(tf.random_uniform([x, y, 1], -1, 1), sigma=sigma, channels=1)) * alpha
        xx, yy = tf.meshgrid(tf.range(x), tf.range(y), indexing='ij')
        xx = tf.cast(xx, tf.float32)
        yy = tf.cast(yy, tf.float32)
        indices = tf.concat([tf.reshape(yy + dy, [-1, 1]), tf.reshape(xx + dx, [-1, 1])], axis=1)

        #apply on img
        image_warp = warp_affine(image, transform_m, interpolation="BILINEAR")
        image_warp = tf.cast(image_warp, tf.float32)
        image_ind = map_coordinates(image_warp, indices)
        #apply on lbl
        lbl_warp = warp_affine(annotation, transform_m, interpolation="NEAREST")
        lbl_warp = tf.cast(lbl_warp, tf.float32)
        lbl_ind = map_coordinates(lbl_warp, indices, interpolation="NEAREST", channels=1)

        #so that the depth is known
        image_ind = tf.reshape(image_ind, [x, y, 3])

        return image_ind[next_:-next_, next_:-next_], lbl_ind[next_:-next_, next_:-next_]

def identity(image, annotation):
    """
    Applies identity to the image and annotation
    """
    return image, annotation
 
def get_affine_transform(src, dest):
    b = tf.reshape(dest, [6])
    L1 = tf.stack([src[0, 0], src[0, 1], 1., 0., 0., 0.])
    L3 = tf.stack([src[1, 0], src[1, 1], 1., 0., 0., 0.])
    L5 = tf.stack([src[2, 0], src[2, 1], 1., 0., 0., 0.])
    L2 = tf.stack([0., 0., 0., src[0, 0], src[0, 1], 1.])
    L4 = tf.stack([0., 0., 0., src[1, 0], src[1, 1], 1.])
    L6 = tf.stack([0., 0., 0., src[2, 0], src[2, 1], 1.])
    A = tf.matrix_inverse(tf.stack([L1, L2, L3, L4, L5, L6]))
    M = tf.matmul(A, tf.expand_dims(b, -1)) 
    return  M

def warp_affine(src, M, interpolation="NEAREST"):
    trans = tf.concat([tf.squeeze(M), tf.constant([0., 0.])], axis=0)
    return tf.contrib.image.transform(src, trans, interpolation=interpolation)  

def random_elastic_deformation(image, annotation, p,
                               alpha=1, 
                               alpha_affine=1,
                               sigma=1,
                               next_=50):
    with tf.name_scope("random_elastic_deformation"):
        coin_ela = coin_flip(p)
        image, annotation = tf.cond(coin_ela, lambda: elastic_deformation(image, annotation, alpha, alpha_affine, sigma, next_)
                                    , lambda: identity(image, annotation))
        return image, annotation

def augment_data(image_f, annotation_f, channels=3):
    with tf.name_scope("data_augmentation"):
        # ElasticDeformation are very slow..
        image_f, annotation_f = random_flip(image_f, annotation_f, 0.5)    
        image_f, annotation_f = random_rotate(image_f, annotation_f, 0.2)  
        image_f, annotation_f = random_blur(image_f, annotation_f, 0.2, channels)
        image_f, annotation_f = random_brightness(image_f, annotation_f, 0.2, channels)
        #image_f, annotation_f = random_elastic_deformation(image_f, annotation_f, 
        #                                                    0.5, 0.06, 0.12, 1.1)
        return image_f, annotation_f


def _parse_function(example_proto, channels=3, height=212, width=212, displacement=0, augment=True):
    """
    function to for deserializing and having the right shape, augmentation, etc..
    """
    dic_features = {'height_img': tf.FixedLenFeature([], tf.int64),
                    'width_img': tf.FixedLenFeature([], tf.int64),
                    'height_mask': tf.FixedLenFeature([], tf.int64),
                    'width_mask': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'mask_raw': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(example_proto, dic_features)
    height_img = tf.cast(features['height_img'], tf.int32)
    width_img = tf.cast(features['width_img'], tf.int32)

    height_mask = tf.cast(features['height_mask'], tf.int32)
    width_mask = tf.cast(features['width_mask'], tf.int32)

    const_img_height = height + 2*displacement
    const_img_width = width + 2*displacement

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)        
        
    image_shape = tf.stack([height_img, width_img, channels])
    annotation_shape = tf.stack([height_mask, width_mask, 1])

    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)

    image_f = tf.cast(image, tf.float32)
    annotation_f = tf.cast(annotation, tf.float32)
    if augment and displacement:
        img_a, lab_a = augment_data(image_f, annotation_f, channels)
        lab_a = lab_a[displacement:-displacement, displacement:-displacement]
    elif augment:
        img_a, lab_a = augment_data(image_f, annotation_f, channels)
    else:
        img_a, lab_a = image_f, annotation_f
    img_a = tf.image.resize_image_with_crop_or_pad(image=img_a,
                                                   target_height=const_img_height,
                                                   target_width=const_img_width)
    
    lab_a = tf.image.resize_image_with_crop_or_pad(image=lab_a,
                                                   target_height=height,
                                                   target_width=width)
    return img_a, lab_a

def read_and_decode(filename_queue, image_height, image_width,
                    batch_size, num_parallel_batches, train=True, channels=3, 
                    displacement=0, buffers=10, shuffle_buffer=10, seed=None):
    """
    Read and decode data from a tfrecord to a tensorflow queue
    """
    dataset = tf.data.TFRecordDataset(filename_queue)
    def f_parse(x):
        """
        parse function for training. Implies augmentation.
        """
        return _parse_function(x, channels, image_height,
                               image_width, displacement=displacement)
    def not_f_parse(x):
        """
        parse function for training. Implies no augmentation.
        """
        return _parse_function(x, channels, image_height, 
                               image_width, augment=False, 
                               displacement=displacement)
    if train:
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=f_parse,
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches
            )
        )
    else:
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=not_f_parse,
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches
            )
        )

    dataset = dataset.prefetch(buffer_size=buffers) 
    #dataset = dataset.cache()
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(
            shuffle_buffer,
            count=None,
            seed=seed
        )
    )

    iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    return iterator.initializer, iterator