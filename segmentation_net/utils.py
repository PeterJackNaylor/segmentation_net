#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" utils packages for segnet package.

Helper functions for the package. These functions 
are not written in tensorflow.

"""

import os
from operator import add

import numpy as np

def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def flip_vertical(picture):
    """ 
    vertical flip
    takes an arbitrary image as entry
    """
    res = np.flip(picture, axis=1)
    return res


def flip_horizontal(picture):
    """
    horizontal flip
    takes an arbitrary image as entry
    """
    res = np.flip(picture, axis=0)
    return res

def expend(image, x_s, y_s):
    """
    Expend the image and mirror the border of size x and y
    """
    rows, cols = image.shape[0], image.shape[1]
    if len(image.shape) == 2:
        enlarged_image = np.zeros(shape=(rows + 2*y_s, cols + 2*x_s))
    else:
        enlarged_image = np.zeros(shape=(rows + 2*y_s, cols + 2*x_s, 3))

    enlarged_image[y_s:(y_s + rows), x_s:(x_s + cols)] = image
    # top part:
    enlarged_image[0:y_s, x_s:(x_s + cols)] = flip_horizontal(
        enlarged_image[y_s:(2*y_s), x_s:(x_s + cols)])
    # bottom part:
    enlarged_image[(y_s + rows):(2*y_s + rows), x_s:(x_s + cols)] = flip_horizontal(
        enlarged_image[rows:(y_s + rows), x_s:(x_s + cols)])
    # left part:
    enlarged_image[y_s:(y_s + rows), 0:x_s] = flip_vertical(
        enlarged_image[y_s:(y_s + rows), x_s:(2*x_s)])
    # right part:
    enlarged_image[y_s:(y_s + rows), (cols + x_s):(2*x_s + cols)] = flip_vertical(
        enlarged_image[y_s:(y_s + rows), cols:(cols + x_s)])
    # top left from left part:
    enlarged_image[0:y_s, 0:x_s] = flip_horizontal(
        enlarged_image[y_s:(2*y_s), 0:x_s])
    # top right from right part:
    enlarged_image[0:y_s, (x_s + cols):(2*x_s + cols)] = flip_horizontal(
        enlarged_image[y_s:(2*y_s), cols:(x_s + cols)])
    # bottom left from left part:
    enlarged_image[(y_s + rows):(2*y_s + rows), 0:x_s] = flip_horizontal(
        enlarged_image[rows:(y_s + rows), 0:x_s])
    # bottom right from right part
    enlarged_image[(y_s + rows):(2*y_s + rows), (x_s + cols):(2*x_s + cols)] = flip_horizontal(
        enlarged_image[rows:(y_s + rows), (x_s + cols):(2*x_s + cols)])
    enlarged_image = enlarged_image.astype('uint8')
    return enlarged_image

def element_wise(list1, list2):
    """
    Perform element wise sum between two lists.
    """
    return list(map(add, list1, list2))

def merge_dictionnary(dic1, dic2):
    """
    Merge two dictionaries by appending on the keys of the other
    """
    res = {}
    if not dic1:
        for key in dic2.keys():
            res[key] = [dic2[key]]
    else:
        for key in dic1.keys():
            dic1[key].append(dic2[key])
        res = dic1
    return res
