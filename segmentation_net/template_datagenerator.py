#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Defining template class for data generator objects used for this 
    package. We implement a mother abstract class that is DataGeneratorTemplate
    that has to have two methods provided: give_length and next.

    We also define ExampleDatagen where we split our images into 4 and feed them
    to the next method.

    We also define ExampleUNetDatagen where we split our images into 4, however,
    we expend the image and take bigger images (as in this situation, one feed an
    image of size 396 and gets a 212 label image).

    We also define ExampleDistDG that return distance maps instead of binary maps.

"""

from operator import add 

from abc import ABCMeta, abstractmethod
from glob import glob
import numpy as np
from skimage.io import imread
import skimage.measure
from scipy.ndimage.morphology import distance_transform_cdt
from tqdm import tqdm

from .utils import expend

class DataGeneratorTemplate:
    """
    Mother abstract class for the following templates
    """
    __metaclass__ = ABCMeta
    @abstractmethod
    def give_length(self):
        """
        Return length of data set
        """
        pass
    @abstractmethod
    def next(self):
        """
        Give next elements of data.
        """
        pass

class ExampleDatagen(DataGeneratorTemplate):
    """
    For this example, we load everything in memory
    as the data is not expected to be big and we divide each image in 4.
    We expect an organisation as follow:
        RGB: folder/Slide_id/image.png
             folder/GT_id/image.png
    """
    def __init__(self, path, verbose=False):
        files = glob(path)
        self.length = len(files) * 4
        self.indices = [(f, i) for i in range(4) for f in files]
        if verbose:
            tqdm.write("Setting up DG.. This may take a while..")
            self.dic = {(f, i): self.create_couple(f, i) for f, i in tqdm(self.indices)}
        else:
            self.dic = {(f, i): self.create_couple(f, i) for f, i in self.indices}
        self.current_iter = 0

    def load_mask(self, image_name):
        """
        Way of loading mask images
        """
        mask_name = image_name.replace('Slide', 'GT')
        mask = imread(mask_name)
        mask[mask > 0] = 1
        return mask

    def load_img(self, image_name):
        """
        Way of loading the rgb images.
        """
        return imread(image_name)[:, :, 0:3]

    def give_length(self):
        """
        Return length of the data set, it could be more elaborate..
        """
        return self.length

    def create_couple(self, image_name, integer):
        """
        Bundle to return to the user.
        """
        return (self.quarter_image(self.load_img(image_name), integer), 
                self.quarter_image(self.load_mask(image_name), integer))

    def next_iter(self, iters):
        """
        Counter that cycles each time it gets to the size of the data
        """
        if (iters + 1) == self.give_length():
            res = 0
        else:
            res = iters + 1
        return res

    def next(self):
        """
        Give next elements of data.
        """
        img, anno = self.dic[self.indices[self.current_iter]]
        self.current_iter = self.next_iter(self.current_iter)
        return img, anno

    def quarter_image(self, image, integer):
        """
        Split images in quarters dependings of integer
        """
        if integer == 0:
            x_b, y_b = 0, 0
            x_e, y_e = 256, 256
        elif integer == 1:
            x_b, y_b = 0, 256
            x_e, y_e = 256, 512
        elif integer == 2:
            x_b, y_b = 256, 0
            x_e, y_e = 512, 256
        elif integer == 3:
            x_b, y_b = 256, 256
            x_e, y_e = 512, 512
        return image[x_b:x_e, y_b:y_e]

class ExampleUNetDatagen(ExampleDatagen):
    """
    Differences: UNet, in the situation of the UNet you feed in 
    a rgb image of size 396 and output a 212 (=396 - 184) image.
    However, due to the data augmentation one performs, it is easier
    to feed in same size images. 
    For the UNet, we choose to feed in rgb and mask samples of the
    same size of 396. The annotation is cropped on the fly. The reason
    is the following: without this, one can induce an error by expending
    the annotation so that you do need to fill in by blanks during augmentation.
    Like with rotation of size 60, the corners have to be filled.
    With the UNet and the slidding window you don't need to fill in as you have
    the original data source.


    For this example, we load everything in memory
    as the data is not expected to be big and we divide each image in 4.
    We expect an organisation as follow:
        RGB: folder/Slide_id/image.png
             folder/GT_id/image.png
    """
    def __init__(self, path, add=92, verbose=False):
        self.add = add
        ExampleDatagen.__init__(self, path, verbose)

    def expand(self, image):
        """
        Expand image of self.add on each side by mirroring
        """
        x_more, y_more = self.add, self.add
        new_img = expend(image, x_more, y_more)
        return new_img

    def create_couple(self, image_name, integer):
        """
        Bundle to return to the user.
        """
        img = self.expand(self.load_img(image_name))
        mask = self.expand(self.load_mask(image_name))

        return (self.quarter_image(img, integer), 
                self.quarter_image(mask, integer))

    def quarter_image(self, image, integer):
        """
        Split images in quarters dependings of integer
        with additionnal expanding
        """
        add2 = self.add * 2
        if integer == 0:
            x_b, y_b = 0, 0
            x_e, y_e = 256 + add2, 256 + add2
        elif integer == 1:
            x_b, y_b = 0, 256
            x_e, y_e = 256 + add2, 512 + add2
        elif integer == 2:
            x_b, y_b = 256, 0
            x_e, y_e = 512 + add2, 256 + add2
        elif integer == 3:
            x_b, y_b = 256, 256
            x_e, y_e = 512 + add2, 512 + add2
        return image[x_b:x_e, y_b:y_e]

def distance_without_normalise(bin_image):
    """
    Takes a binary image and returns a distance transform version of it.
    """
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype('uint8')
    return res


class ExampleDistDG(ExampleUNetDatagen):
    """
    Differences:: we transform the mask into a distance map
    during the loading.
    """
    def load_mask(self, image_name):
        """
        Way of loading mask images
        """
        mask_name = image_name.replace('Slide', 'GT')
        mask = skimage.measure.label(imread(mask_name))
        dist_mask = distance_without_normalise(mask)
        return dist_mask
