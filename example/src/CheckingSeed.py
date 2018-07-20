
# coding: utf-8

#  This notebook is a guide to use SegNetsTF package for segmentation.
#  We will compare the performances of a simple PangNet and a more complicated one U-net applied to the task of segmenting tissue in the WSI at low resolution.
#  The data is fairly scarse.
# 
# Table of Contents
# ======
# 1. [Input Data](#Input-Data)
# 2. [PangNet](#PangNet)
# 3. [U-net](#U-net)
# 4. [U-net + BN](#U-net and batch normalization)
# 
# # Input Data
# 
# We will have a quick look at the data. Helper functions are provided by the HelperDemo script where we implement some fairly trivial ploting functions but also the data generators needed for building all the necessary components for the training phase. In particular, the mean of the data, the records for the training and validation.

# In[1]:


import warnings
warnings.filterwarnings("ignore")

from glob import glob
import matplotlib.pylab as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(18,10)})
import numpy as np
import os
import pandas as pd
import tensorflow as tf

# In[2]:
import collections
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

import HelperDemo
from HelperDemo import AddContours, CheckOrCreate, plot_annotation, plot_overlay, plot_triplet


PATH_Train = './Data/Data_TissueSegmentation/Annotations/*.nii.gz'
PATH_Test = './Data/Data_TissueSegmentation/AnnotationsTest/*.nii.gz'

annotations = glob(PATH_Train)
annotations_test = glob(PATH_Test)


print('We have {} images in the training set.'.format(len(annotations)))
print('We have {} images in the test set.'.format(len(annotations_test)))



from HelperDemo import TissueGenerator

MeanFile = "./tmp/mean_file.npy"
TrainRec = "./tmp/train.tfrecord"
TestRec = "./tmp/test.tfrecord"


DG_Train = TissueGenerator(PATH_Train)
DG_Test  = TissueGenerator(PATH_Test, train=False)

# Images feed in to the neural networ

#for DG in [DG_Train, DG_Test]:
#    for i in range(0):
#        img, anno = DG.next()
#        plot_overlay(img, anno)


# # PangNet
# ## Training the model
# Hyper-parameters to set:
# - Number of epochs
# - Learning rate
# - Batch size
# - Weight decay
# - Archictecture
# - Data augmentations
# - Epochs
# - Stopping criteria
# - ...

# In[4]:


from segnet_tf import SegNet, Unet, UnetPadded, BatchNormedUnet
from segnet_tf import create_tfrecord, compute_mean
from HelperDemo import CheckOrCreate
CheckOrCreate('./tmp')


# In[5]:


## Setting up the model for the next parts

MeanFile = "./tmp/mean_file.npy"
TrainRec = "./tmp/train.tfrecord"
TestRec = "./tmp/test.tfrecord"

mean_array = compute_mean(MeanFile, [DG_Train, DG_Test])
# create_tfrecord(TrainRec, [DG_Train])
# create_tfrecord(TestRec, [DG_Test])


# In[6]:


lr = 0.001
wd = 0.0005
log = '{}__{}'.format(lr, wd)
model = SegNet(image_size=(256, 256), log=log, verbose=2, seed=42)
val_check1 = [model.sess.run(val).flatten() for val in model.training_variables]

dic = model.train(TrainRec, TestRec, learning_rate=lr,
                  lr_procedure="2epoch", weight_decay=0.0005, 
                  batch_size=4, decay_ema=0.9999, k=0.96, 
                  n_epochs=40, early_stopping=2, 
                  mean_array=mean_array, loss_func=tf.nn.l2_loss, 
                  verbose=1, save_weights=True, num_parallele_batch=8,
                  log=log, restore=False)
val_check2 = [model.sess.run(val).flatten() for val in model.training_variables]
model.sess.close()
#print(dic["test"].tail(1))


lr = 0.001
wd = 0.0005
log = "{}__{}_repeat".format(lr, wd)
model = SegNet(image_size=(256, 256), log=log, verbose=0, seed=42)
val_check11 = [model.sess.run(val).flatten() for val in model.training_variables]


dic2 = model.train(TrainRec, TestRec, learning_rate=lr,
                  lr_procedure="2epoch", weight_decay=0.0005, 
                  batch_size=4, decay_ema=0.9999, k=0.96, 
                  n_epochs=40, early_stopping=4, 
                  mean_array=mean_array, loss_func=tf.nn.l2_loss, 
                  verbose=1, save_weights=True, num_parallele_batch=8,
                  log=log, restore=False)
val_check22 = [model.sess.run(val).flatten() for val in model.training_variables]
model.sess.close()
val_check1 = [item for sublist in val_check1 for item in sublist]
val_check11 = [item for sublist in val_check11 for item in sublist]
val_check2 = [item for sublist in val_check2 for item in sublist]
val_check22 = [item for sublist in val_check22 for item in sublist]
print(compare(val_check1, val_check11))
print(compare(val_check2, val_check22))

#print(dic2["test"].tail(1))
