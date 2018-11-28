#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    In this module we define a helper object for training.
    It records most of training and validation metrics at each epoch.
    It defines an early stopping criteria.
"""

import os
from os.path import isfile
import numpy as np

import pandas as pd

def fill_table_line(data, index, dic):
    """
    Fills one line in specific table with two list
    values and their corresponding names.
    """
    for name, val in dic.items():
        if not np.isscalar(val):
            val = "NOT SCALAR"
        data.loc[index, name] = val
    return data 

def any_bigger_in_array(data, tracked_var, lag, eps, increase=True):
    """
    Checks on a pandas table if to early stop with respect to the variable_name
    """
    data = data.fillna(0)
    stop = False
    tracked_values = np.array(data[tracked_var])
    if not increase:
        tracked_values = tracked_values * -1
    if len(tracked_values) > lag - 1:
        lagged_value = tracked_values[-lag]
        after_lagged_values = tracked_values[-(lag-1):]
        time_to_stop = ((lagged_value - after_lagged_values) > -1*eps).all()
        stop = time_to_stop
    return stop

class ScoreRecorder(object):
    """
    Useful object to store training and test metrics
    during training.
    It can also perform early stopping.
    """
    def __init__(self, saver, session, log,
                 stop_early=True, lag=10, eps=10**-4):

        self.train_file_name = os.path.join(log, 'train_scores.csv')
        self.test_file_name = os.path.join(log, 'test_scores.csv')

        if isfile(self.train_file_name):
            self.data_train = pd.read_csv(self.train_file_name, index_col=0) 
        else:
            self.data_train = pd.DataFrame()

        if isfile(self.test_file_name):
            self.data_test = pd.read_csv(self.test_file_name, index_col=0) 
        else:
            self.data_test = pd.DataFrame()

        self.steps_before_allowed_to_check = lag
        self.track = None
        self.eps = eps
        self.lag = lag
        self.log = log
        self.stop_early = stop_early
        self.saver = saver
        self.sess = session
#        self.saver._max_to_keep = lag + 1

    def all_tables(self):
        """
        return all_tables
        """
        return {'train': self.data_train, 'test': self.data_test}

    def diggest(self, epoch_number, training_dic, train=True):
        """
        Collects and fills a table for two given list (names and values) 
        Do be used in the for loop training
        """
        if train:
            self.data_train = fill_table_line(self.data_train, epoch_number,
                                              training_dic)
            self.data_train.to_csv(self.train_file_name)
        else:
            self.data_test = fill_table_line(self.data_test, epoch_number,
                                             training_dic)
            self.data_test.to_csv(self.test_file_name)
    def find_best_epoch(self, tracking_variable, increase=True):
        """
        Find best epoch for model test set
        """
        data = self.data_test.fillna(0.)
        if increase:
            lagged_value = data.tail(self.lag + 1)[tracking_variable].idxmax()
        else:
            lagged_value = data.tail(self.lag + 1)[tracking_variable].idxmin()
        return lagged_value

    def save_best(self, tracking_variable, save_weights=True):
        """
        Save best model
        """
        increase = True
        if tracking_variable == "loss":
            increase = False
        best_epoch = self.find_best_epoch(tracking_variable, increase)
        self.saver.restore(self.sess, "{}/model.ckpt-{}".format(self.log, best_epoch + 1))
        last_step = self.data_test.index.max()
        if best_epoch != last_step:
            self.data_test.loc[last_step + 1] = self.data_test.loc[best_epoch]
            if save_weights:
                self.saver.save(self.sess, self.log + '/' + "model.ckpt", last_step + 1)

    def stop(self, tracking_variable, train_set=True):
        """
        If or not to perform early stopping for a given
        tracking variable on the test set.
        """
        self.steps_before_allowed_to_check -= 1
        if self.steps_before_allowed_to_check == 0:
            if self.stop_early:
                increase = True
                if tracking_variable == "loss":
                    increase = False
                if train_set:
                    stop = any_bigger_in_array(self.data_train, tracking_variable, 
                                               self.lag, self.eps, increase)
                else:
                    stop = any_bigger_in_array(self.data_test, tracking_variable, 
                                               self.lag, self.eps, increase)                  
            else:
                stop = False
        else:
            stop = False
        return stop
