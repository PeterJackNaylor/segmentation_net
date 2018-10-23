#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" metrics to feed in to the training model.


In this module we implement metric objects that will compute metrics
in a given a tensorflow graph.

"""

from abc import ABCMeta, abstractmethod

import tensorflow as tf


class MetricsAbstract:
    """
    Mother abstract class for the following templates
    """
    def __init__(self, label, prediction, name="MotherClass"):
        self.name = name
        self.tensor_out = None
        with tf.name_scope(name):
            self.computation_graph(label, prediction)
    __metaclass__ = ABCMeta
    @abstractmethod
    def computation_graph(self, label, prediction):
        """
        Defines the computation graph needed to 
        compute the desired metrics.
        """
        pass
    def return_name(self):
        """
        Return name of metric.
        """
        return self.name
    def return_tensor(self):
        """
        Return appropriate tensor.
        """
        return self.tensor_out
    def return_summary(self):
        """
        Return appropriate summary for tensor
        """
        return tf.summary.scalar(self.name, self.tensor_out)

def binary_confusion(lbl, pred):
    """
    Returns tp, tn, fp, fn
    """
    true_p = tf.count_nonzero(pred * lbl)
    true_n = tf.count_nonzero((pred - 1) * (lbl - 1))
    false_p = tf.count_nonzero(pred * (lbl - 1))
    false_n = tf.count_nonzero((pred - 1) * lbl)
    return true_p, true_n, false_p, false_n

class Accuracy(MetricsAbstract):
    """
    Accuracy metric
    """
    def __init__(self, prediction, label):
        super(Accuracy, self).__init__(label, prediction, "accuracy")

    def computation_graph(self, label, prediction):
        """
        Defining the computation graph.
        """
        correct_prediction = tf.cast(tf.equal(label, prediction), tf.float32)
        self.tensor_out = tf.reduce_mean(correct_prediction)

class Precision(MetricsAbstract):
    """
    Precision metric
    """
    def __init__(self, prediction, label):
        super(Precision, self).__init__(label, prediction, "precision")

    def computation_graph(self, label, prediction):
        true_p, _, false_p, _ = binary_confusion(label, prediction)
        precision = tf.divide(true_p, tf.add(true_p, false_p))
        precision = tf.where(tf.is_nan(precision), tf.zeros_like(precision), precision)
        self.tensor_out = tf.divide(true_p, tf.add(true_p, false_p))

class Recall(MetricsAbstract):
    """
    Recall metric
    """
    def __init__(self, prediction, label):
        super(Recall, self).__init__(label, prediction, "recall")

    def computation_graph(self, label, prediction):
        true_p, _, _, false_n = binary_confusion(label, prediction)
        recall = tf.divide(true_p, tf.add(true_p, false_n))
        recall = tf.where(tf.is_nan(recall), tf.zeros_like(recall), recall)
        self.tensor_out = recall

class F1_score(MetricsAbstract):
    """
    F1 metric
    """
    def __init__(self, label, prediction):
        super(F1_score, self).__init__(label, prediction, "f1_score")

    def computation_graph(self, prediction, label):
        true_p, _, false_p, false_n = binary_confusion(label, prediction)
        precision = tf.divide(true_p, tf.add(true_p, false_p))
        recall = tf.divide(true_p, tf.add(true_p, false_n))
        num = tf.multiply(precision, recall)
        dem = tf.add(precision, recall)
        f1 = tf.scalar_mul(2, tf.divide(num, dem))
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        self.tensor_out = f1

class Performance(MetricsAbstract):
    """
    Performance metric, defined as the mean between true positive rate and
    the true negative rate.
    """
    def __init__(self, prediction, label):
        super(Performance, self).__init__(label, prediction, "performance")

    def computation_graph(self, label, prediction):
        true_p, true_n, false_p, false_n = binary_confusion(label, prediction)
        precision = tf.divide(true_p, tf.add(true_p, false_p))
        negative_precision = tf.divide(true_n, tf.add(true_n, false_n))
        self.tensor_out = tf.divide(tf.add(precision, negative_precision), 2)

class TFMetricsHandler:

    def __init__(self, label, prediction, list_metrics):
        self.summaries_list = []
        self.tensors_name_list = []
        self.tensors_out_list = []

        for metric in list_metrics:
            met_obj = metric(label, prediction)
            self.summaries_list.append(met_obj.return_summary())
            self.tensors_name_list.append(met_obj.return_name())
            self.tensors_out_list.append(met_obj.return_tensor())

    def length(self):
        return len(self.tensors_out_list)

    def summaries(self):
        return self.summaries_list

    def tensors_out(self):
        return self.tensors_out_list

    def tensors_name(self):
        return self.tensors_name_list

BINARY_BUNDLE = [Accuracy, Recall, Precision, F1_score, Performance]

