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
    def __init__(self, name="MotherClass"):
        self.name = name
        self.tensor_out = None
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
    def unpack_graph(self, label, prediction):
        """
        Unpack tensorflow graph
        """
        with tf.name_scope(self.name):
            self.computation_graph(label, prediction)
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
    def __init__(self):
        super(Accuracy, self).__init__("accuracy")

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
    def __init__(self):
        super(Precision, self).__init__("precision")

    def computation_graph(self, label, prediction):
        true_p, _, false_p, _ = binary_confusion(label, prediction)
        precision = tf.divide(true_p, tf.add(true_p, false_p))
        precision = tf.where(tf.is_nan(precision), tf.zeros_like(precision), precision)
        self.tensor_out = tf.divide(true_p, tf.add(true_p, false_p))

class Recall(MetricsAbstract):
    """
    Recall metric
    """
    def __init__(self):
        super(Recall, self).__init__("recall")

    def computation_graph(self, label, prediction):
        true_p, _, _, false_n = binary_confusion(label, prediction)
        recall = tf.divide(true_p, tf.add(true_p, false_n))
        recall = tf.where(tf.is_nan(recall), tf.zeros_like(recall), recall)
        self.tensor_out = recall

class F1_score(MetricsAbstract):
    """
    F1 metric
    """
    def __init__(self):
        super(F1_score, self).__init__("f1_score")

    def computation_graph(self, label, prediction):
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
    def __init__(self):
        super(Performance, self).__init__("performance")

    def computation_graph(self, label, prediction):
        true_p, true_n, false_p, false_n = binary_confusion(label, prediction)
        precision = tf.divide(true_p, tf.add(true_p, false_p))
        negative_precision = tf.divide(true_n, tf.add(true_n, false_n))
        self.tensor_out = tf.divide(tf.add(precision, negative_precision), 2)

class MeanAccuracy(MetricsAbstract):
    """
    Mean accuracy metric, defined as the mean where each class is weighted the accordingly.
    """
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        super(MeanAccuracy, self).__init__("mean_accuracy")

    def computation_graph(self, label, prediction):

        label = tf.Print(label, [tf.shape(label), tf.shape(prediction), label])
        label = tf.argmax(label, axis=-1)
        prediction = tf.argmax(prediction, axis=-1)
        label = tf.reshape(label, [-1])
        prediction = tf.reshape(prediction, [-1])
        confusion = tf.confusion_matrix(labels=label, 
                                        predictions=prediction, 
                                        num_classes=self.num_classes)

        per_class = tf.linalg.diag_part(confusion)
        confusion_col_sum = tf.reduce_sum(confusion, 1)
        per_class = per_class / confusion_col_sum
        score = tf.reduce_mean(per_class)

        def f1():
            per_class_nan = tf.where(tf.is_nan(per_class), tf.zeros_like(per_class), per_class)
            # per_class_nan = tf.Print(per_class_nan, [], message='y_pred contains classes not in y_true')
            no_nan_score = tf.reduce_mean(per_class_nan)
            return no_nan_score
        def f2():
            return score

        self.tensors_out = tf.cond(tf.is_nan(score), f1, f2)

class TFMetricsHandler:

    def __init__(self, label, prediction, list_metrics):
        self.summaries_list = []
        self.tensors_name_list = []
        self.tensors_out_list = []

        for metric in list_metrics:
            met_obj = metric.unpack_graph(label, prediction)
            self.summaries_list.append(metric.return_summary())
            self.tensors_name_list.append(metric.return_name())
            self.tensors_out_list.append(metric.return_tensor())

    def length(self):
        return len(self.tensors_out_list)

    def summaries(self):
        return self.summaries_list

    def tensors_out(self):
        return self.tensors_out_list

    def tensors_name(self):
        return self.tensors_name_list

def metric_bundles_function(num_label):
    if num_label < 3:
        bundle = [Accuracy(), Recall(), Precision(), F1_score(), Performance()]
    else:
        bundle = [Accuracy()] # , MeanAccuracy(num_classes=num_label)]
    return bundle
