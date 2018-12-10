#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package file tf_record

Segmentation_base_class ->  SegmentationInput -> SegmentationCompile -> 
SegmentationSummaries -> Segmentation_model_utils -> Segmentation_train

"""

from .segmentation_input import *
from ..tensorflow_metrics import TFMetricsHandler


class SegmentationCompile(SegmentationInput):

    def exponential_moving_average(self, optimizer, var_list, decay=0.9999):
        """
        Adding exponential moving average to increase performance.
        This aggregates parameters from different steps in order to have
        a more robust classifier.
        Args:
            optimizer: tensorflow object, usually an tensorflow optimizer
                       on which you wish to create a dependency. For instance,
                       each time we call the exponential moving average operation
                       we run the optimizer.

            var_list: list of variables on which we apply the exponential moving average.
            decay: float (default: 0.9999), value of the exponential decay to apply on the
                   variables.
        Returns:
            A training op that tries to solve the optimization problem
            while applying exponential moving average to the training weights.
        """
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        maintain_averages_op = ema.apply(var_list)

        # Create an op that will update the moving averages after each training
        # step.  This is what we will use in place of the usual training op.
        with tf.control_dependencies([optimizer]):
            training_op = tf.group(maintain_averages_op)
        return training_op

    def learning_rate_scheduler(self, learning_rate, k, lr_procedure,
                                steps_in_epoch):
        """

        Defines the learning rate scheduler to apply during training. 
        Args:
            learning_rate: float, Initial learning rate for the 
                           gradient descent update.
            k : float, value by which the learning rate decays every 
                update
            lr_procedure : string or int. If string, it has to be
                           in the shape of '{number}epoch' (or 'epoch/2') and it
                           will perfome an update of the learning rate
                           every number epochs (or everything half an epoch).
                           If it is an int, it will the learning rate decay will
                           be applyied every int steps.
            steps_in_epoch: int, if lr_procedure is an integer this value is ignored.
                            Otherwise, this parameter is useful to decline the number 
                            of iterations to complete an epoch.
        Returns:
            A tensorflow operation that will decay the learning rate.
        """
        if isinstance(lr_procedure, int):
            decay_step = float(lr_procedure)
        elif isinstance(lr_procedure, str):
            if lr_procedure == "epoch/2":
                decay_step = steps_in_epoch / 2
            else:
                num = int(lr_procedure[:-5])
                decay_step = float(num) * steps_in_epoch

        learning_rate_tf = tf.train.exponential_decay(learning_rate,
                                                      self.global_step,
                                                      decay_step,
                                                      k,
                                                      staircase=True)
        if self.tensorboard:
            s_lr = tf.summary.scalar("learning_rate", learning_rate_tf)
            self.additionnal_summaries.append(s_lr)

        return learning_rate_tf

    def optimization(self, learning_rate_tf, loss_tf, var_list):
        """
        Defines the optimizer that will be the main operation
        to run during training.

        Args:
            learning_rate_tf : float or tensorflow object returned
                               by the instance learning_rate_scheduler.
            loss_tf : tensorflow variable that is the function to solve.
            var_list : list of tensorflow variable on which to apply the
                       gradient descent optimization.
        Returns:
            A tensorflow object optimizer.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate_tf)
        grads = optimizer.compute_gradients(loss_tf, var_list=var_list)
        optimizer_obj = optimizer.apply_gradients(grads, global_step=self.global_step)     

        if self.tensorboard:
            for grad, var in grads:
                self.gradient_to_summarise.append((grad, var))  

        return optimizer_obj

    def penalize_with_regularization(self, loss_tf, var, loss_func, weight_decay):
        """
        Penalizes the given loss by adding to the loss a term that is the
        weight_decay * loss_func(var). This is one method to avoir overfitting
        by penalizing with respect to the magnitude of each parameter in the model.
        Args:
            loss_tf : tensorflow variable that you wish to penalize.
            var : tensorflow variable that you wish to penalize the loss by.
            loss_func : tensorflow loss function, like tf.nn.l2_loss
            weight_decay : float, scalar by which to multiply the loss term by.
        Returns:
            The modified loss function.
        """
        if var is not None:
            loss_tf = loss_tf + weight_decay * loss_func(var)
        return loss_tf

    def regularize_model(self, loss_tf, loss_func, weight_decay):
        """
        Penalizes the given loss by adding to the loss a term all the variables
        in self.var_to_reg such as loss = loss + sum_{var} weight_decay * loss_func(var).
        This is one method to avoir overfitting by penalizing with respect 
        to the magnitude of each parameter in the model.
        Args:
            loss_tf : tensorflow variable that you wish to penalize.
            loss_func : tensorflow loss function, like tf.nn.l2_loss
            weight_decay : float, scalar by which to multiply the loss term by.
        Returns:
            The modified loss function.
        """
        for var in self.var_to_reg:
            loss_tf = self.penalize_with_regularization(loss_tf, var, loss_func, weight_decay)
        return loss_tf

    def tf_compute_metrics(self, lbl, pred, list_metrics):
        """
        Builds the tensorflow graph between a label variable tensorflow and
        a prediction variable tensorflow. It returns the graph under the form
        of a python object from the TFMetricsHandler class. List_metrics defines
        the list of metrics to monitor.
        Args:
            lbl : tensorflow variable that is meant to be the label.
            pred : tensorflow variable that is meant to be the prediction.
            list_metrics : list of metric objects that will define the metric
                           handler.
        Returns:
            Python object metric handler.
        """
        metric_handler = TFMetricsHandler(lbl, pred, list_metrics)

        # if self.tensorboard:
        #     for __s in metric_handler.summaries():
        #         self.additionnal_summaries.append(__s)
                # Disabled as we need several steps averaged
                # self.test_summaries.append(__s)
        return metric_handler
