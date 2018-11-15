#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package mother class is defined here: SegNet


In this module we implement a object defined as SegNet.
SegNet is a generic class from which we derive many
of the more complex networks. Pang-net, U-net, D-net,
and possibly many others. SegmentationNet is the core of many 
of the structures used and implements many basic functions.
Such as a convolutional layer, ordering of the initialization
procedure and creation of the graph needed by tensorflow.

This module is not intented to be run as but can be.
It implements PangNet

"""

import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from . import utils_tensorflow as ut
from .data_read_decode import read_and_decode
from .net_utils import ScoreRecorder
from .tensorflow_metrics import TFMetricsHandler, metric_bundles_function
from .utils import check_or_create, element_wise, merge_dictionnary

MAX_INT = sys.maxsize

def verbose_range(beg, end, word, verbose, verbose_thresh):
    """
    If verbose, use tqdm to take care of estimating end of training.
    """
    returned_range = None
    if verbose > verbose_thresh:
        returned_range = trange(beg, end, desc=word)
    else:
        returned_range = range(beg, end)
    return returned_range

class SegmentationNet:

    def evaluation_graph(self, list_metrics, verbose=0):
        """
        Graph optimization part, here we define the loss and how the model is evaluated
        """

        with tf.name_scope('evaluation'):
            self.predictions = tf.argmax(self.last, axis=3, output_type=tf.int32)
            with tf.name_scope('loss'):
                softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
                lbl_int32 = tf.cast(self.label_node, tf.int32)
                labels = tf.squeeze(lbl_int32, squeeze_dims=[3])
                self.loss = tf.reduce_mean(softmax(logits=self.last,
                                                   labels=labels, name="entropy"))
            if self.tensorboard:
                loss_sum = tf.summary.scalar("entropy", self.loss)
                self.additionnal_summaries.append(loss_sum)
                # Disabled as we need several steps averaged
                # self.test_summaries.append(loss_sum)
            if list_metrics:
                self.tf_compute_metrics(labels, self.predictions, list_metrics)
        
        #tf.global_variables_initializer().run()
        if verbose > 1:
            tqdm.write('computational graph initialised')

    def init_architecture(self, verbose):
        """
        Initialises variables for the graph
        """
        strides = [1, 1, 1, 1]

        self.conv1 = self.conv_layer_f(self.input_node,
                                       self.num_channels,
                                       8, 3, strides=strides, 
                                       scope_name="conv1/")
        self.conv2 = self.conv_layer_f(self.conv1, 8,
                                       8, 3, strides=strides,
                                       scope_name="conv2/")
        self.conv3 = self.conv_layer_f(self.conv2, 8,
                                       8, 3, strides=strides,
                                       scope_name="conv3/")
        self.logit = self.conv_layer_f(self.conv3, 8,
                                       self.num_labels, 1, 
                                       strides=strides,
                                       scope_name="logit/")

        with tf.name_scope('final_layers'): 
            self.probability = tf.nn.softmax(self.logit, axis=-1)
            self.last = self.logit
        if verbose > 1:
            tqdm.write('model variables initialised')

    def predict_list(self, list_rgb, mean=None, label=None):
        """
        Predict from list of rgb files.
        """
        if label is None:
            label = [None for el in list_rgb]
        dic_res = {}
        for i, rgb in enumerate(list_rgb):
            dic_out = self.predict(rgb, mean=mean, label=label[i])
            dic_res = merge_dictionnary(dic_res, dic_out)
        return dic_res

    def predict(self, rgb, label=None, mean=None):
        """
        It will predict and return a dictionnary of results
        """
        if mean is not None:
            rgb = rgb - mean

        tensor = np.expand_dims(rgb, 0)
        feed_dict = {self.input_node: tensor,
                     self.is_training: False}
        tensors_names = ["predictions", "probability"]
        tensors_to_get = [self.predictions, self.probability]
        if label is not None:
            lbl = np.expand_dims(np.expand_dims(label, 2), 0)
            feed_dict[self.label_node] = lbl
            tensors_add = [self.loss] + self.metric_handler.tensors_out()
            tensors_to_get = tensors_to_get + tensors_add
            names = ["loss"] + self.metric_handler.tensors_name()
            tensors_names = tensors_names + names
        tensors_out = self.sess.run(tensors_to_get,
                                    feed_dict=feed_dict)
        out_dic = {}
        for name, tens in zip(tensors_names, tensors_out):
            if len(tens.shape) == 0:
                out_dic[name] = tens
            else:
                out_dic[name] = tens[0]
        return out_dic

    def predict_test_record(self, test_record, mean_array=None):
        if mean_array is not None:
            self.setup_mean(mean_array)
            # TODO remove self.mean_array
        self.mean_array = mean_array
        self.init_queue_test(test_record, self.verbose)
        self.init_queue_node()
        with tf.name_scope('queue_assigning'):
            # Control the dependency to allow the flow thought the data queues
            assign_rgb_to_queue = tf.assign(self.inp_variable, self.image, validate_shape=False)
            assign_lbl_to_queue = tf.assign(self.lbl_variable, self.annotation, 
                                            validate_shape=False)
        with tf.control_dependencies([assign_rgb_to_queue, assign_lbl_to_queue]):
            self.evaluation_graph(self.list_metrics, verbose=0)
        self.setup_init()
        test_steps = ut.record_size(test_record)
        score = self.record_test_set(0, 0, test_steps, verbose=self.verbose, summaries=None,
                                     follow_tensorboard_self=False)
        return score

    def record_test_set(self, epoch_iteration, total, test_steps, verbose=0, summaries=None, 
                        follow_tensorboard_self=True):
        """
        Process the whole test set according to model definition
        """
        if follow_tensorboard_self:
            tensorboard_first = self.tensorboard
            tensorboard2 = self.tensorboard
        else:
            tensorboard_first = False
            tensorboard2 = False
        range_ = verbose_range(0, test_steps, "testing ",
                               verbose, 1)
        summed_tensors = [0. for el in range(self.metric_handler.length() + 1)]
        for _ in range_:

            tensors_out = self.record_iteration(epoch_iteration, total, 
                                                tensorboard=tensorboard_first,
                                                verbose=0, train=False, 
                                                summaries=summaries)
            summed_tensors = element_wise(summed_tensors, tensors_out)
            tensorboard_first = False
            summaries = None
        test_scores = [el / test_steps for el in summed_tensors]
        if tensorboard2:
            names = ["loss"] + self.metric_handler.tensors_name()
            ut.add_values_to_summary(test_scores, names, self.summary_test_writer,
                                     epoch_iteration, tag="evaluation")
        if verbose > 1:
            msg = "Test :"
            tqdm.write(msg)
            names = ["loss"] + self.metric_handler.tensors_name()
            msg = ""
            for value, name in zip(test_scores, names):
                msg += "{} : {:.5f}  ".format(name, value)
            tqdm.write(msg)
        return test_scores

    def record_iteration(self, epoch_iteration, total, tensorboard=True,
                         verbose=0, train=True, summaries=None):
        """
        Process a step with/without tensorboard or verbose in train
        or test setting.
        """
        feed_dict = {self.is_training: train}
        tensors = [self.loss] + self.metric_handler.tensors_out()
        if tensorboard:
            if train:
                summary_list = [self.merged_summaries]
            else:
                summary_list = [self.merged_summaries_test]
            if summaries is not None:
                summary_list = summaries
            tensors = tensors + summary_list

        tensors.append(self.image_bf_mean)
        tensors.append(self.annotation)
        tensors.append(self.label_int)
        tensors.append(self.label_node)
        tensors.append(self.predictions)
        tensors.append(self.last)

        tensors_out = self.sess.run(tensors, feed_dict=feed_dict)

        from skimage.io import imsave
        import os
        try:
            if train:
                parent_folder = "imagetrain_folder"
            else:
                parent_folder = "imagetest_folder"
            os.mkdir(parent_folder)
        except:
            pass
        folder = parent_folder + "/{}/".format(np.random.randint(1000000))
        try:
            os.mkdir(folder)
        except:
            pass
        imsave(folder + "last.png", (tensors_out[-1][0,:,:,0] * float(255) / tensors_out[-1][0].max()).astype("uint8"))
        print(folder, "max:", np.max(tensors_out[-1][0,:,:,0]), "min:", np.min(tensors_out[-1][0,:,:,0]))
        del tensors_out[-1]
        imsave(folder + "predictions.png", (tensors_out[-1][0,:,:,0] * 255).astype("uint8"))
        del tensors_out[-1]
        imsave(folder +"label_node.png", (tensors_out[-1][0,:,:,0] * float(255) / tensors_out[-1][0].max()).astype("uint8"))
        del tensors_out[-1]
        imsave(folder +"label_int.png", (tensors_out[-1][0,:,:,0] * 255).astype("uint8"))
        del tensors_out[-1]
        imsave(folder +"annotation.png", (tensors_out[-1][0,:,:,0] * float(255) / tensors_out[-1][0].max()).astype("uint8"))
        del tensors_out[-1]
        imsave(folder + "input_bf_mean.png", tensors_out[-1][0,92:-92,92:-92].astype("uint8"))
        del tensors_out[-1]

        if tensorboard:
            if train:
                self.summary_writer.add_summary(tensors_out[-1], epoch_iteration)
            else:
                self.summary_test_writer.add_summary(tensors_out[-1], epoch_iteration)
            del tensors_out[-1]

        if verbose > 1:
            i = datetime.now()
            msg = i.strftime('[%Y/%m/%d %H:%M:%S]: ')
            msg += '  Epoch {} / {}'.format(epoch_iteration, total)
            msg += "  training :" if train else "test :"
            tqdm.write(msg)
            names = ["loss"] + self.metric_handler.tensors_name()
            msg = ""
            for value, name in zip(tensors_out, names):
                msg += "{} : {:.5f}  ".format(name, value)
            tqdm.write(msg)
        return tensors_out

    def record_train_step(self, epoch_iteration, total, verbose):
        """
        Process training step according to model definition
        """
        return self.record_iteration(epoch_iteration, total, tensorboard=self.tensorboard,
                                     verbose=verbose, train=True)

    def setup_training(self, train_record, test_record=None,
                       learning_rate=0.001, lr_procedure="1epoch",
                       weight_decay=0.0005, batch_size=1, k=0.96, 
                       decay_ema=0.9999, loss_func=tf.nn.l2_loss,
                       early_stopping=5, steps_in_epoch=100, 
                       mean_array=None, num_parallele_batch=8,
                       verbose=0, log=None):
        """
        Setups the queues and the last layer for predictions and metric
        computations. it also inits everything needed.
        """
        # Get the input queues ready
        with tf.name_scope('input_data_from_queue'):
            self.setup_mean(mean_array)
            self.setup_input(train_record, test_record, 
                             batch_size, num_parallele_batch,
                             verbose)
            with tf.name_scope('queue_assigning'):
                # Control the dependency to allow the flow thought the data queues
                assign_rgb_to_queue = tf.assign(self.inp_variable, self.image, validate_shape=False)
                assign_lbl_to_queue = tf.assign(self.lbl_variable, self.annotation, 
                                                validate_shape=False)

            # To plug in the queue to the main graph

        with tf.control_dependencies([assign_rgb_to_queue, assign_lbl_to_queue]):
            with tf.name_scope("final_calibrations"):
                with tf.name_scope('learning_rate_scheduler'):
                    self.learning_rate_scheduler(learning_rate, k, lr_procedure,
                                                 steps_in_epoch)
                with tf.name_scope('regularization'):
                    self.regularize_model(loss_func, weight_decay)

                with tf.name_scope('optimization'):
                    self.optimization(self.training_variables)

                with tf.name_scope('exponential_moving_average'):
                    self.exponential_moving_average(self.training_variables, decay_ema)
            
            self.evaluation_graph(self.list_metrics, verbose=0)

            if self.tensorboard:
                self.setup_summary()

        self.setup_init()
        self.score_recorder = ScoreRecorder(self.saver, self.sess, 
                                            log, lag=early_stopping)

    def train(self, train_record, test_record=None,
              learning_rate=0.001, lr_procedure="1epoch",
              weight_decay=0.0005, batch_size=1,
              decay_ema=0.9999, k=0.96, n_epochs=10,
              early_stopping=3, mean_array=None, 
              loss_func=tf.nn.l2_loss, save_weights=True, 
              num_parallele_batch=8, log=None, 
              restore=False, track_variable="loss"):
        """
        Train the model
        restore allows to re-initializing the variable as log will be empty.
        """
        if log is None:
            log = self.log
        else:
            check_or_create(log)

        if early_stopping != 3:
            self.saver = self.saver_object(keep=early_stopping + 1, 
                                           log=log,
                                           restore=restore)

        steps_in_epoch = ut.record_size(train_record) // batch_size
        test_steps = ut.record_size(test_record) if test_record is not None else None
        max_steps = steps_in_epoch * n_epochs
        self.setup_training(train_record, test_record, learning_rate,
                            lr_procedure, weight_decay, batch_size, k,
                            decay_ema, loss_func, early_stopping, 
                            steps_in_epoch, mean_array, num_parallele_batch,
                            self.verbose, log)

        begin_iter = int(self.global_step.eval())
        begin_epoch = begin_iter // steps_in_epoch
        last_epoch = begin_epoch + n_epochs
        last_iter = max_steps + begin_iter
        range_ = verbose_range(begin_iter, last_iter, "training ",
                               self.verbose, 0)
        # import pdb; pdb.set_trace()
        for step in range_:
            self.sess.run(self.training_op)
            if (step - begin_epoch + 1) % steps_in_epoch == 0 and (step - begin_epoch) != 0:
                epoch_number = step // steps_in_epoch
                if save_weights:  
                    self.saver.save(self.sess, log + '/' + "model.ckpt", 
                                    global_step=epoch_number)
                names = ["loss"] + self.metric_handler.tensors_name()
                
                values_train = self.record_train_step(epoch_number, last_epoch, self.verbose)
                self.score_recorder.diggest(epoch_number, values_train, names)
                
                if test_record:
                    values_test = self.record_test_set(epoch_number, last_epoch, 
                                                       test_steps, self.verbose)

                    self.score_recorder.diggest(epoch_number, values_test, names, train=False)

                    if self.score_recorder.stop(track_variable):
                        if self.verbose > 0:
                            tqdm.write('stopping early')
                        break
        if test_record:
            self.score_recorder.save_best(track_variable, save_weights)

        return self.score_recorder.all_tables()
