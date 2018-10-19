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
from .tensorflow_metrics import TFMetricsHandler, BINARY_BUNDLE
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
    """
    Generic object for create DNN models.
    This class instinciates all functions 
    needed for DNN operations.
    """
    def __init__(
            self,
            image_size=(224, 224),
            log="/tmp/net",
            num_channels=3,
            num_labels=2,
            tensorboard=True,
            seed=None,
            verbose=0,
            displacement=0,
            list_metrics=BINARY_BUNDLE):
        ## this doesn't work yet... 
        ## when the graph is initialized it is completly equal everywhere.
        ## https://github.com/tensorflow/tensorflow/issues/9171
        np.random.seed(seed)
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        # Parameters
        self.image_size = image_size
        self.log = log
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.tensorboard = tensorboard
        self.seed = seed
        self.displacement = displacement #Variable important for the init queues

        check_or_create(log)

        # Preparing inner variables
        self.sess = tf.InteractiveSession()
        self.sess.as_default()
        

        self.gradient_to_summarise = []
        self.test_summaries = []
        self.additionnal_summaries = []
        self.training_variables = []
        self.var_activation_to_summarise = []
        self.var_to_reg = []
        self.var_to_summarise = []

        # Preparing tensorflow variables
        with tf.name_scope('default_inputs'):
            self.global_step = tf.Variable(0., name='global_step', trainable=False)
            self.is_training = tf.placeholder_with_default(True, shape=[])

            input_shape = (None, None, None, self.num_channels)
            label_shape = (None, None, None, 1)
            x_inp, y_inp = image_size[0] + 2*displacement, image_size[1] + 2*displacement
            default_input = np.zeros(shape=(1, x_inp, y_inp, 3), dtype='float32')
            default_label = np.zeros(shape=(1, image_size[0], image_size[1], 1), dtype='float32')
            self.inp_variable = tf.Variable(default_input, validate_shape=False, 
                                            name="rgb_input")
            self.lbl_variable = tf.Variable(default_label, validate_shape=False, 
                                            name="lbl_input")
            self.input_node = tf.placeholder_with_default(self.inp_variable, shape=input_shape)
            self.label_node = tf.placeholder_with_default(self.lbl_variable, shape=label_shape)
        # Empty variables to be defined in sub methods
        # Usually these variables will become tensorflow
        # variables
        self.annotation = None
        self.data_init = None
        self.data_init_test = None
        self.image = None
        self.image_bf_mean = None
        self.image_placeholder = None
        self.learning_rate = None
        self.loss = None
        self.mean_tensor = None
        self.merged_summaries = None
        self.merged_summaries_test = None
        self.metric_handler = None
        self.predictions = None
        self.optimizer = None
        self.score_recorder = None
        self.summary_writer = None
        self.summary_test_writer = None
        self.test_iterator = None
        self.training_op = None
        self.train_iterator = None

        # Initializing models

        self.init_architecture(verbose)
        self.evaluation_graph(list_metrics, verbose)
        # self.init_training_graph()
        self.saver = self.saver_object(self.training_variables, 3, verbose, 
                                       self.log, restore=True)

    def add_summary_images(self):
        """
        Image summary to add to the summary
        TODO Does not work with the cond in the network flow...
        """
        input_s = tf.summary.image("input", self.input_node, max_outputs=4)
        label_s = tf.summary.image("label", self.label_node, max_outputs=4)
        pred = tf.expand_dims(tf.cast(self.predictions, tf.float32), dim=3)
        predi_s = tf.summary.image("pred", pred, max_outputs=4)
        for __s in [input_s, label_s, predi_s]:
            self.additionnal_summaries.append(__s)
            self.test_summaries.append(__s)

    def conv_layer_f(self, i_layer, s_input_channel, size_output_channel,
                     ks, strides=[1, 1, 1, 1], scope_name="conv", random_init=0.1, 
                     padding="SAME"):
        """
        Defining convolution layer
        """
        with tf.name_scope(scope_name):
            weights = self.weight_xavier(ks, s_input_channel, 
                                         size_output_channel)
            biases = ut.biases_const_f(random_init, size_output_channel)
            conv_layer = tf.nn.conv2d(i_layer, weights, 
                                      strides=strides, 
                                      padding=padding)
            conv_with_bias = tf.nn.bias_add(conv_layer, biases)
            act = self.non_activation_f(conv_with_bias)

            self.training_variables.append(weights)
            self.training_variables.append(biases)
            self.var_to_reg.append(weights)

            if self.tensorboard:
                self.var_to_summarise.append(weights)
                self.var_to_summarise.append(biases)
                self.var_to_summarise.append(conv_layer)
                self.var_activation_to_summarise.append(act)
            return act

    def drop_out(self, i_layer, keep_prob=0.5, scope="drop_out"):
        """
        Performs drop out layer and checks if the seed is set
        """
        if self.seed:
            seed = np.random.randint(MAX_INT)
        else: 
            seed = self.seed
        return ut.drop_out_layer(i_layer, keep_prob, scope, seed)

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
                loss_sum = tf.summary.scalar("entropy", self.loss)
            if self.tensorboard:
                self.additionnal_summaries.append(loss_sum)
                # Disabled as we need several steps averaged
                # self.test_summaries.append(loss_sum)
            if list_metrics:
                self.tf_compute_metrics(labels, self.predictions, list_metrics)
        
        #tf.global_variables_initializer().run()
        if verbose > 1:
            tqdm.write('computational graph initialised')

    def exponential_moving_average(self, var_list, decay=0.9999):
        """
        Adding exponential moving average to increase performance.
        This aggregates parameters from different steps in order to have
        a more robust classifier.
        """
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        maintain_averages_op = ema.apply(var_list)

        # Create an op that will update the moving averages after each training
        # step.  This is what we will use in place of the usual training op.
        with tf.control_dependencies([self.optimizer]):
            self.training_op = tf.group(maintain_averages_op)


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

    def init_queue(self, train, test, batch_size, num_parallele_batch, verbose):
        """
        New queues for coordinator
        """
        with tf.device('/cpu:0'):
            with tf.name_scope('training_queue'):
                out = read_and_decode(train, 
                                      self.image_size[0], 
                                      self.image_size[1],
                                      batch_size,
                                      num_parallele_batch,
                                      displacement=self.displacement,
                                      seed=self.seed)
                self.data_init, self.train_iterator = out
            if test:
                with tf.name_scope('test_queue'):
                    out = read_and_decode(test, 
                                          self.image_size[0], 
                                          self.image_size[1],
                                          1,
                                          1,
                                          train=False,
                                          displacement=self.displacement,
                                          buffers=1,
                                          shuffle_buffer=1,
                                          seed=self.seed)
                    self.data_init_test, self.test_iterator = out
        if verbose > 1:
            tqdm.write("queue initialized")

    def init_queue_node(self):
        """
        The input node can now come from the record or can be inputed
        via a feed dict (for testing for example)
        """
        def f_true():
            """
            The callable to be performed if pred is true.
            i.e. fetching data from the train queue
            """
            train_images, train_labels = self.train_iterator.get_next()
            return train_images, train_labels

        def f_false():
            """
            The callable to be performed if pred is false.
            i.e. fetching data from the test queue
            """
            test_images, test_labels = self.test_iterator.get_next()
            return test_images, test_labels

        with tf.name_scope('switch'):
            self.image_bf_mean, self.annotation = tf.cond(self.is_training, f_true, f_false)
        if self.mean_tensor is not None:
            self.image = self.image_bf_mean - self.mean_tensor
        else:
            self.image = self.image_bf_mean

    def initialize_all(self):
        """
        initialize all variables
        """
        self.sess.run(tf.global_variables_initializer())

    def learning_rate_scheduler(self, learning_rate, k, lr_procedure,
                                steps_in_epoch):
        """
        Defines the learning rate. 
        Parameters:
            - learning rate: initial value given to learning rate
            - decay value to multiply learning rate by
            - lr procedure are defined procedures. It can be int, 
            in this case the learning rate decayes every lr_procedure
            step. If it is string, it can be 'epoch/2' where it 
            automatically determines the right steps and decays the learning
            rate every half epoch. Otherwise it follows the convention
            '[int]epoch' where [int] is a integer. Like '10epoch' will decay
            the learning rate every 10 epoch.

        """
        if isinstance(lr_procedure, int):
            decay_step = float(lr_procedure)
        elif isinstance(lr_procedure, str):
            if lr_procedure == "epoch/2":
                decay_step = steps_in_epoch / 2
            else:
                num = int(lr_procedure[:-5])
                decay_step = float(num) * steps_in_epoch

        self.learning_rate = tf.train.exponential_decay(learning_rate,
                                                        self.global_step,
                                                        decay_step,
                                                        k,
                                                        staircase=True)
        if self.tensorboard:
            s_lr = tf.summary.scalar("learning_rate", self.learning_rate)
            self.additionnal_summaries.append(s_lr)

    def non_activation_f(self, i_layer):
        """
        Defining relu layer.
        The default is relu. Feel free to redefine non activation
        function for the standard convolution_
        """
        act = tf.nn.relu(i_layer)
        return act

    def optimization(self, var_list):
        """
        Defining the optimization method to solve the task
        """
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = optimizer.compute_gradients(self.loss, var_list=var_list)
        self.optimizer = optimizer.apply_gradients(grads, global_step=self.global_step)     

        if self.tensorboard:
            for grad, var in grads:
                self.gradient_to_summarise.append((grad, var))   

    def penalize_with_regularization(self, var, loss_func, weight_decay):
        """
        Combines loss with regularization loss
        """
        if var is not None:
            self.loss = self.loss + weight_decay*loss_func(var)

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

    def record_test_set(self, epoch_iteration, total, test_steps, verbose=0, summaries=None):
        """
        Process the whole test set according to model definition
        """
        tensorboard_first = self.tensorboard
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
        if self.tensorboard:
            names = ["loss"] + self.metric_handler.tensors_name()
            ut.add_values_to_summary(test_scores, names, self.summary_test_writer,
                                     epoch_iteration, tag="evaluation")
        if verbose > 1:
            tqdm.write('  epoch %d of %d' % (epoch_iteration, total))
            tqdm.write("test ::")
            names = ["loss"] + self.metric_handler.tensors_name()
            for value, name in zip(test_scores, names):
                tqdm.write("{} : {}".format(name, value))
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
        tensors_out = self.sess.run(tensors, feed_dict=feed_dict)
        
        if tensorboard:
            if train:
                self.summary_writer.add_summary(tensors_out[-1], epoch_iteration)
            else:
                self.summary_test_writer.add_summary(tensors_out[-1], epoch_iteration)
            del tensors_out[-1]

        if verbose > 1:
            tqdm.write('  epoch {} / {}'.format(epoch_iteration, total))
            word = "training ::" if train else "test ::"
            tqdm.write(word)
            names = ["loss"] + self.metric_handler.tensors_name()
            for value, name in zip(tensors_out, names):
                tqdm.write("{} : {}".format(name, value))
        return tensors_out

    def record_train_step(self, epoch_iteration, total, verbose):
        """
        Process training step according to model definition
        """
        return self.record_iteration(epoch_iteration, total, tensorboard=self.tensorboard,
                                     verbose=verbose, train=True)

    def regularize_model(self, loss_func, weight_decay):
        """
        Adds regularization to parameters of the model given LOSS_FUNC
        """
        for var in self.var_to_reg:
            self.penalize_with_regularization(var, loss_func, weight_decay)

    def restore(self, saver, log, verbose=0):
        """
        restore the model
        if no log files exit, it re initializes the architecture variables
        """
        ckpt = tf.train.get_checkpoint_state(log)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            if verbose:
                tqdm.write("model restored...")
        else:
            if verbose:
                tqdm.write("values have been initialized")
            self.initialize_all()

    def saver_object(self, list_variables, keep=3, verbose=0, log=None, restore=False):
        """
        Defining the saver, it will load if possible.
        """
        if verbose > 1:
            tqdm.write("setting up saver...")
        saver = tf.train.Saver(list_variables, max_to_keep=keep)
        if log and restore:
            self.restore(saver, log, verbose)
        return saver

    def init_uninit(self, extra_variables):
        ut.initialize_uninitialized_vars(self.sess, {self.is_training: False},
                                         extra_variables)

    def setup_init(self):
        """
        Initialises everything that has not been initialized.
        TODO: better initialization for data
        """
        with tf.name_scope("initialisation"):
            extra_variables = []
            # if self.data_init_test:
            #     extra_variables.append(tf.group(self.data_init, self.data_init_test))
            # else:
            #     extra_variables.append(tf.group(self.data_init))
            self.init_uninit(extra_variables)

            if self.data_init_test: 
                init_op = tf.group(self.data_init, self.data_init_test)
            else:
                init_op = tf.group(self.data_init)
        self.sess.run(init_op)

    def setup_input(self, train_record, test_record=None, batch_size=1, 
                    num_parallele_batch=8, verbose=0):
        """
        Setting up data feed queues
        """
        self.init_queue(train_record, test_record, batch_size, num_parallele_batch, verbose)
        self.init_queue_node()

    def setup_last_graph(self, train_record, test_record=None,
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

            if self.tensorboard:
                self.add_summary_images()
                self.summary_writer = tf.summary.FileWriter(log + '/train', 
                                                            graph=self.sess.graph)
                self.merged_summaries = self.summarise_model()
                self.merged_summaries_test = self.summarise_model(train=False)
                if test_record is not None:
                    self.summary_test_writer = tf.summary.FileWriter(log + '/test',
                                                                     graph=self.sess.graph)
        self.setup_init()
        self.score_recorder = ScoreRecorder(self.saver, self.sess, 
                                            log, lag=early_stopping)
     
    def setup_mean(self, mean_array):
        """
        Adds mean substraction into the graph.
        If you use the queues or the self.image placeholder
        the mean will be subtracted automatically.
        """
        if mean_array is not None:
            mean_tensor = tf.constant(mean_array, dtype=tf.float32)
            self.mean_tensor = tf.reshape(mean_tensor, [1, 1, self.num_channels])

    def summarise_model(self, train=True):
        """
        Lists all summaries in one list that is then merged.
        """
        if train:
            summaries = [ut.add_to_summary(var) for var in self.var_to_summarise]
            with tf.name_scope("activations"):
                for var in self.var_activation_to_summarise:
                    hist_scal_var = ut.add_activation_summary(var)
                    summaries = summaries + hist_scal_var
            for grad, var in self.gradient_to_summarise:
                summaries.append(ut.add_gradient_summary(grad, var))
            # for var in self.images_to_summarise:
            #     summaries.append(ut.add_image_summary(var))
            for summary in self.additionnal_summaries:
                summaries.append(summary)
        else:
            summaries = self.test_summaries

        with tf.name_scope("create_summary"):
            summaries = [el for el in summaries if el is not None]
            merged_summary = tf.summary.merge(summaries)
            return merged_summary



    def train(self, train_record, test_record=None,
              learning_rate=0.001, lr_procedure="1epoch",
              weight_decay=0.0005, batch_size=1,
              decay_ema=0.9999, k=0.96, n_epochs=10,
              early_stopping=3, mean_array=None, 
              loss_func=tf.nn.l2_loss, verbose=0,
              save_weights=True, num_parallele_batch=8,
              log=None, restore=False, track_variable="loss"):
        """
        Train the model
        restore allows to re-initializing the variable as log will be empty.
        """
        if log is None:
            log = self.log
        else:
            check_or_create(log)

        if early_stopping != 3:
            self.saver = self.saver_object(self.training_variables, 
                                           keep=early_stopping, 
                                           verbose=verbose, log=log,
                                           restore=restore)

        steps_in_epoch = ut.record_size(train_record) // batch_size
        test_steps = ut.record_size(test_record) if test_record else None
        max_steps = steps_in_epoch * n_epochs
        self.setup_last_graph(train_record, test_record, learning_rate,
                              lr_procedure, weight_decay, batch_size, k,
                              decay_ema, loss_func, early_stopping, 
                              steps_in_epoch, mean_array, num_parallele_batch,
                              verbose, log)

        begin_iter = int(self.global_step.eval())
        begin_epoch = begin_iter // steps_in_epoch
        last_epoch = begin_epoch + n_epochs
        last_iter = max_steps + begin_iter
        range_ = verbose_range(begin_iter, last_iter, "training ",
                               verbose, 0)
        for step in range_:
            self.sess.run(self.training_op)
            if (step - begin_epoch + 1) % steps_in_epoch == 0 and (step - begin_epoch) != 0:
                epoch_number = step // steps_in_epoch
                if verbose > 1:
                    i = datetime.now()
                    tqdm.write('\n')
                    tqdm.write(i.strftime('[%Y/%m/%d %H:%M:%S]: '))
                if save_weights:  
                    self.saver.save(self.sess, log + '/' + "model.ckpt", 
                                    global_step=epoch_number)
                names = ["loss"] + self.metric_handler.tensors_name()
                
                values_train = self.record_train_step(epoch_number, last_epoch, verbose)
                self.score_recorder.diggest(epoch_number, values_train, names)
                
                if test_record:
                    values_test = self.record_test_set(epoch_number, last_epoch, 
                                                       test_steps, verbose)

                    self.score_recorder.diggest(epoch_number, values_test, names, train=False)

                    if self.score_recorder.stop(track_variable):
                        if verbose > 0:
                            tqdm.write('stopping early')
                        break
        if test_record:
            self.score_recorder.save_best(track_variable, save_weights)

        return self.score_recorder.all_tables()

    def tf_compute_metrics(self, lbl, pred, list_metrics):
        """
        Compute tensorflow metrics for a pred and a lbl
        """
        self.metric_handler = TFMetricsHandler(lbl, pred, list_metrics)

        if self.tensorboard:
            for __s in self.metric_handler.summaries():
                self.additionnal_summaries.append(__s)
                # Disabled as we need several steps averaged
                # self.test_summaries.append(__s)
    def weight_xavier(self, kernel, s_input_channel, size_output_channel):
        """
        xavier initialization with 'random seed' if seed is set.
        """
        if self.seed:
            seed = np.random.randint(MAX_INT)
        else:
            seed = self.seed
        weights = ut.weight_xavier(kernel, s_input_channel, 
                                   size_output_channel, seed=seed)
        return weights
