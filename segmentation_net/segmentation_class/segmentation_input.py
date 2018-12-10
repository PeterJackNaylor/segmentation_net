#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package file tf_record

Segmentation_base_class ->  SegmentationInput -> SegmentationCompile -> 
SegmentationSummaries -> Segmentation_model_utils -> Segmentation_train

"""

from ..data_read_decode import read_and_decode
from .segmentation_base_class import *

class SegmentationInput(SegmentationBaseClass):

    def init_queue_train(self, train_record, batch_size, num_parallele_batch, decode):
        """ Builds training queue for the network.

        Args:
            train_record: string, path to a tensorflow record file for training.
            batch_size : integer, size of batch to be feeded at each 
                         iterations.
            num_parallele_batch : integer, number of workers to use to 
                                  perform paralelle computing.
            decode: tensorflow function (default: tf.float32) how to decode the bytes in
                    the tensorflow records for the input rgb data.
        Returns:
            A tuple, tensorflow initializer and tensorlfow iterator.

        """
        with tf.device('/cpu:0'):
            with tf.name_scope('training_queue'):
                out = read_and_decode(train_record, 
                                      self.image_size[0], 
                                      self.image_size[1],
                                      batch_size,
                                      num_parallele_batch,
                                      channels=self.num_channels,
                                      displacement=self.displacement,
                                      seed=self.seed,
                                      decode=decode)
                data_init, train_iterator = out
        if self.verbose > 1:
            tqdm.write("train queue initialized")
        return data_init, train_iterator

    def init_queue_test(self, test_record, batch_size, num_parallele_batch, decode):
        """ Builds testing queue for the network.

        Args:
            train_record: string, path to a tensorflow record file for training.
            batch_size : integer, size of batch to be feeded at each 
                         iterations.
            num_parallele_batch : integer, number of workers to use to 
                                  perform paralelle computing.
            decode: tensorflow function (default: tf.float32) how to decode the bytes in
                    the tensorflow records for the input rgb data.
        Returns:
            A tuple, tensorflow initializer and tensorlfow iterator.

        """
        with tf.device('/cpu:0'):
            with tf.name_scope('testing_queue'):
                out = read_and_decode(test_record, 
                                      self.image_size[0], 
                                      self.image_size[1],
                                      batch_size,
                                      num_parallele_batch,
                                      train=False,
                                      channels=self.num_channels,
                                      displacement=self.displacement,
                                      buffers=10,
                                      shuffle_buffer=1,
                                      seed=self.seed,
                                      decode=decode)
                data_init_test, test_iterator = out
            
        if self.verbose > 1:
            tqdm.write("test queue initialized")
        return data_init_test, test_iterator

    def init_queue_node(self, train_iterator, test_iterator):
        """ Builds testing queue for the network. Train_iterator 
            and test_iterator can be None and will therefore be ignored.

        Args:
            train_iterator : tensorflow iterator for training.
            test_iterator : tensorflow iterator for testing.
        Returns:
            Tuple of tensorflow variables, one for the raw input and one
            for the annoations. 

        """
        def f_true():
            """
            The callable to be performed if pred is true.
            i.e. fetching data from the train queue
            """
            train_images, train_labels = train_iterator.get_next()
            # train_images = tf.Print(train_images, [tf.shape(train_images)], "dequeueing TRAIN.")
            return train_images, train_labels

        def f_false():
            """
            The callable to be performed if pred is false.
            i.e. fetching data from the test queue
            """
            test_images, test_labels = test_iterator.get_next()
            # test_images = tf.Print(test_images, [], "dequeueing TEST.")
            return test_images, test_labels

        with tf.name_scope('switch'):

            only_train = train_iterator is not None
            only_test = test_iterator is not None

            if only_train and only_test:
                image_outqueue, annotation_outqueue = tf.cond(self.is_training, f_true, f_false)
            elif only_train:
                image_outqueue, annotation_outqueue = f_true()
            elif only_test:
                image_outqueue, annotation_outqueue = f_false()
            else:
                tqdm.write("No queue were found")
        return image_outqueue, annotation_outqueue

    def setup_queues(self, train_record=None, test_record=None, batch_size=1, 
                    num_parallele_batch=8, decode=tf.float32):
        """Setting up data queues for the model. If train, or test_record is None
           they will be ignored.

        Args:
            train_record: string (default: None), path to a tensorflow record file for training.
            test_record: string (default: None), if given, the model will be evaluated on 
                         the test data at every epoch.
            batch_size : integer (default: 1) Size of batch to be feeded at each 
                         iterations.
            num_parallele_batch : integer (default: 8) number of workers to use to 
                                  perform paralelle computing.
            decode: tensorflow function (default: tf.float32) how to decode the bytes in
                    the tensorflow records for the input rgb data.
        Returns:
            Tuple of tensorflow variables, one for the raw input and one
            for the annoations. 
        """
        to_init = []

        if train_record is not None:
            dtrain_init, train_iterator = self.init_queue_train(train_record, batch_size, 
                                                                num_parallele_batch, decode)
            to_init.append(dtrain_init)
        else:
            train_iterator = None


        if test_record is not None:
            dtest_init, test_iterator = self.init_queue_test(test_record, batch_size, 
                                                             num_parallele_batch, decode)
            to_init.append(dtest_init)
        else:
            test_iterator = None

        if to_init:
            init_op = tf.group(*to_init)
            self.sess.run(init_op)
        else:
            tqdm.write("No data initialization possible")
        
        image_out, anno_out = self.init_queue_node(train_iterator, test_iterator)
        
        return image_out, anno_out

    def setup_input_network(self):
        """ Magical function that creates both for the raw input and annotation input
        two tensorflow object, one placeholder that will be the variable for manual 
        feeding of data and one tensorflow variable that will be assigned the value of
        the queues that will be linked to it. In such a case, the place holder will take
        as default value the value of this tensorflow variable.

        Returns:
            Four tensorflow objecs, raw input variable, annotation input variable,
            raw input placeholder, annotation input placeholder.

        """
        with tf.name_scope('default_inputs'):

            self.global_step = tf.Variable(0., name='global_step', trainable=False)
            self.is_training = tf.placeholder_with_default(True, shape=[])

            input_shape = (None, None, None, self.num_channels)
            label_shape = (None, None, None, 1)

            rgb_default_shape = (self.fake_batch, self.image_size[0] + 2 * self.displacement, 
                                 self.image_size[1] + 2 * self.displacement, 
                                 self.num_channels)
            lbl_default_shape = (self.fake_batch, self.image_size[0], self.image_size[1], 1)
            default_input = zeros(shape=rgb_default_shape, dtype='float32')
            default_label = zeros(shape=lbl_default_shape, dtype='float32')

            # self.inp_variable = tf.Variable(default_input, validate_shape=False, 
            #                                 name="rgb_input")
            # self.lbl_variable = tf.Variable(default_label, validate_shape=False, 
            #                                 name="lbl_input")
            from tensorflow.python.ops import resource_variable_ops as rr

            rgb_variable = rr.ResourceVariable(default_input, dtype=tf.float32, validate_shape=False, name="rgb_variable")
            lbl_variable = rr.ResourceVariable(default_label, dtype=tf.float32, validate_shape=False, name="lbl_variable")
            # rgb_variable = tf.Variable(default_input, validate_shape=False, 
            #                            name="rgb_variable")
            # lbl_variable = tf.Variable(default_label, validate_shape=False, 
            #                            name="lbl_variable")

            # rgb_variable2 = tf.Print(rgb_variable, [tf.shape(rgb_variable)], "accessing rgb variable")
            rgb_placeholder = tf.placeholder_with_default(rgb_variable, 
                                                          shape=input_shape)
            # rgb_placeholder = tf.Print(rgb_placeholder, [tf.shape(rgb_placeholder)], "accessing rgb placeholder")
            lbl_placeholder = tf.placeholder_with_default(lbl_variable, 
                                                          shape=label_shape)

            return rgb_variable, lbl_variable, rgb_placeholder, lbl_placeholder
        
    def setup_mean(self):
        """
        Adds mean substraction into the graph.
        If you use the queues or the self.image placeholder
        the mean will be subtracted automatically.
        """
        if self.mean_array is not None:
            mean_tensor = tf.constant(self.mean_array, dtype=tf.float32)
            self.mean_tensor = tf.reshape(mean_tensor, [1, 1, self.num_channels])

    def subtract_mean(self, variable):
        """
        Performs tensorflow operations to subtract mean if this one is available.

        Args:
            variable : tensorflow variable on which we will perform mean substraction if available.

        Returns:
            The variable given in input minus (if defined) the mean tensor.
        """
        with tf.name_scope('mean_subtraction'):
            if self.mean_array is not None:
                result_node = variable - self.mean_tensor
            else:
                result_node = variable
            return result_node

