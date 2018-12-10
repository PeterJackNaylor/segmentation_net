#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package file tf_record

Segmentation_base_class ->  SegmentationInput -> SegmentationCompile -> 
SegmentationSummaries -> Segmentation_model_utils -> Segmentation_train

"""

from .segmentation_compile import *

class SegmentationSummaries(SegmentationCompile):
    def setup_summary(self, log, test_record):
        """ Creates tensorflow summary writters that will
        write their output to log. And if test_record is given
        write also the test metrics.

        Args:
            log: string, where to writte the summaries.
            test_record: string, if not none will prepare the test
                         summary writter.
        Returns:
            The summary tensorflow object and the merged summaries to call
            during the session. Optionnaly with the test summaries.
        """
        self.add_summary_images()
        summary_writer = tf.summary.FileWriter(log + '/train', graph=self.sess.graph)
        merged_summaries = self.summarise_model()

        if test_record is not None:
            merged_test_summaries = self.summarise_model(train=False)
            summary_test_writer = tf.summary.FileWriter(log + '/test', graph=self.sess.graph)
        else:
            merged_test_summaries = None
            summary_test_writer = None

        return summary_writer, merged_summaries, summary_test_writer, merged_test_summaries

    def add_summary_images(self):
        """ 
        Adds the image summaries to the attribute additionnal_summaries
        and test_summaries where they will be collected by the summary manager.
        """

        input_s = tf.summary.image("input", self.rgb_ph, max_outputs=3)
        label_s = tf.summary.image("label", self.lbl_ph, max_outputs=3)
        pred = tf.expand_dims(tf.cast(self.predictions, tf.float32), axis=3)
        prob = tf.expand_dims(tf.cast(tf.multiply(self.probability[:, :, :, 0], 255.), tf.float32), axis=3)
        
        predi_s = tf.summary.image("pred", pred, max_outputs=3)
        probi_s = tf.summary.image("prob", prob, max_outputs=3)
        for __s in [input_s, label_s, predi_s, probi_s]:
            self.additionnal_summaries.append(__s)
            self.test_summaries.append(__s)


    def summarise_model(self, train=True):
        """
        Lists all summaries in one list that is then merged.
        Args:
            train: bool, wether you are training or not.
        Returns:
            A list of summaries merged in a tensorflow object 
            respective if it is training or not. 

        If not training, the summaries do not include the gradients,
        the histogrammes and the activations.
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

