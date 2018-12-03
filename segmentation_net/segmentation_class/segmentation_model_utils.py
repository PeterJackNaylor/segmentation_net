#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package file tf_record

Segmentation_base_class ->  SegmentationInput -> SegmentationCompile -> 
SegmentationSummaries -> Segmentation_model_utils -> Segmentation_train

"""

import numpy as np

from .segmentation_summaries import *
from ..utils import merge_dictionnary

class SegmentationModelUtils(SegmentationSummaries):

    def infer(self, train=True, tensor_rgb=None, tensor_lbl=None, keep_prob=False, metrics=False,
              overrule_tensorboard=True, step=0, with_input=False, control=None, extra_tensors=None,
              extra_tensors_names=None):
        feed_dict = {self.is_training: train}

        if tensor_rgb is not None:
            feed_dict[self.rgb_ph] = tensor_rgb
            if tensor_lbl is not None:
                feed_dict[self.lbl_ph] = tensor_lbl

        tensors_names = []
        tensors_to_get = []
        
        if keep_prob:
            tensors_names += ["predictions", "probability"]
            tensors_to_get += [self.predictions, self.probability]

        # raise error if tensor not none, lbl is none and metric is true?
        if metrics:
            tensors_names += ["loss"] + self.metric_handler.tensors_name()
            tensors_to_get += [self.loss] + self.metric_handler.tensors_out()

        if with_input:
            tensors_names += ["input_rgb", "input_lbl"]
            tensors_to_get += [self.rgb_ph, self.lbl_ph]

        if extra_tensors:
            tensors_to_get += extra_tensors
            if not extra_tensors_names:
                extra_tensors_names= ["extra_input_{}".format(i) for i in range(len(extra_tensors))]
            tensors_names += extra_tensors_names

        if overrule_tensorboard:
            if self.tensorboard:
                if train:
                    tensors_names += ["train_summary"]
                    tensors_to_get += [self.merged_summaries]
                else:
                    tensors_names += ["test_summary"]
                    tensors_to_get += [self.merged_summaries_test]

        if control is not None:
            tensors_to_get = tf.tuple(tensors_to_get, control_inputs=control)

        tensors_out = self.sess.run(tensors_to_get, feed_dict=feed_dict)

        if overrule_tensorboard:
            if self.tensorboard:
                if train:
                    self.summary_writer.add_summary(tensors_out[-1], step)
                else:
                    self.summary_test_writer.add_summary(tensors_out[-1], step)
                del tensors_names[-1]
                del tensors_out[-1]

        out_dic = {}
        for name, tens in zip(tensors_names, tensors_out):
            out_dic[name] = tens
        return out_dic

    def infer_train_step(self, step=0, control=None):
        dic_step = self.infer(train=True, tensor_rgb=None, tensor_lbl=None, 
                              keep_prob=False, metrics=True, overrule_tensorboard=True, 
                              step=step, control=control)
        if self.tensorboard:
            for name, value in sorted(dic_step.items()):
                if np.isscalar(value):
                    ut.add_value_to_summary(value, name, self.summary_writer,
                                            step, tag="evaluation")
        if self.verbose:
            msg = "training : "
            for name, value in sorted(dic_step.items()):
                if np.isscalar(value):
                    msg += "{} : {:.5f}  ".format(name, value)
            tqdm.write(msg)

        return dic_step 

    def infer_test_set(self, step, n_test, probability=False, during_training=False, control=None): 
        dic = {}
        first = True
        for _ in range(n_test):
            dic_step = self.infer(train=False, tensor_rgb=None, tensor_lbl=None,
                                  keep_prob=probability, metrics=True, overrule_tensorboard=first, 
                                  step=step, control=control)
            first = False
            dic = merge_dictionnary(dic, dic_step)

        if self.tensorboard:
            for name, value in sorted(dic.items()):
                if np.isscalar(value[0]):
                    ut.add_value_to_summary(np.mean(value), name, self.summary_test_writer,
                                            step, tag="evaluation")

        if self.verbose:
            msg = "    test : "
            for name, value in sorted(dic.items()):
                if np.isscalar(value[0]):
                    value = np.mean(value)
                    msg += "{} : {:.5f}  ".format(name, value)
                    if during_training:
                        dic[name] = value
            tqdm.write(msg)   
        return dic


    def predict_list(self, list_rgb, list_lbl=None):
        """
        Predict from list of rgb files.
        """

        if list_lbl is None:
            list_lbl = [None for el in list_rgb]

        dic_res = {}
        for i, rgb in enumerate(list_rgb):
            dic_out = self.predict(rgb, label=list_lbl[i])
            dic_res = merge_dictionnary(dic_res, dic_out)

        return dic_res

    def predict(self, rgb, label=None):
        """
        It will predict and return a dictionnary of results
        """


        tensor = np.expand_dims(rgb, 0)
        metrics = False

        if label is not None:
            label = np.expand_dims(label, 0)
            label = np.expand_dims(label, -1)
            metrics = True

        dic_step = self.infer(train=False, tensor_rgb=tensor, tensor_lbl=label, 
                              keep_prob=True, metrics=metrics, overrule_tensorboard=False)
        dic_step 
        for name, value in sorted(dic_step.items()):
            if not np.isscalar(value):
                dic_step[name] = value[0]
        return dic_step

    def predict_record(self, record, extra_tensors=None, extra_tensors_names=None, 
                       init_queues=True, length=None, inputs=False, batch_size=1,
                       num_parallele_batch=1, to_control=None, keep_prob=True,
                       metrics=True):

        if init_queues:
            image_out, anno_out = self.setup_queues(record, None, 
                                                    batch_size, num_parallele_batch)
            assign_rgb_to_queue = tf.assign(self.rgb_v, image_out, 
                                            validate_shape=False)
            assign_lbl_to_queue = tf.assign(self.lbl_v, anno_out, 
                                            validate_shape=False)
            assign_to_variable = [assign_rgb_to_queue, assign_lbl_to_queue]
            if to_control:
                tqdm.write("Overwritting to_control variable as initialising queue")
            to_control = tf.tuple(assign_to_variable, control_inputs=[image_out, anno_out])
            self.init_uninit([])
            blank = tf.tuple([self.is_training], name=None, control_inputs=to_control)
            self.sess.run(blank)

        if not length:
            length = ut.record_size(record) // batch_size

        dic = {}

        for _ in range(length):
            dic_step = self.infer(train=False, tensor_rgb=None, tensor_lbl=None,
                                  keep_prob=keep_prob, metrics=metrics, overrule_tensorboard=False, 
                                  step=-1, control=to_control, with_input=inputs, extra_tensors=extra_tensors,
                                  extra_tensors_names=extra_tensors_names)
            first = False
            dic = merge_dictionnary(dic, dic_step)

        if self.verbose:
            msg = "    test : "
            for name, value in sorted(dic.items()):
                if np.isscalar(value[0]):
                    value = np.mean(value)
                    msg += "{} : {:.5f}  ".format(name, value)
            tqdm.write(msg)   
        return dic  

# this was in infer_test_set 
        # import matplotlib.pylab as plt
        # # import pdb; pdb.set_trace()

        # f1, axes1 = plt.subplots(nrows=5, ncols=8);
        # for j in range(5):
        #     for i in range(8):
        #         axes1[j, i].imshow(dic["input_lbl"][j][i,:,:,0].astype('uint8'))
        # f1.suptitle('label distance', fontsize=16)
        # plt.savefig("test/test_dist_{}.png".format(step))


        # f0, axes0 = plt.subplots(nrows=5, ncols=8);
        # for j in range(5):
        #     for i in range(8):
        #         axes0[j, i].imshow(dic['input_lblint'][j][i,:,:,0])
        # f0.suptitle('label int', fontsize=16)
        # plt.savefig("test/test_lbl_int_{}.png".format(step))


        # import pdb; pdb.set_trace()
        # plt.show()
        # f, axes = plt.subplots(nrows=5, ncols=8);
        # for j in range(5):
        #     for i in range(8):

        #         if dic['input_rgb'][j].shape[1] == dic['probability'][j].shape[1]:
        #             axes[j, i].imshow(dic['input_rgb'][j][i,:,:].astype('uint8'))
        #         else:
        #             axes[j, i].imshow(dic['input_rgb'][j][i,92:-92,92:-92].astype('uint8'))

        # f.suptitle('RGB', fontsize=16)
        # plt.savefig("test/test_RGB_{}.png".format(step))


        # f2, axes2 = plt.subplots(nrows=5, ncols=8);
        # for j in range(5):
        #     for i in range(8):
        #         axes2[j, i].imshow(dic['predictions'][j][i,:,:])
        # f2.suptitle('predictions', fontsize=16)
        # plt.savefig("test/test_prediction_{}.png".format(step))


        # f3, axes3 = plt.subplots(nrows=5, ncols=8);
        # for j in range(5):
        #     for i in range(8):
        #         axes3[j, i].imshow(dic['probability'][j][i,:,:,0])
        # f3.suptitle('probability', fontsize=16)
        # plt.savefig("test/test_probability_{}.png".format(step))

        # plt.show()
        # import pdb; pdb.set_trace()

        # change tensors_out so that the printing is good.
        # also need to feed the data handler.

# this was in infer_train_set
        # import pdb; pdb.set_trace()
        # import matplotlib.pylab as plt

        # num = dic_step['input_lbl'].shape[0]

        # f, axes = plt.subplots(nrows=4, ncols=num);
        # for i in range(num):
        #     axes[0, i].imshow(dic_step['input_lbl'][i,:,:,0].astype('uint8'))
        #     # axes[1, i].imshow(dic_step['input_lblint'][i,:,:,0])
        #     if dic_step['input_rgb'].shape[1] == dic_step['predictions'].shape[1]:
        #         axes[1, i].imshow(dic_step['input_rgb'][i,:,:].astype('uint8'))
        #     else:
        #         axes[1, i].imshow(dic_step['input_rgb'][i,92:-92,92:-92].astype('uint8'))
        #     axes[2, i].imshow(dic_step['predictions'][i,:,:])
        #     axes[3, i].imshow(dic_step['probability'][i,:,:,0])
        #     axes[3, i].set_title("{:.3f}_{:.3f}".format(dic_step['probability'][i,:,:,0].max(), dic_step['probability'][i,:,:,0].min() ))
        # plt.savefig("train/record_train_{}.png".format(step))


        # plt.show()
        # import pdb; pdb.set_trace()