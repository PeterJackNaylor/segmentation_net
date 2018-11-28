#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""segnet package file tf_record

Segmentation_base_class ->  SegmentationInput -> SegmentationCompile -> 
SegmentationSummaries -> Segmentation_model_utils -> Segmentation_train

"""

from datetime import datetime

from tqdm import trange
from ..net_utils import ScoreRecorder
from .segmentation_model_utils import *



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

class SegmentationTrain(SegmentationModelUtils):
    def train(self, train_record, test_record=None,
              learning_rate=0.001, lr_procedure="1epoch",
              weight_decay=0.0005, batch_size=1,
              decay_ema=0.9999, k=0.96, n_epochs=10,
              early_stopping=3, loss_func=tf.nn.l2_loss, 
              save_weights=True, new_log=None, 
              num_parallele_batch=8, restore=False, 
              track_variable="loss", track_training=False,
              tensorboard=True, restore_best=False):
        """
        Train the model
        restore allows to re-initializing the variable as log will be empty.
        Any parameter set to 0 or None is ignored.
        """

        steps_in_epoch = ut.record_size(train_record) // batch_size
        test_steps = ut.record_size(test_record) //batch_size if test_record is not None else None
        max_steps = steps_in_epoch * n_epochs
        self.tensorboard = tensorboard

        if new_log is None:
            new_log = self.log
        else:
            check_or_create(new_log)

        stop_early = early_stopping is not None and early_stopping != 0
        if not stop_early:
            early_stopping = 0

        if early_stopping in [0, 3]:
            ## this saver is to ensure that we can restore to the best weights at the end
            self.saver = self.saver_object(keep=early_stopping + 1, 
                                           log=new_log,
                                           restore=restore)

        
                
        self.score_recorder = ScoreRecorder(self.saver, self.sess, 
                                            new_log, stop_early=stop_early, 
                                            lag=early_stopping)

        if not (k == 0 or k is None or lr_procedure is None or lr_procedure == ""):
            with tf.name_scope('learning_rate_scheduler'):
                lrs = self.learning_rate_scheduler(learning_rate, k, lr_procedure,
                                                   steps_in_epoch)
                if self.verbose:
                    tqdm.write("learning_rate_scheduler NOT WORKING")
                    tqdm.write("learning_rate_scheduler implemented")
        else:
            lrs = learning_rate

        if self.tensorboard:
            sw, ms, stw, mts = self.setup_summary(new_log, test_record)
            self.summary_writer = sw
            self.merged_summaries = ms
            if test_record:
                self.summary_test_writer = stw
                self.merged_summaries_test = mts
            if self.verbose:
                tqdm.write("summaries added")

        if weight_decay != 0:
            with tf.name_scope('regularization'):
                self.loss = self.regularize_model(self.loss, loss_func, weight_decay)
                if self.verbose:
                    tqdm.write("CAN'T FIND REGULARIZATION in graph")
                    tqdm.write('regularization weight decay added: {}'.format(weight_decay))

        with tf.name_scope('optimization'):
            opt = self.optimization(lrs, self.loss, self.training_variables)

        if decay_ema != 0 and decay_ema is not None:
            with tf.name_scope('exponential_moving_average'):
                training_op = self.exponential_moving_average(opt, self.training_variables, decay_ema)
                if self.verbose:
                    tqdm.write("Exponential moving average added to prediction")
        else:
            training_op = opt

        with tf.name_scope('input_from_queue'):
            image_out, anno_out = self.setup_queues(train_record, test_record, 
                                                    batch_size, num_parallele_batch)

            # To plug in the queue to the main graph
        # with tf.control_dependencies([image_out, anno_out]):
        with tf.name_scope('queue_assigning'):
            # Control the dependency to allow the flow thought the data queues
            assign_rgb_to_queue = tf.assign(self.rgb_v, image_out, 
                                            validate_shape=False)
            assign_lbl_to_queue = tf.assign(self.lbl_v, anno_out, 
                                            validate_shape=False)
            assign_to_variable = [assign_rgb_to_queue, assign_lbl_to_queue]
        
            to_control = tf.tuple(assign_to_variable, control_inputs=[image_out, anno_out])
            # to_control_train = tf.tuple(assign_to_variable + [lrs], control_inputs=[image_out, anno_out])
            blank = tf.tuple([self.is_training], name=None, control_inputs=to_control)
            train_op = tf.tuple([training_op], name=None, control_inputs=to_control)

        self.init_uninit([])

        tqdm.write("NO REINITIALIZE GLOBAL STEP")
        begin_iter = int(self.global_step.eval())
        begin_epoch = begin_iter // steps_in_epoch
        last_epoch = begin_epoch + n_epochs
        last_iter = max_steps + begin_iter

        range_ = verbose_range(begin_iter, last_iter, "training ",
                               self.verbose, 0)
        self.sess.run(blank)

        for step in range_:
            self.sess.run(train_op)
            #     tqdm.write("WARNING.... RECORDING EVERY STEP")
            if (step - begin_epoch + 1) % steps_in_epoch == 0 and (step - begin_epoch) != 0:
                # If we are at the end of an epoch
                epoch_number = step // steps_in_epoch
                if self.verbose:
                    i = datetime.now()
                    msg = i.strftime('[%Y/%m/%d %H:%M:%S]: ')
                    msg += '  Epoch {} / {}'.format(epoch_number + 1, last_epoch)
                    tqdm.write(msg)

                if save_weights:  
                    self.saver.save(self.sess, new_log + '/' + "model.ckpt", 
                                    global_step=epoch_number + 1)
                
                dic_train_record = self.infer_train_step(epoch_number, control=to_control)
                self.score_recorder.diggest(epoch_number, dic_train_record)
                
                if test_record:
                    self.sess.run(blank, feed_dict={self.is_training:False})
                    dic_test_record = self.infer_test_set(epoch_number, test_steps, 
                                                          during_training=True, control=to_control)
                    self.sess.run(blank, feed_dict={self.is_training:True})

                    self.score_recorder.diggest(epoch_number, dic_test_record, train=False)

                    if self.score_recorder.stop(track_variable, train_set=track_training):
                        if self.verbose > 0:
                            tqdm.write('stopping early')
                        break
        if test_record:
            self.score_recorder.save_best(track_variable, save_weights)
        if restore_best:
            tqdm.write("restore_best NOT IMPLEMENTED")
        return self.score_recorder.all_tables()


        # ttt1, ttt2 = self.sess.run([test, self.conv1])
        # ttt1, ttt2 = self.sess.run([test, self.conv1])
        # import matplotlib.pylab as plt
        # f, axes = plt.subplots(nrows=9, ncols=ttt1[0].shape[0])
        # for i in range(ttt1[0].shape[0]):
        #     for j in range(8):
        #         axes[j, i].imshow(ttt2[i,:,:,j].astype('uint8'))
        #     axes[-1, i].imshow(ttt1[0][i,:,:].astype('uint8'))

        # ttt1, ttt2 = self.sess.run([test, self.conv1])
        # ttt1, ttt2 = self.sess.run([test, self.conv1])

        # fig, axes2 = plt.subplots(nrows=9, ncols=ttt1[0].shape[0])
        # for i in range(ttt1[0].shape[0]):
        #     for j in range(8):
        #         axes2[j, i].imshow(ttt2[i,:,:,j].astype('uint8'))
        #     axes2[-1, i].imshow(ttt1[0][i,:,:].astype('uint8'))

        # plt.show()
        # import pdb; pdb.set_trace()

        # size = self.sess.run([warm_up, warm_up2]) 
        # tqdm.write(str(size[0]))
        # size = self.sess.run([warm_up, warm_up2]) 
        # tqdm.write(str(size[0]))
        # self.sess.run(warm)

            # a, c, d, b, e, ff, ff3, ff2, ff1 = self.sess.run([test, self.probability, self.predictions, self.rgb_ph, self.lbl_ph, self.logit, self.conv3, self.conv2, self.conv1]) # self.label_int,


            # import matplotlib.pylab as plt

            # f, axes = plt.subplots(nrows=4, ncols=c.shape[0]);
            # if b.shape[1] == c.shape[1]:
            #     dis = 0
            # else:
            #     dis = 92
            # if c.shape[0] == 1:
            #     axes[0].imshow(c[0,:,:,0])
            #     if dis== 0:
            #         axes[1].imshow(b[0,:,:].astype('uint8'))
            #     else:
            #         axes[1].imshow(b[0,dis:-dis,dis:-dis].astype('uint8'))
            #     axes[2].imshow(d[0,:,:])
            #     axes[3].imshow(e[0,:,:,0])
            #     #axes[4].imshow(entry[0,:,:,0])
            #     for j in range(5):
            #         axes[j].axis('off')
            # else:
            #     for i in range(c.shape[0]):
            #         axes[0, i].imshow(c[i,:,:,0])
            #         if dis== 0:
            #             axes[1, i].imshow(b[i,:,:].astype('uint8'))
            #         else:
            #             axes[1, i].imshow(b[i,dis:-dis,dis:-dis].astype('uint8'))
            #         axes[2, i].imshow(d[i,:,:])
            #         axes[3, i].imshow(e[i,:,:,0])
            #         #axes[4, i].imshow(entry[i,:,:,0])
            #         for j in range(4):
            #             axes[j, i].axis('off')
            # plt.savefig("train/train_{}.png".format(step))
            # f, axes = plt.subplots(nrows=2, ncols=c.shape[0]);
            # if c.shape[0] == 1:
            #     for j in range(2):
            #         axes[j].imshow(ff[0,:,:,j])
            #         axes[j].axis('off')
            # else:
            #     for i in range(c.shape[0]):
            #         for j in range(2):
            #             axes[j, i].imshow(ff[i,:,:,j])
            #             axes[j, i].axis('off')
        
            # plt.savefig("train/logit_{}.png".format(step))
            # f, axes = plt.subplots(nrows=8, ncols=c.shape[0]);
            # if c.shape[0] == 1:
            #     for j in range(8):
            #         axes[j].imshow(ff3[0,:,:,j])
            #         axes[j].axis('off')
            # else:
            #     for i in range(c.shape[0]):
            #         for j in range(8):
            #             axes[j, i].imshow(ff3[i,:,:,j])
            #             axes[j, i].axis('off')
        
            # plt.savefig("train/conv3_{}.png".format(step))

            # f, axes = plt.subplots(nrows=8, ncols=c.shape[0]);
            # if c.shape[0] == 1:
            #     for j in range(8):
            #         axes[j].imshow(ff2[0,:,:,j])
            #         axes[j].axis('off')
            # else:
            #     for i in range(c.shape[0]):
            #         for j in range(8):
            #             axes[j, i].imshow(ff2[i,:,:,j])
            #             axes[j, i].axis('off')
        
            # plt.savefig("train/conv2_{}.png".format(step))


            # f, axes = plt.subplots(nrows=8, ncols=c.shape[0]);
            # if c.shape[0] == 1:
            #     for j in range(8):
            #         axes[j].imshow(ff1[0,:,:,j])
            #         axes[j].axis('off')
            # else:
            #     for i in range(c.shape[0]):
            #         for j in range(8):
            #             axes[j, i].imshow(ff1[i,:,:,j])
            #             axes[j, i].axis('off')
        
            # plt.savefig("train/conv1_{}.png".format(step))


