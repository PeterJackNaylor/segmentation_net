

from segmentation_input import *

class SegmentationCompile(SegmentationInput):



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

    def regularize_model(self, loss_func, weight_decay):
        """
        Adds regularization to parameters of the model given LOSS_FUNC
        """
        for var in self.var_to_reg:
            self.penalize_with_regularization(var, loss_func, weight_decay)

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
