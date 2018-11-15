
from segmentation_compile import *

class SegmentationInit(SegmentationCompile):
	
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

            test_init = self.data_init_test is not None
            train_init = self.data_init is not None

            if train_init and test_init: 
                init_op = tf.group(self.data_init, self.data_init_test)
            elif train_init:
                init_op = tf.group(self.data_init)
            elif test_init:
                init_op = tf.group(self.data_init_test)
            else:
                if self.verbose:
                    tqdm.write("No data initialization possible")
        self.sess.run(init_op)

    def initialize_all(self):
        """
        initialize all variables
        """
        self.sess.run(tf.global_variables_initializer())

    def init_uninit(self, extra_variables):
        ut.initialize_uninitialized_vars(self.sess, {self.is_training: False},
                                         extra_variables)


