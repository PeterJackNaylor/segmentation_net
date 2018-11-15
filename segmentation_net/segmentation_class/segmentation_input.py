
from segmentation_summaries import *


class SegmentationInput(SegmentationSummaries):

    def init_queue_train(self, train, batch_size, num_parallele_batch, verbose):
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
        if verbose > 1:
            tqdm.write("train queue initialized")

    def init_queue_test(self, test, verbose):
        """
        New queues for coordinator
        """
        with tf.device('/cpu:0'):
            with tf.name_scope('testing_queue'):
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
            tqdm.write("test queue initialized")

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
            only_train = self.train_iterator is not None
            only_test = self.test_iterator is not None
            if only_train and only_test:
                self.image_bf_mean, self.annotation = tf.cond(self.is_training, f_true, f_false)
            elif only_train:
                self.image_bf_mean, self.annotation = f_true()
            elif only_test:
                self.image_bf_mean, self.annotation = f_false()
            else:
                tqdm.write("No queue were found")
        if self.mean_tensor is not None:
            self.image = self.image_bf_mean - self.mean_tensor
        else:
            self.image = self.image_bf_mean

    def setup_input(self, train_record, test_record=None, batch_size=1, 
                    num_parallele_batch=8, verbose=0):
        """
        Setting up data feed queues
        """
        self.init_queue_train(train_record, batch_size, num_parallele_batch, verbose)
        if test_record is not None:
            self.init_queue_test(test_record, verbose)
        self.init_queue_node()