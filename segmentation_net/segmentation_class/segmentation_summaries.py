



from base_class import *

class SegmentationSummaries(SegmentationBaseClass):

	def setup_summary(self):
		"""
		Main member to setup summaries for tensorboard.
		"""
        self.add_summary_images()
        self.summary_writer = tf.summary.FileWriter(log + '/train', 
                                                    graph=self.sess.graph)
        self.merged_summaries = self.summarise_model()
        self.merged_summaries_test = self.summarise_model(train=False)
        if test_record is not None:
            self.summary_test_writer = tf.summary.FileWriter(log + '/test',
                                                             graph=self.sess.graph)

    def add_summary_images(self):
        """
        Image summary to add to the summary
        TODO Does not work with the cond in the network flow...
        """
        input_s = tf.summary.image("input", self.input_node, max_outputs=3)
        label_s = tf.summary.image("label", self.label_node, max_outputs=3)
        pred = tf.expand_dims(tf.cast(self.predictions, tf.float32), dim=3)
        prob = tf.expand_dims(tf.cast(tf.multiply(self.probability[:, :, :, 0], 255.), tf.float32), dim=3)
        
        predi_s = tf.summary.image("pred", pred, max_outputs=3)
        probi_s = tf.summary.image("prob", prob, max_outputs=3)
        for __s in [input_s, label_s, predi_s, probi_s]:
            self.additionnal_summaries.append(__s)
            self.test_summaries.append(__s)


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

