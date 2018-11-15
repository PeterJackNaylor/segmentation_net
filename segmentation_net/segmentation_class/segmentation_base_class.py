


import tensorflow as tf

class SegmentationBaseClass:
	def __init__(
            self,
            image_size=(224, 224),
            log="/tmp/net",
            mean_array=None,
            num_channels=3,
            num_labels=2,
            seed=None,
            verbose=0,
            displacement=0):
        ## this doesn't work yet... 
        ## when the graph is initialized it is completly equal everywhere.
        ## https://github.com/tensorflow/tensorflow/issues/9171
        np.random.seed(seed)
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        # Parameters
        self.image_size = image_size
        self.log = log
        self.mean_array = mean_array
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.tensorboard = tensorboard
        self.seed = seed
        self.verbose = verbose
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
        self.setup_input_network()
		self.setup_mean()
        # Empty variables to be defined in sub methods
        # Usually these variables will become tensorflow
        # variables

        # self.annotation = None
        # self.data_init = None
        # self.data_init_test = None
        # self.image = None
        # self.image_bf_mean = None
        # self.image_placeholder = None
        # self.learning_rate = None
        # self.loss = None
        # self.mean_tensor = None
        # self.merged_summaries = None
        # self.merged_summaries_test = None
        # self.metric_handler = None
        # self.predictions = None
        # self.optimizer = None
        # self.score_recorder = None
        # self.summary_writer = None
        # self.summary_test_writer = None
        # self.test_iterator = None
        # self.training_op = None
        # self.train_iterator = None
        # self.test_iterator = None

        # multi-class or binary
        self.list_metrics = metric_bundles_function(num_labels)

        # Initializing models

        self.init_architecture(self.verbose)
        self.evaluation_graph(self.list_metrics, self.verbose)
        # self.init_training_graph()
        self.saver = self.saver_object(self.training_variables, 3, self.verbose, 
                                       self.log, restore=True)

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


    def non_activation_f(self, i_layer):
        """
        Defining relu layer.
        The default is relu. Feel free to redefine non activation
        function for the standard convolution_
        """
        act = tf.nn.relu(i_layer)
        return act

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

    def saver_object(self,  keep=3, verbose=0, restore=False):
        """
        Defining the saver, it will load if possible.
        """
        if verbose > 1:
            tqdm.write("setting up saver...")
        saver = tf.train.Saver(self.training_variables, max_to_keep=keep)
        if log and restore:
            self.restore(saver, self.log, verbose)
        return saver

    def setup_input_network(self):

        with tf.name_scope('default_inputs'):

            self.global_step = tf.Variable(0., name='global_step', trainable=False)
            self.is_training = tf.placeholder_with_default(True, shape=[])

            input_shape = (None, None, None, self.num_channels)
            label_shape = (None, None, None, 1)

            rgb_default_shape = (1, self.image_size[0] + 2 * self.displacement, 
            					 self.image_size[1] + 2 * self.displacement, 3)
            lbl_default_shape = (1, self.image_size[0], self.image_size[1], 1)
            default_input = np.zeros(shape=rgb_default_shape, dtype='float32')
            default_label = np.zeros(shape=lbl_default_shape, dtype='float32')

            self.inp_variable = tf.Variable(default_input, validate_shape=False, 
                                            name="rgb_input")
            self.lbl_variable = tf.Variable(default_label, validate_shape=False, 
                                            name="lbl_input")
            
            self.input_node = tf.placeholder_with_default(self.inp_variable, shape=input_shape)
            self.label_node = tf.placeholder_with_default(self.lbl_variable, shape=label_shape)
     
    def setup_mean(self):
        """
        Adds mean substraction into the graph.
        If you use the queues or the self.image placeholder
        the mean will be subtracted automatically.
        """
        if self.mean_array is not None:
            mean_tensor = tf.constant(mean_array, dtype=tf.float32)
            self.mean_tensor = tf.reshape(mean_tensor, [1, 1, self.num_channels])
