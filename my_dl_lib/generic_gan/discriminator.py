import tensorflow as tf


class GenericDiscriminator:

    def __init__(self, input_height, input_width, number_of_input_channels=3, number_of_filters_at_first_layer=64,
                 architecture_file=None, tensor_board_logs_dir='./logs', inputs=None,
                 y_labels_dim=None, y_placeholder=None, gan_type='dcgan', r_placeholder=None):
        """

        :param input_height:
        :param input_width:
        :param number_of_input_channels:
        :param number_of_filters_at_first_layer:
        :param architecture_file:
        :param tensor_board_logs_dir:
        :param inputs: if it is not None, then this is the input placeholder which comes from the generator. otherwise,
                       we will initialize this input with a placeholder that will later get a real image.
        :param gan_type: one of ['dcgan', 'conditional_dcgan', 'vanilla_gan', 'vanilla_conditional_gan' ]
        """
        if gan_type not in ['dcgan', 'conditional_dcgan', 'vanilla_gan', 'vanilla_conditional_gan']:
            raise AssertionError("undefined gan type", gan_type)

        self.activated_output_last_layer = None
        self.last_linear_layer_before_activation = None
        self.num_channels = number_of_input_channels
        self.y_dim = y_labels_dim
        self.r_peak_placeholder = r_placeholder
        self.y_input_placeholder = y_placeholder
        self.input_width = input_width
        self.input_height = input_height
        self.number_of_filters_at_first_layer = number_of_filters_at_first_layer
        with tf.variable_scope("discriminator"):
            if inputs is None:
                if input_width == 1:
                    # This means we are working with 1d signal:
                    self.input_placeholder = tf.placeholder(tf.float32, [None, input_height], name='input_placeholder')

                else:
                    self.input_placeholder = tf.placeholder(tf.float32, [None, input_height, input_width,
                                                                     number_of_input_channels], name='input_placeholder')
                self.discriminator_type = 'REAL'
            else:
                self.input_placeholder = inputs
                self.discriminator_type = 'FAKE'

            if gan_type == 'dcgan':
                self.build_regular_dcgan(architecture_file)
            elif gan_type == 'conditional_dcgan':
                self.build_conditional_discriminator(architecture_file)
            elif gan_type == 'vanilla_conditional_gan':
                self.build_conditional_vanilla_discriminator()
            elif gan_type == 'vanilla_gan':
                self.build_vanilla_discriminator()
            elif gan_type == 'ecg_gan':
                self.build_ecg_discriminator()

        self.create_loss_tensors()

    def build_ecg_discriminator(self):
        """

        :return:
        """

        if self.input_width == 1:
            y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, self.y_dim], name='y_reshaped')

            rpeak_input_reshaped = tf.reshape(self.r_peak_placeholder, [-1, 1, 1], name='r_resshaped')

        else:
            y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, 1, self.y_dim], name='y_reshaped')

        with tf.variable_scope("first_mask_concat_layer"):
            if self.input_width == 1:
                reshaped = tf.reshape(self.input_placeholder, [-1, self.input_height, 1])
                mask1 = self.conv_cond_concat_1d(reshaped, y_input_reshaped)
                mask1 = self.conv_cond_concat_1d(mask1, rpeak_input_reshaped)
            else:
                mask1 = self.conv_cond_concat(self.input_placeholder, y_input_reshaped)

        with tf.variable_scope("first_conv_layer"):

            if self.input_width == 1:
                first_conv_layer = tf.layers.conv1d(inputs=mask1,
                                                    filters=self.num_channels + self.y_dim,
                                                    kernel_size=5, padding="same",
                                                    activation=None,
                                                    kernel_initializer=
                                                    tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                    name="first_conv_layer", reuse=tf.AUTO_REUSE)

            else:
                first_conv_layer = tf.layers.conv2d(self.input_placeholder,
                                                    filters=self.num_channels + self.y_dim,
                                                    kernel_size=[5, 5],
                                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                    strides=[2, 2], padding="same", name='first_conv_layer',
                                                    reuse=tf.AUTO_REUSE)
            output_first_conv_layer = self.lrelu(first_conv_layer, name='lrelu_first_layer')

            if self.input_width == 1:
                output_first_conv_layer_mask = self.conv_cond_concat_1d(output_first_conv_layer, y_input_reshaped)
            else:
                output_first_conv_layer_mask = self.conv_cond_concat(output_first_conv_layer, y_input_reshaped)

        with tf.variable_scope("second_conv_layer"):

            if self.input_width == 1:
                second_conv_layer = tf.layers.conv1d(inputs=output_first_conv_layer_mask,
                                                     filters=self.number_of_filters_at_first_layer + self.y_dim,
                                                     kernel_size=5, padding="same",
                                                     activation=None,
                                                     kernel_initializer=
                                                     tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                     name="second_conv_layer", reuse=tf.AUTO_REUSE)
            else:
                second_conv_layer = tf.layers.conv2d(output_first_conv_layer_mask,
                                                     filters=self.number_of_filters_at_first_layer + self.y_dim,
                                                     kernel_size=[5, 5],
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         stddev=0.02),
                                                     strides=[2, 2], padding="same",
                                                     reuse=tf.AUTO_REUSE)

            batch_normalization_second_layer = tf.contrib.layers.batch_norm(second_conv_layer,
                                                                            decay=0.9,
                                                                            updates_collections=None,
                                                                            epsilon=1e-5, scale=True,
                                                                            is_training=True,
                                                                            scope=
                                                                            "batch_normalization_second_"
                                                                            "layer",
                                                                            reuse=tf.AUTO_REUSE)

            output_second_layer = self.lrelu(batch_normalization_second_layer, name='lrelu_second_layer')

            shape_output_second_layer = output_second_layer.get_shape().as_list()
            if self.input_width == 1:

                reshaped_second_layer = tf.reshape(output_second_layer, [tf.shape(self.y_input_placeholder)[0],
                                                                         shape_output_second_layer[1] *
                                                                         shape_output_second_layer[2]])
            else:
                reshaped_second_layer = tf.reshape(output_second_layer, [tf.shape(self.y_input_placeholder)[0],
                                                                         shape_output_second_layer[1] *
                                                                         shape_output_second_layer[2] *
                                                                         shape_output_second_layer[3]])

            concat_second_layer = tf.concat([reshaped_second_layer, self.y_input_placeholder], 1)

        with tf.variable_scope("First_layer_fc"):

            first_linear_layer_before_activation = tf.layers.dense(concat_second_layer, 1024,
                                                                   activation=None,
                                                                   name="fully_connected_first_layer",
                                                                   kernel_initializer=
                                                                   tf.random_normal_initializer(stddev=0.02),
                                                                   reuse=tf.AUTO_REUSE)

            batch_normalization_first_linear = tf.contrib.layers.batch_norm(first_linear_layer_before_activation,
                                                                            decay=0.9,
                                                                            updates_collections=None,
                                                                            epsilon=1e-5, scale=True,
                                                                            is_training=True,
                                                                            scope=
                                                                            "batch_normalization_linear_"
                                                                            "layer",
                                                                            reuse=tf.AUTO_REUSE)

            output_linear_layer = self.lrelu(batch_normalization_first_linear, name='lrelu_linear_first_layer')

            concat_linear_layet = tf.concat([output_linear_layer, self.y_input_placeholder], 1)

        with tf.variable_scope("last_layer_fc"):
            last_linear_layer_before_activation = tf.layers.dense(concat_linear_layet, 1,
                                                                  activation=None,
                                                                  name="fully_connected_last_layer",
                                                                  kernel_initializer=
                                                                  tf.random_normal_initializer(stddev=0.02),
                                                                  reuse=tf.AUTO_REUSE)

            activated_output_last_layer = tf.nn.sigmoid(last_linear_layer_before_activation)

        self.activated_output_last_layer = activated_output_last_layer
        self.last_linear_layer_before_activation = last_linear_layer_before_activation

    def build_vanilla_discriminator(self):
        """

        :return:
        """

        with tf.variable_scope("first_fc_layer"):
            num_of_neurons_first_layer = 128
            first_linear_layer = tf.layers.dense(self.input_placeholder, num_of_neurons_first_layer,
                                                 activation=tf.nn.relu,
                                                 name="fully_connected_first_layer", reuse=tf.AUTO_REUSE)

        with tf.variable_scope("last_layer_fc"):
            last_linear_layer_before_activation = tf.layers.dense(first_linear_layer, 1,
                                                                  activation=None,
                                                                  name="fully_connected_last_layer",
                                                                  reuse=tf.AUTO_REUSE)
            activated_output_last_layer = tf.nn.sigmoid(last_linear_layer_before_activation)

        self.activated_output_last_layer = activated_output_last_layer
        self.last_linear_layer_before_activation = last_linear_layer_before_activation

    def build_conditional_vanilla_discriminator(self):
        """
        Build a vanilla version of conditional discriminator.
        :return:
        """
        with tf.variable_scope("input_layer"):
            inputs = tf.concat([self.input_placeholder, self.y_input_placeholder], 1)

        with tf.variable_scope("first_fc_layer"):
            num_of_neurons_first_layer = 128
            first_linear_layer = tf.layers.dense(inputs, num_of_neurons_first_layer, activation=tf.nn.relu,
                                                 name="fully_connected_first_layer", reuse=tf.AUTO_REUSE)

        with tf.variable_scope("last_layer_fc"):
            last_linear_layer_before_activation = tf.layers.dense(first_linear_layer, 1,
                                                                  activation=None,
                                                                  name="fully_connected_last_layer",
                                                                  reuse=tf.AUTO_REUSE)
            activated_output_last_layer = tf.nn.sigmoid(last_linear_layer_before_activation)

        self.activated_output_last_layer = activated_output_last_layer
        self.last_linear_layer_before_activation = last_linear_layer_before_activation

    def build_conditional_discriminator(self, architecture_file):
        """

        :param architecture_file:
        :return:
        """
        if architecture_file is None:

            if self.input_width == 1:
                y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, self.y_dim], name='y_reshaped')
            else:
                y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, 1, self.y_dim], name='y_reshaped')

            with tf.variable_scope("first_mask_concat_layer"):
                if self.input_width == 1:
                    reshaped = tf.reshape(self.input_placeholder, [-1, self.input_height, 1])
                    mask1 = self.conv_cond_concat_1d(reshaped, y_input_reshaped)
                else:
                    mask1 = self.conv_cond_concat(self.input_placeholder, y_input_reshaped)

            with tf.variable_scope("first_conv_layer"):

                if self.input_width == 1:
                    first_conv_layer = tf.layers.conv1d(inputs=mask1,
                                                        filters=self.num_channels + self.y_dim,
                                                        kernel_size=5, padding="same",
                                                        activation=None,
                                                        kernel_initializer=
                                                        tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                        name="first_conv_layer", reuse=tf.AUTO_REUSE)

                else:
                    first_conv_layer = tf.layers.conv2d(self.input_placeholder,
                                                        filters=self.num_channels + self.y_dim,
                                                        kernel_size=[5, 5],
                                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                        strides=[2, 2], padding="same", name='first_conv_layer',
                                                        reuse=tf.AUTO_REUSE)
                output_first_conv_layer = self.lrelu(first_conv_layer, name='lrelu_first_layer')

                if self.input_width == 1:
                    output_first_conv_layer_mask = self.conv_cond_concat_1d(output_first_conv_layer, y_input_reshaped)
                else:
                    output_first_conv_layer_mask = self.conv_cond_concat(output_first_conv_layer, y_input_reshaped)

            with tf.variable_scope("second_conv_layer"):

                if self.input_width == 1:
                    second_conv_layer = tf.layers.conv1d(inputs=output_first_conv_layer_mask,
                                                         filters=self.number_of_filters_at_first_layer + self.y_dim,
                                                         kernel_size=5, padding="same",
                                                         activation=None,
                                                         kernel_initializer=
                                                         tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                         name="second_conv_layer", reuse=tf.AUTO_REUSE)
                else:
                    second_conv_layer = tf.layers.conv2d(output_first_conv_layer_mask,
                                                         filters=self.number_of_filters_at_first_layer + self.y_dim,
                                                         kernel_size=[5, 5],
                                                         kernel_initializer=tf.truncated_normal_initializer(
                                                             stddev=0.02),
                                                         strides=[2, 2], padding="same",
                                                         reuse=tf.AUTO_REUSE)

                batch_normalization_second_layer = tf.contrib.layers.batch_norm(second_conv_layer,
                                                                                decay=0.9,
                                                                                updates_collections=None,
                                                                                epsilon=1e-5, scale=True,
                                                                                is_training=True,
                                                                                scope=
                                                                                "batch_normalization_second_"
                                                                                "layer",
                                                                                reuse=tf.AUTO_REUSE)

                output_second_layer = self.lrelu(batch_normalization_second_layer, name='lrelu_second_layer')

                shape_output_second_layer = output_second_layer.get_shape().as_list()
                if self.input_width == 1:

                    reshaped_second_layer = tf.reshape(output_second_layer, [tf.shape(self.y_input_placeholder)[0],
                                                                             shape_output_second_layer[1] *
                                                                             shape_output_second_layer[2]])
                else:
                    reshaped_second_layer = tf.reshape(output_second_layer, [tf.shape(self.y_input_placeholder)[0],
                                                                             shape_output_second_layer[1] *
                                                                             shape_output_second_layer[2] *
                                                                             shape_output_second_layer[3]])

                concat_second_layer = tf.concat([reshaped_second_layer, self.y_input_placeholder], 1)

            with tf.variable_scope("First_layer_fc"):

                first_linear_layer_before_activation = tf.layers.dense(concat_second_layer, 1024,
                                                                      activation=None,
                                                                      name="fully_connected_first_layer",
                                                                      kernel_initializer=
                                                                      tf.random_normal_initializer(stddev=0.02),
                                                                      reuse=tf.AUTO_REUSE)

                batch_normalization_first_linear = tf.contrib.layers.batch_norm(first_linear_layer_before_activation,
                                                                                decay=0.9,
                                                                                updates_collections=None,
                                                                                epsilon=1e-5, scale=True,
                                                                                is_training=True,
                                                                                scope=
                                                                                "batch_normalization_linear_"
                                                                                "layer",
                                                                                reuse=tf.AUTO_REUSE)

                output_linear_layer = self.lrelu(batch_normalization_first_linear, name='lrelu_linear_first_layer')

                concat_linear_layet = tf.concat([output_linear_layer,  self.y_input_placeholder], 1)

            with tf.variable_scope("last_layer_fc"):
                last_linear_layer_before_activation = tf.layers.dense(concat_linear_layet, 1,
                                                                      activation=None,
                                                                      name="fully_connected_last_layer",
                                                                      kernel_initializer=
                                                                      tf.random_normal_initializer(stddev=0.02),
                                                                      reuse=tf.AUTO_REUSE)

                activated_output_last_layer = tf.nn.sigmoid(last_linear_layer_before_activation)

            self.activated_output_last_layer = activated_output_last_layer
            self.last_linear_layer_before_activation = last_linear_layer_before_activation

        else:
            #TODO: complete
            pass

    @staticmethod
    def conv_cond_concat_1d(a, b):
        """
        TODO: understand
        :param a:
        :param b:
        :return:
        """
        # a_shapes = a.shape()
        # b_shapes = b.shape()
        return tf.concat([a, b * tf.ones([tf.shape(a)[0], tf.shape(a)[1], tf.shape(b)[2]])], 2)

    @staticmethod
    def conv_cond_concat(x, y):
        """Concatenate conditioning vector on feature map axis."""
        # x_shapes = x.shape()
        # y_shapes = y.shape()
        return tf.concat([
            x, y * tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(y)[3]])], 3)

    def build_regular_dcgan(self, architecture_file):
        """

        :param architecture_file:
        :return:
        """
        if architecture_file is None:
            # Init default architecture:
            with tf.variable_scope("first_conv_layer"):
                if self.input_width == 1:
                    reshaped = tf.reshape(self.input_placeholder, [-1, self.input_height, 1])

                    first_conv_layer = tf.layers.conv1d(inputs=reshaped,
                                                        filters=self.number_of_filters_at_first_layer,
                                                        kernel_size=5, padding="same",
                                                        activation=None,
                                                        kernel_initializer=
                                                        tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                        name="first_conv_layer", reuse=tf.AUTO_REUSE)

                else:
                    first_conv_layer = tf.layers.conv2d(self.input_placeholder,
                                                        filters=self.number_of_filters_at_first_layer,
                                                        kernel_size=[5, 5],
                                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                        strides=[2, 2], padding="same", name='first_conv_layer',
                                                        reuse=tf.AUTO_REUSE)
                output_first_conv_layer = self.lrelu(first_conv_layer, name='lrelu_first_layer')

            with tf.variable_scope("second_conv_layer"):

                if self.input_width  == 1:
                    second_conv_layer = tf.layers.conv1d(inputs=output_first_conv_layer,
                                                         filters=self.number_of_filters_at_first_layer * 2,
                                                         kernel_size=5, padding="same",
                                                         activation=None,
                                                         kernel_initializer=
                                                         tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                         name="second_conv_layer", reuse=tf.AUTO_REUSE)
                else:
                    second_conv_layer = tf.layers.conv2d(output_first_conv_layer,
                                                         filters=self.number_of_filters_at_first_layer
                                                                 * 2, kernel_size=[5, 5],
                                                         kernel_initializer=tf.truncated_normal_initializer(
                                                             stddev=0.02),
                                                         strides=[2, 2], padding="same",
                                                         reuse=tf.AUTO_REUSE)

                batch_normalization_second_layer = tf.contrib.layers.batch_norm(second_conv_layer,
                                                                                decay=0.9,
                                                                                updates_collections=None,
                                                                                epsilon=1e-5, scale=True,
                                                                                is_training=True,
                                                                                scope=
                                                                                "batch_normalization_second_"
                                                                                "layer",
                                                                                reuse=tf.AUTO_REUSE)

                output_second_layer = self.lrelu(batch_normalization_second_layer, name='lrelu_second_layer')

            with tf.variable_scope("third_conv_layer"):

                if self.input_width  == 1:
                    third_conv_layer = tf.layers.conv1d(inputs=output_second_layer,
                                                        filters=self.number_of_filters_at_first_layer * 4,
                                                        kernel_size=5, padding="same",
                                                        activation=None,
                                                        kernel_initializer=
                                                        tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                        name="third_conv_layer", reuse=tf.AUTO_REUSE)
                else:
                    third_conv_layer = tf.layers.conv2d(output_second_layer,
                                                        filters=self.number_of_filters_at_first_layer
                                                                * 4, kernel_size=[5, 5],
                                                        kernel_initializer=tf.truncated_normal_initializer(
                                                            stddev=0.02),
                                                        strides=[2, 2], padding="same", reuse=tf.AUTO_REUSE)

                batch_normalization_third_layer = tf.contrib.layers.batch_norm(third_conv_layer,
                                                                               decay=0.9,
                                                                               updates_collections=None,
                                                                               epsilon=1e-5, scale=True,
                                                                               is_training=True,
                                                                               scope=
                                                                               "batch_normalization_third_"
                                                                               "layer",
                                                                               reuse=tf.AUTO_REUSE)

                output_third_layer = self.lrelu(batch_normalization_third_layer, name='lrelu_third_layer')

            with tf.variable_scope("fourth_conv_layer"):
                if self.input_width  == 1:
                    fourth_conv_layer = tf.layers.conv1d(inputs=output_third_layer,
                                                         filters=self.number_of_filters_at_first_layer * 8,
                                                         kernel_size=5, padding="same",
                                                         activation=None,
                                                         kernel_initializer=
                                                         tf.truncated_normal_initializer(stddev=0.02), strides=2,
                                                         name="fourth_conv_layer", reuse=tf.AUTO_REUSE)
                else:
                    fourth_conv_layer = tf.layers.conv2d(output_third_layer,
                                                         filters=self.number_of_filters_at_first_layer
                                                                 * 8, kernel_size=[5, 5],
                                                         kernel_initializer=tf.truncated_normal_initializer(
                                                             stddev=0.02),
                                                         strides=[2, 2], padding="same", reuse=tf.AUTO_REUSE)

                batch_normalization_fourth_layer = tf.contrib.layers.batch_norm(fourth_conv_layer,
                                                                                decay=0.9,
                                                                                updates_collections=None,
                                                                                epsilon=1e-5, scale=True,
                                                                                is_training=True,
                                                                                scope=
                                                                                "batch_normalization_fourth_"
                                                                                "layer",
                                                                                reuse=tf.AUTO_REUSE)

                output_fourth_layer = self.lrelu(batch_normalization_fourth_layer, name='lrelu_fourth_layer')

            with tf.variable_scope("last_layer_fc"):
                shape_of_output_fourth_layer = output_fourth_layer.get_shape().as_list()
                if self.input_width  == 1:
                    reshaped_last_layer = tf.reshape(output_fourth_layer, [-1, shape_of_output_fourth_layer[1] *
                                                                           shape_of_output_fourth_layer[2]])
                else:
                    reshaped_last_layer = tf.reshape(output_fourth_layer, [-1, shape_of_output_fourth_layer[1] *
                                                                           shape_of_output_fourth_layer[2] *
                                                                           shape_of_output_fourth_layer[3]])

                last_linear_layer_before_activation = tf.layers.dense(reshaped_last_layer, 1,
                                                                      activation=None,
                                                                      name="fully_connected_last_layer",
                                                                      kernel_initializer=
                                                                      tf.random_normal_initializer(stddev=0.02),
                                                                      reuse=tf.AUTO_REUSE)

                activated_output_last_layer = tf.nn.sigmoid(last_linear_layer_before_activation)

            self.activated_output_last_layer = activated_output_last_layer
            self.last_linear_layer_before_activation = last_linear_layer_before_activation

        else:
            # TODO : Init from file.
            pass
            print("Not Implemented yet")

    def run_discriminator(self, session, x_input):
        """
        run the discriminator on an existing session.
        :param session:
        :param x_input : input to the discriminator of shape : [batch_size, self.input_h, self.input_w, self.input_C]
        :return: the output of the session.
        """
        # shape_of_input = x_input.shape
        # assert shape_of_input[1] == self.input_dim
        result_pred = session.run(self.activated_output_last_layer, feed_dict={self.input_placeholder: x_input})
        return result_pred


    @staticmethod
    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x, name=name)

    def create_loss_tensors(self):
        """

        :return:
        """
        with tf.variable_scope("discriminator"):
            with tf.variable_scope(self.discriminator_type + "_loss"):
                if self.discriminator_type == "REAL":
                    # If we feed into the discriminator real image, than the label of the discriminator is 1 == real.
                    # The shape here will be just 1.
                    self.tags_placeholder = tf.ones_like(self.last_linear_layer_before_activation)
                else:
                    self.tags_placeholder = tf.zeros_like(self.last_linear_layer_before_activation)
                diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tags_placeholder,
                                                               logits=self.last_linear_layer_before_activation)
                with tf.name_scope(self.discriminator_type + "_total_loss"):
                    self.cost = tf.reduce_mean(diff)
                self.loss_summary = tf.summary.scalar('cross_entropy_loss' + self.discriminator_type, self.cost)

        if self.discriminator_type == "FAKE":
            # Here we define the loss of the Generator:
            with tf.variable_scope("generator"):
                with tf.variable_scope("loss"):
                    '''
                    We want the generator network to create images that will fool the discriminator. The generator wants
                    the discriminator to output a 1 (positive example). Therefore, we want to compute the loss between 
                    the output of the discriminator when the input comes from the generator and label of 1.
                    '''
                    diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=
                                                                   tf.ones_like(self.last_linear_layer_before_activation),
                                                                   logits=self.last_linear_layer_before_activation)
                    with tf.name_scope(self.discriminator_type + "_total_loss"):
                        self.generator_cost = tf.reduce_mean(diff)
                        self.generator_loss_summary = tf.summary.scalar('cross_entropy_loss_' + self.discriminator_type,
                                                                        self.generator_cost)

            '''
            with tf.variable_scope("fake_loss"):
                self.fake_tags_placeholder = tf.zeros_like(self.last_linear_layer_before_activation)
                diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fake_tags_placeholder,
                                                               logits=self.last_linear_layer_before_activation)
                with tf.name_scope("total_loss"):
                    self.fake_cost = tf.reduce_mean(diff)
                tf.summary.scalar('cross_entropy_loss', self.fake_cost)
            '''

if __name__ == "__main__":

    # 1. Basic Testing for the descriminator :
    '''
    new_discriminator = GenericDiscriminator(input_height=28, input_width=28, number_of_input_channels=1,
                                             number_of_filters_at_first_layer=64, architecture_file=None,
                                             tensor_board_logs_dir='./logs')
    mnist_data = input_data.read_data_sets("mnist_data", one_hot=True)
    train_set = mnist_data.train
    x = train_set.images
    x = x.reshape((55000, 28, 28, 1)).astype(np.float)
    y_tags = train_set.labels
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # reshape_tensor = tf.reshape(input_to_layer, [-1, layer[1], layer[2], layer[3]])
        test_input = x[0:3]
        gen_output = new_discriminator.run_discriminator(sess, test_input)
        print("DONE")
    '''

    # Test on ECG data:
    ecg_discriminator = GenericDiscriminator(input_height=240, input_width=1, number_of_input_channels=1,
                                             number_of_filters_at_first_layer=64, architecture_file=None,
                                             tensor_board_logs_dir='./logs')
    # train_set = ecg_data.train()