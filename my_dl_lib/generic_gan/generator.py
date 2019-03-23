import math
import os
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from my_dl_lib.generic_gan import ops


class GenericGenerator:

    def __init__(self, input_dim, output_height, output_width, output_channels, architecture_file=None,
                 y_labels_dim=None, add_activation_at_end=False, gan_type='dcgan'):
        """
        initialize a new Generator object, the architecture of the generator is determined through the architecture file
        which will be initialized using the generic_neural_network model.
        if the architecture file is None then we will initialize the generator like here:
        https://github.com/carpedm20/DCGAN-tensorflow
        :param input_dim: the second dimension of the input to the generator ( the first is the Batch size )
                x : [batch_size, input_dim]
        :param output_height:
        :param output_width:
        :param output_channels:
        :param architecture_file:
        :param is_conditional_gan: boolean, if true then this is an architecture of conditional gan.
        :param y_labels_dim: dimensions of the label if we are building a conditional gan.
        :param gan_type: one of ['dcgan', 'conditional_dcgan', 'vanilla_gan', 'vanilla_conditional_gan' ]
        """

        if gan_type not in ['dcgan', 'conditional_dcgan', 'vanilla_gan', 'vanilla_conditional_gan', 'ecg_gan']:
            raise AssertionError("undefined gan type", gan_type)

        self.r_peak_placeholder = None
        self.add_activation_at_end = add_activation_at_end
        self.last_layer_generator_tensor = None
        self.output_channels = output_channels
        self.input_dim = input_dim
        self.output_height = output_height
        self.output_width = output_width
        self.y_dim = y_labels_dim
        self.y_input_placeholder = None
        self.x_input_placeholder = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32,
                                                  name="input_placeholder")

        if gan_type == 'dcgan':
            self.build_regular_dcgan(architecture_file)
        elif gan_type == 'conditional_dcgan':
            self.y_input_placeholder = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='y_input_placeholder')
            self.build_conditional_generator(architecture_file)
        elif gan_type == 'vanilla_conditional_gan':
            self.y_input_placeholder = tf.placeholder(tf.float32, shape=[None, self.y_dim], name='y_input_placeholder')
            self.build_conditional_vanilla_generator()
        elif gan_type == 'vanilla_gan':
            self.build_vanilla_gan()
        elif gan_type == 'ecg_gan':
            self.build_ecg_gan()

    def build_ecg_gan(self):
        """

        :return:
        """
        self.r_peak_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='r_peak_placeholder')

        with tf.variable_scope("generator"):

            # Last output:
            s_h, s_w = self.output_height, self.output_width  # 240, 1
            # One and two layers before last output:
            if self.output_width == 1:
                s_w2, s_w4 = 1, 1
            else:
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

            s_h2, s_h4 = int(s_h / 2), int(s_h / 4)  # 120, 60

            with tf.variable_scope("first_layer"):
                if self.output_width == 1:
                    y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, self.y_dim])
                else:
                    y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, 1, self.y_dim])

                z_concat_y = tf.concat([self.x_input_placeholder, self.y_input_placeholder], 1)

                # Now concat with r peak value:
                z_concat_y = tf.concat([z_concat_y, self.r_peak_placeholder], 1)

                num_of_neurons_first_layer = 1024
                first_linear_layer = tf.layers.dense(z_concat_y, num_of_neurons_first_layer, activation=None,
                                                     name="fully_connected_first_layer")

                batch_normalized_first_layer = tf.contrib.layers.batch_norm(first_linear_layer, decay=0.9,
                                                                            updates_collections=None, epsilon=1e-5,
                                                                            scale=True, is_training=True,
                                                                            scope="batch_normalization_first_layer")
                output_first_layer = tf.nn.relu(batch_normalized_first_layer)

            with tf.variable_scope("second_layer"):

                concat_input_second_layer = tf.concat([output_first_layer, self.y_input_placeholder], 1)
                num_of_neurons_second_layer = 64 * 2 * s_h4 * s_w4  # 64 * 2 * 60
                second_linear_layer = tf.layers.dense(concat_input_second_layer, num_of_neurons_second_layer,
                                                      activation=None, name="fully_connected_second_layer")

                if self.output_width == 1:
                    # [batch_size, 60, 64 *2]
                    reshaped_second_layer = tf.reshape(second_linear_layer, [-1, s_h4, 64 * 2])
                    # Mask: TODO: understand operation.
                    # reshaped_second_layer_shape = reshaped_second_layer.shape()
                    mask_layer_2 = y_input_reshaped * tf.ones([tf.shape(reshaped_second_layer)[0],
                                                                tf.shape(reshaped_second_layer)[1], self.y_dim])
                    concat_reshaped_second_layer_with_y = tf.concat([reshaped_second_layer, mask_layer_2], 2)
                else:
                    reshaped_second_layer = tf.reshape(second_linear_layer, [-1, s_h4, s_w4, 64 * 2])
                    # Mask: TODO: understand operation.
                    #reshaped_second_layer_shape = reshaped_second_layer.shape()
                    '''
                    mask_layer_2 = y_input_reshaped * tf.ones([[reshaped_second_layer_shape[0],
                                                                reshaped_second_layer_shape[1],
                                                                reshaped_second_layer_shape[2], self.y_dim]])
                    '''
                    mask_layer_2 = y_input_reshaped * tf.ones([tf.shape(reshaped_second_layer)[0],
                                                                tf.shape(reshaped_second_layer)[1],
                                                                tf.shape(reshaped_second_layer)[2], self.y_dim])
                    concat_reshaped_second_layer_with_y = tf.concat([reshaped_second_layer, mask_layer_2], 3)

            with tf.variable_scope("first_deconv_layer"):
                if self.output_width == 1:
                    # This means 1d-signal generation:
                    first_deconv_layer = self.my_conv1d_transpose(inputs=concat_reshaped_second_layer_with_y,
                                                                  output_shape=[None, s_h2, 64 * 2], filter_size=5,
                                                                  strides_size=2, name='first_deconv_layer')
                else:
                    first_deconv_layer = self.my_conv2d_transpose(inputs=concat_reshaped_second_layer_with_y,
                                             output_shape=[None, s_h2, s_w2, 64 * 2],
                                             filter_height=5, filter_width=5, strides_height=2,
                                             strides_width=2, name='first_deconv_layer')

                batch_normalized_deconv_second_layer = tf.contrib.layers.batch_norm(first_deconv_layer, decay=0.9,
                                                                                   updates_collections=None,
                                                                                   epsilon=1e-5, scale=True,
                                                                                   is_training=True,
                                                                                   scope=
                                                                                   "batch_normalization_"
                                                                                   "deconv_first_layer")
                output_first_deconv_layer = tf.nn.relu(batch_normalized_deconv_second_layer)

            with tf.variable_scope("second_conv_and_concat"):
                if self.output_width == 1:

                    # Mask: TODO: understand operation.
                    # output_first_deconv_layer_shape = output_first_deconv_layer.shape()
                    mask_layer_3 = y_input_reshaped * tf.ones([tf.shape(output_first_deconv_layer)[0],
                                                                tf.shape(output_first_deconv_layer)[1], self.y_dim])
                    concat_3 = tf.concat([output_first_deconv_layer, mask_layer_3], 2)
                else:
                    # Mask: TODO: understand operation.
                    # output_first_deconv_layer_shape = output_first_deconv_layer.shape()
                    mask_layer_3 = y_input_reshaped * tf.ones([tf.shape(output_first_deconv_layer)[0],
                                                                tf.shape(output_first_deconv_layer)[1],
                                                                tf.shape(output_first_deconv_layer)[2], self.y_dim])
                    concat_3 = tf.concat([output_first_deconv_layer, mask_layer_3], 3)

            with tf.variable_scope("last_deconv_layer"):

                if self.output_width == 1:
                    # This means 1d-signal generation:
                    last_deconv_layer = self.my_conv1d_transpose(inputs=concat_3, output_shape=
                    [None, self.output_height, 1], filter_size=5, strides_size=2, name='last_deconv_layer')
                else:
                    last_deconv_layer = self.my_conv2d_transpose(inputs=concat_3,
                                                                 output_shape=[None, self.output_height,
                                                                               self.output_width,
                                                                               self.output_channels],
                                                                 filter_height=5, filter_width=5, strides_height=2,
                                                                 strides_width=2, name='last_deconv_layer',
                                                                 padding="SAME")
                if self.add_activation_at_end:
                    output_last_layer = tf.nn.sigmoid(last_deconv_layer)
                else:
                    output_last_layer = last_deconv_layer
                self.last_layer_generator_tensor = output_last_layer

    def build_vanilla_gan(self):
        """

        :return:
        """
        with tf.variable_scope("generator"):

            with tf.variable_scope("fully_connected_layer"):
                num_of_neurons_first_layer = 128
                first_linear_layer = tf.layers.dense(self.x_input_placeholder, num_of_neurons_first_layer,
                                                     activation=tf.nn.relu,
                                                     name="fully_connected_first_layer")

            with tf.variable_scope("fully_connected_last_layer"):
                last_layer = tf.layers.dense(first_linear_layer, self.output_height, activation=None,
                                             name="fully_connected_last_layer")

                if self.add_activation_at_end:
                    output_last_layer = tf.nn.sigmoid(last_layer)
                else:
                    output_last_layer = last_layer
                self.last_layer_generator_tensor = output_last_layer

    def build_conditional_vanilla_generator(self):
        """
        Build a vanilla version of conditional generator.
        :return: None
        """
        with tf.variable_scope("generator"):
            with tf.variable_scope("input_layer_concatenated"):
                z_concat_y = tf.concat([self.x_input_placeholder, self.y_input_placeholder], 1)

            with tf.variable_scope("fully_connected_layer"):
                num_of_neurons_first_layer = 128
                first_linear_layer = tf.layers.dense(z_concat_y, num_of_neurons_first_layer, activation=tf.nn.relu,
                                                     name="fully_connected_first_layer")

            with tf.variable_scope("fully_connected_last_layer"):
                last_layer = tf.layers.dense(first_linear_layer, self.output_height, activation=None,
                                             name="fully_connected_last_layer")

                if self.add_activation_at_end:
                    output_last_layer = tf.nn.sigmoid(last_layer)
                else:
                    output_last_layer = last_layer
                self.last_layer_generator_tensor = output_last_layer

    def build_conditional_generator(self, architecture_file):
        """
        Build a conditional generator.
        :param architecture_file:
        :return:
        """

        with tf.variable_scope("generator"):

            if architecture_file is None:
                # Determine the structure of the network according to the desired output :

                # Last output:
                s_h, s_w = self.output_height, self.output_width  # 240, 1
                # One and two layers before last output:
                if self.output_width == 1:
                    s_w2, s_w4 = 1, 1
                else:
                    s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)  # 120, 60

                with tf.variable_scope("first_layer"):
                    if self.output_width == 1:
                        y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, self.y_dim])
                    else:
                        y_input_reshaped = tf.reshape(self.y_input_placeholder, [-1, 1, 1, self.y_dim])

                    z_concat_y = tf.concat([self.x_input_placeholder, self.y_input_placeholder], 1)

                    num_of_neurons_first_layer = 1024
                    first_linear_layer = tf.layers.dense(z_concat_y, num_of_neurons_first_layer, activation=None,
                                                         name="fully_connected_first_layer")

                    batch_normalized_first_layer = tf.contrib.layers.batch_norm(first_linear_layer, decay=0.9,
                                                                                updates_collections=None, epsilon=1e-5,
                                                                                scale=True, is_training=True,
                                                                                scope="batch_normalization_first_layer")
                    output_first_layer = tf.nn.relu(batch_normalized_first_layer)

                with tf.variable_scope("second_layer"):

                    concat_input_second_layer = tf.concat([output_first_layer, self.y_input_placeholder], 1)
                    num_of_neurons_second_layer = 64 * 2 * s_h4 * s_w4  # 64 * 2 * 60
                    second_linear_layer = tf.layers.dense(concat_input_second_layer, num_of_neurons_second_layer,
                                                          activation=None, name="fully_connected_second_layer")

                    if self.output_width == 1:
                        # [batch_size, 60, 64 *2]
                        reshaped_second_layer = tf.reshape(second_linear_layer, [-1, s_h4, 64 * 2])
                        # Mask: TODO: understand operation.
                        # reshaped_second_layer_shape = reshaped_second_layer.shape()
                        mask_layer_2 = y_input_reshaped * tf.ones([tf.shape(reshaped_second_layer)[0],
                                                                    tf.shape(reshaped_second_layer)[1], self.y_dim])
                        concat_reshaped_second_layer_with_y = tf.concat([reshaped_second_layer, mask_layer_2], 2)
                    else:
                        reshaped_second_layer = tf.reshape(second_linear_layer, [-1, s_h4, s_w4, 64 * 2])
                        # Mask: TODO: understand operation.
                        #reshaped_second_layer_shape = reshaped_second_layer.shape()
                        '''
                        mask_layer_2 = y_input_reshaped * tf.ones([[reshaped_second_layer_shape[0],
                                                                    reshaped_second_layer_shape[1],
                                                                    reshaped_second_layer_shape[2], self.y_dim]])
                        '''
                        mask_layer_2 = y_input_reshaped * tf.ones([tf.shape(reshaped_second_layer)[0],
                                                                    tf.shape(reshaped_second_layer)[1],
                                                                    tf.shape(reshaped_second_layer)[2], self.y_dim])
                        concat_reshaped_second_layer_with_y = tf.concat([reshaped_second_layer, mask_layer_2], 3)

                with tf.variable_scope("first_deconv_layer"):
                    if self.output_width == 1:
                        # This means 1d-signal generation:
                        first_deconv_layer = self.my_conv1d_transpose(inputs=concat_reshaped_second_layer_with_y,
                                                                      output_shape=[None, s_h2, 64 * 2], filter_size=5,
                                                                      strides_size=2, name='first_deconv_layer')
                    else:
                        first_deconv_layer = self.my_conv2d_transpose(inputs=concat_reshaped_second_layer_with_y,
                                                 output_shape=[None, s_h2, s_w2, 64 * 2],
                                                 filter_height=5, filter_width=5, strides_height=2,
                                                 strides_width=2, name='first_deconv_layer')



                    batch_normalized_deconv_second_layer = tf.contrib.layers.batch_norm(first_deconv_layer, decay=0.9,
                                                                                       updates_collections=None,
                                                                                       epsilon=1e-5, scale=True,
                                                                                       is_training=True,
                                                                                       scope=
                                                                                       "batch_normalization_"
                                                                                       "deconv_first_layer")
                    output_first_deconv_layer = tf.nn.relu(batch_normalized_deconv_second_layer)

                with tf.variable_scope("second_conv_and_concat"):
                    if self.output_width == 1:

                        # Mask: TODO: understand operation.
                        # output_first_deconv_layer_shape = output_first_deconv_layer.shape()
                        mask_layer_3 = y_input_reshaped * tf.ones([tf.shape(output_first_deconv_layer)[0],
                                                                    tf.shape(output_first_deconv_layer)[1], self.y_dim])
                        concat_3 = tf.concat([output_first_deconv_layer, mask_layer_3], 2)
                    else:
                        # Mask: TODO: understand operation.
                        # output_first_deconv_layer_shape = output_first_deconv_layer.shape()
                        mask_layer_3 = y_input_reshaped * tf.ones([tf.shape(output_first_deconv_layer)[0],
                                                                    tf.shape(output_first_deconv_layer)[1],
                                                                    tf.shape(output_first_deconv_layer)[2], self.y_dim])
                        concat_3 = tf.concat([output_first_deconv_layer, mask_layer_3], 3)

                with tf.variable_scope("last_deconv_layer"):

                    if self.output_width == 1:
                        # This means 1d-signal generation:
                        last_deconv_layer = self.my_conv1d_transpose(inputs=concat_3, output_shape=
                        [None, self.output_height, 1], filter_size=5, strides_size=2, name='last_deconv_layer')
                    else:
                        last_deconv_layer = self.my_conv2d_transpose(inputs=concat_3,
                                                                      output_shape=[None, self.output_height,
                                                                                    self.output_width,
                                                                                    self.output_channels],
                                                                      filter_height=5, filter_width=5, strides_height=2,
                                                                      strides_width=2, name='last_deconv_layer', padding="SAME")
                    if self.add_activation_at_end:
                        output_last_layer = tf.nn.sigmoid(last_deconv_layer)
                    else:
                        output_last_layer = last_deconv_layer
                    self.last_layer_generator_tensor = output_last_layer

    def build_regular_dcgan(self, architecture_file):
        """
        Build a gan which is not condional gan.
        :return:
        """
        with tf.variable_scope("generator"):
            if architecture_file is None:
                # Determine the structure of the network according to the desired output :
                third_deconv_layer_h, third_deconv_layer_w = int(math.ceil(float(self.output_height) / float(2))), \
                                                              int(math.ceil(float(self.output_width) / float(2)))

                second_deconv_layer_h, second_deconv_layer_w = int(math.ceil(float(third_deconv_layer_h) / float(2))), \
                                                              int(math.ceil(float(third_deconv_layer_w) / float(2)))

                first_deconv_layer_h, first_deconv_layer_w = int(math.ceil(float(second_deconv_layer_h) / float(2))), \
                                                               int(math.ceil(float(second_deconv_layer_w) / float(2)))

                first_linear_layer_h, first_linear_layer_w = int(math.ceil(float(first_deconv_layer_h) / float(2))), \
                                                             int(math.ceil(float(first_deconv_layer_w) / float(2)))

                # Add noise to first layer:
                '''
                print("### Added Gaussian Noise to first layer of generator ##################")
                gaussian_noise_layer = self.gaussian_noise_layer(self.x_input_placeholder, 0.2)
                print("######################################################################")
                '''
                # Init the default generator:
                with tf.variable_scope("first_layer"):
                    first_linear_layer = tf.layers.dense(self.x_input_placeholder, 1024 * first_linear_layer_h *
                                                         first_linear_layer_w, activation=None,
                                                         name="fully_connected_first_layer")

                    if self.output_width == 1:
                        reshaped_first_layer = tf.reshape(first_linear_layer, [-1, first_linear_layer_h, 1024])
                    else:
                        reshaped_first_layer = tf.reshape(first_linear_layer, [-1, first_linear_layer_h,
                                                                               first_linear_layer_w, 1024])

                    # TODO: Read about batch normalization :
                    batch_normalized_first_layer = tf.contrib.layers.batch_norm(reshaped_first_layer, decay=0.9,
                                                                                updates_collections=None, epsilon=1e-5,
                                                                                scale=True, is_training=True,
                                                                                scope="batch_normalization_first_layer")

                    output_first_layer = tf.nn.relu(batch_normalized_first_layer)

                with tf.variable_scope("first_deconv_layer"):

                    if self.output_width == 1:
                        # This means 1d-signal generation:
                        first_deconv_layer = self.my_conv1d_transpose(inputs=output_first_layer, output_shape=
                        [None, first_deconv_layer_h, 512], filter_size=5, strides_size=2, name='first_deconv_layer')

                    else:
                        first_deconv_layer = self.my_conv2d_transpose(inputs=output_first_layer,
                                                                      output_shape=[None, first_deconv_layer_h,
                                                                                    first_deconv_layer_w, 512],
                                                                      filter_height=5, filter_width=5, strides_height=2,
                                                                      strides_width=2, name='first_deconv_layer')

                        '''
                        first_deconv_layer= tf.layers.conv2d_transpose(inputs=output_first_layer, filters=512,
                                                                       kernel_size=[5, 5], strides=(2, 2), padding='same',
                                                                       activation=None, kernel_initializer=
                                                                       tf.random_normal_initializer(stddev=stddev))
                        '''

                    batch_normalized_deconv_first_layer = tf.contrib.layers.batch_norm(first_deconv_layer, decay=0.9,
                                                                                       updates_collections=None,
                                                                                       epsilon=1e-5, scale=True,
                                                                                       is_training=True,
                                                                                       scope=
                                                                                       "batch_normalization_"
                                                                                       "deconv_first_layer")
                    output_second_layer = tf.nn.relu(batch_normalized_deconv_first_layer)

                with tf.variable_scope("second_deconv_layer"):

                    if self.output_width == 1:
                        # This means 1d-signal generation:
                        second_deconv_layer = self.my_conv1d_transpose(inputs=output_second_layer, output_shape=
                        [None, second_deconv_layer_h, 256], filter_size=5, strides_size=2, name='second_deconv_layer')
                    else:
                        second_deconv_layer = self.my_conv2d_transpose(inputs=output_second_layer,
                                                                      output_shape=[None, second_deconv_layer_h,
                                                                                    second_deconv_layer_w, 256],
                                                                      filter_height=5, filter_width=5, strides_height=2,
                                                                      strides_width=2, name='second_deconv_layer')
                    '''
                    second_deconv_layer = tf.layers.conv2d_transpose(inputs=output_second_layer, filters=256,
                                                                    kernel_size=[5, 5], strides=(2, 2), padding='same',
                                                                    activation=None, kernel_initializer=
                                                                    tf.random_normal_initializer(stddev=stddev))
                    '''

                    batch_normalized_deconv_second_layer = tf.contrib.layers.batch_norm(second_deconv_layer, decay=0.9,
                                                                                       updates_collections=None,
                                                                                       epsilon=1e-5, scale=True,
                                                                                       is_training=True,
                                                                                       scope=
                                                                                       "batch_normalization_"
                                                                                       "deconv_first_layer")
                    output_third_layer = tf.nn.relu(batch_normalized_deconv_second_layer)

                with tf.variable_scope("third_deconv_layer"):

                    if self.output_width == 1:
                        # This means 1d-signal generation:
                        third_deconv_layer = self.my_conv1d_transpose(inputs=output_third_layer, output_shape=
                        [None, third_deconv_layer_h, 128], filter_size=5, strides_size=2, name='third_deconv_layer')

                    else:
                        third_deconv_layer = self.my_conv2d_transpose(inputs=output_third_layer,
                                                                       output_shape=[None, third_deconv_layer_h,
                                                                                     third_deconv_layer_w, 128],
                                                                       filter_height=5, filter_width=5, strides_height=2,
                                                                       strides_width=2, name='third_deconv_layer')
                    '''
                    third_deconv_layer = tf.layers.conv2d_transpose(inputs=output_third_layer, filters=128,
                                                                     kernel_size=[5, 5], strides=(2, 2), padding='same',
                                                                     activation=None, kernel_initializer=
                                                                     tf.random_normal_initializer(stddev=stddev))
                    '''

                    batch_normalized_deconv_third_layer = tf.contrib.layers.batch_norm(third_deconv_layer,
                                                                                        decay=0.9,
                                                                                        updates_collections=None,
                                                                                        epsilon=1e-5, scale=True,
                                                                                        is_training=True,
                                                                                        scope=
                                                                                        "batch_normalization_"
                                                                                        "deconv_first_layer")
                    output_fourth_layer = tf.nn.relu(batch_normalized_deconv_third_layer)

                with tf.variable_scope("last_deconv_layer"):

                    if self.output_width == 1:
                        # This means 1d-signal generation:
                        last_deconv_layer = self.my_conv1d_transpose(inputs=output_fourth_layer, output_shape=
                        [None, self.output_height, 1], filter_size=5, strides_size=2, name='last_deconv_layer')
                    else:
                        last_deconv_layer = self.my_conv2d_transpose(inputs=output_fourth_layer,
                                                                      output_shape=[None, self.output_height,
                                                                                    self.output_width, self.output_channels],
                                                                      filter_height=5, filter_width=5, strides_height=2,
                                                                      strides_width=2, name='last_deconv_layer')
                    '''
                    last_deconv_layer = tf.layers.conv2d_transpose(inputs=output_fourth_layer, filters=output_channels,
                                                                    kernel_size=[5, 5], strides=(2, 2), padding='same',
                                                                    activation=None, kernel_initializer=
                                                                    tf.random_normal_initializer(stddev=stddev))
                    '''

                    if self.add_activation_at_end:
                        output_last_layer = tf.nn.tanh(last_deconv_layer)
                    else:
                        output_last_layer = (last_deconv_layer)
                    self.last_layer_generator_tensor = output_last_layer

            else:
                pass  # TODO: parse architecture file.

    @ staticmethod
    def gaussian_noise_layer(input_layer, std):
        """
        Add noise to the generated input layer.
        :param input_layer:
        :param std:
        :return:
        """
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    @ staticmethod
    def my_conv2d_transpose(inputs, output_shape, filter_height, filter_width, strides_height, strides_width,
                            name, padding='SAME'):
        """
        Creates a tensor which performs a transope convolution.
        :param inputs: input - [batch_size, h, w, c] to the transpose convolution layer
        :param output_shape: the desired shape of the output after performing the transpose convolution
        :param filter_height: height of the filter
        :param filter_width: width of the filter
        :param strides_height: how much strides to jump up and down
        :param strides_width: how much strides to jump lfet and right
        :param name: name of the operation
        :return:
        """

        number_of_filters = output_shape[-1]  # The last dim of the desired output shape comes from the number of
        # filters in the deconvolution layer.

        # Work around for the problem that the batch size is None :
        batch_size = tf.shape(inputs)[0]
        output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])

        input_number_of_channels = inputs.get_shape().as_list()[-1]

        # 1. Define the filter: [height, width, output_channels, in_channels]
        w_filter = tf.get_variable('w', [filter_height, filter_width, number_of_filters, input_number_of_channels],
                                   initializer=tf.random_normal_initializer(stddev=0.02))

        # 2. Define the transpose convolution layer :
        conv2d_transpose_layer = tf.nn.conv2d_transpose(inputs, filter=w_filter, output_shape=output_shape, name=name, padding=padding,
                                                        strides=[1, strides_height, strides_width, 1],)

        # 3. Add Bias : ( number of biases should be as the number of filters )
        biases = tf.get_variable('biases', [number_of_filters], initializer=tf.constant_initializer(0.0))

        # 4. Add the Biases to the deconv :
        conv2d_transpose_layer_with_biases = tf.nn.bias_add(conv2d_transpose_layer, biases)

        return conv2d_transpose_layer_with_biases

    def my_conv1d_transpose(self, inputs, output_shape, filter_size, strides_size, name):
        """

        :param inputs: input - [batch_size, input_dim, input_channles] to the transpose convolution layer
        :param output_shape: the desired shape of the output after performing the transpose convolution
        :param filter_size:
        :param strides_size:
        :param name:
        :return:
        """
        number_of_filters = output_shape[-1]  # The last dim of the desired output shape comes from the number of
        # filters in the deconvolution layer.

        # Work around for the problem that the batch size is None :
        batch_size = tf.shape(inputs)[0]
        output_shape = tf.stack([batch_size, output_shape[1], output_shape[2]])

        input_number_of_channels = inputs.get_shape().as_list()[-1]

        # 1. Define the filter: [filter_length, in_channels]
        w_filter = tf.get_variable('w', [filter_size, number_of_filters, input_number_of_channels],
                                   initializer=tf.random_normal_initializer(stddev=0.02))

        # 2. Define the transpose convolution layer :
        conv1d_transpose_layer = ops.conv1d_transpose(inputs, filter=w_filter, output_shape=output_shape,
                                                      stride=strides_size, name=name)

        # 3. Add Bias : ( number of biases should be as the number of filters )
        biases = tf.get_variable('biases', [number_of_filters], initializer=tf.constant_initializer(0.0))

        # 4. Add the Biases to the deconv :
        conv1d_transpose_layer_with_biases = tf.nn.bias_add(conv1d_transpose_layer, biases)

        return conv1d_transpose_layer_with_biases

    def run_generator(self, session, x_input, y_samples=None):
        """
        run the generator on an existing session.
        :param session:
        :param x_input : input to the generator of shape : [batch_size, self.input_dim]
        :return: the output of the session.
        """
        shape_of_input = x_input.shape
        assert shape_of_input[1] == self.input_dim
        if y_samples is None:
            result_image = session.run(self.last_layer_generator_tensor, feed_dict={self.x_input_placeholder: x_input})
        else:
            result_image = session.run(self.last_layer_generator_tensor, feed_dict={self.x_input_placeholder: x_input, self.y_input_placeholder:y_samples})
        return result_image

    def test_on_mnist(self):
        """

        :return:
        """
        # z_dimensions = 100
        # test_z = np.random.normal(-1, 1, [1, z_dimensions])


    @staticmethod
    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def generate_data_from_trained_generator(num_of_samples_to_generate, meta_file_name,
                                         is_conditional_gan=False, label_index=None):
    """

    :param num_of_samples_to_generate:
    :param meta_file_name:
    :param meta_file_directory:
    :return:
    """
    print("Restoring graph Generator:")
    print(os.path.isfile(meta_file_name + '.meta'))
    # Generate input points:
    generator_input = np.random.uniform(-1, 1, [num_of_samples_to_generate, 100])
    # Restore the graph:
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file_name + '.meta')
        # lateset_checkpoint = tf.train.latest_checkpoint(meta_file_directory)
        # print("Latest checkpoint ", meta_file_name + '.meta')
        saver.restore(sess, save_path=meta_file_name)

        graph = tf.get_default_graph()
        if not is_conditional_gan:
            # for op in graph.get_operations():
            #    print(str(op.name))
            input_placeholder = graph.get_tensor_by_name('input_placeholder:0')
            last_layer_tensor = graph.get_tensor_by_name('generator/last_deconv_layer/last_deconv_layer:0')
            # last_layer_tensor = graph.get_tensor_by_name('generator/fully_connected_last_layer/fully_connected_last_layer/BiasAdd:0')
            data = sess.run(last_layer_tensor, feed_dict={input_placeholder: generator_input})
        else:
            # Case it is a conditional GAN:
            # print([n.name for n in graph.as_graph_def().node])
            for op in graph.get_operations():
                print(str(op.name))

            input_placeholder = graph.get_tensor_by_name('input_placeholder:0')
            y_placeholder = graph.get_tensor_by_name('y_input_placeholder:0')
            last_layer_tensor = graph.get_tensor_by_name('generator/fully_connected_last_layer/fully_connected_first_layer/BiasAdd:0')

            y_dim = y_placeholder.get_shape().as_list()[1]
            y_samples = [0 if i != label_index else 1 for i in range(y_dim)]
            y_samples = np.array([y_samples for _ in range(num_of_samples_to_generate)])
            data = sess.run(last_layer_tensor, feed_dict={input_placeholder: generator_input,
                                                                 y_placeholder: y_samples})

        print("Generated data shape: ", data.shape)
        data = data.reshape(num_of_samples_to_generate, 216)
        print("Creating example of one sample:")
        '''
        plt.figure()
        plt.plot(data[0])
        plt.savefig(os.path.join('example_from_gan.png'))
        '''
        return data


if __name__ == "__main__":

    # 1. Regular Testing :
    '''
    new_generator = GenericGenerator(100, 28, 28, 1, architecture_file=None, tensor_board_logs_dir='./logs')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_input = np.random.normal(-1, 1, [1, 100])
        gen_output = new_generator.run_generator(sess, test_input)
        print("DONE")
    '''
    # 2. Test on ECG Data :
    '''
    ecg_generator = GenericGenerator(100, 240, 1, 1, architecture_file=None, tensor_board_logs_dir='./logs')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_input = np.random.normal(-1, 1, [1, 100])
        gen_output = ecg_generator.run_generator(sess, test_input)
        print("DONE")
    '''

    generate_data_from_trained_generator(5, os.path.join('saved_models', 'gan', 'R', 'dcgan_batch_64_iters_50000', 'iter_22300'))