import logging
import os
import shutil
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from scipy import interp
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from tensorflow.examples.tutorials.mnist import input_data

from my_dl_lib import generic_dataset


class GenericNeuralNetwork:
    def __init__(self, architecture_file=None, tensor_board_logs_dir='./logs', model_save_dir='./saved_models',
                 dropout_prob=1, mode='TRAIN'):
        """
        C'tor gets as input a text file which describes the architecture of the network and create a new network with
        initialized weights according to that design.
        Note:
        - When adding conv layer, we assume stride of filter is always one, and padding is applied to save the same dims
          max pooling is always perfomed, and is configured with stride of 2.

        :param architecture_file: a text file which describes the network.
        file format:
        > model_name: X
        > input_size: d
        > number_of_layers: N
        > layer 1:
        > number_of_neurons: l1
        > activation_function: sigmoid/tanh/relu
        > layer 2:
        > ....
        > cast_input_to_image:
        > width: 22
        > height: 33
        > channels: 3
        > ....
        > conv layer: 25
        > number_of_filters: 32
        > filter_size:
        > rows: 5
        > columns: 5
        > channels: 3
        > activation_function: relu/sigmoid/tanh
        > ...
        > 1D conv layer: 70
        > number_of_filters: 10
        > filter_length: 5
        > channels: 1
        > activation_function: relu/sigmoid/tanh
        > ...
        > SVM layer: 71
        > kernel_type: must be one of ( ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable )
        > degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
        > penalty: string, ‘l1’ or ‘l2’ (default=’l2’)
        > ...
        > layer with additional data 101:
        > additional data input size: 40
        > number_of_neurons: 10
        > activation_function: softmax
        > ...
        > layer N:
        > number_of_neurons: lN
        > activation_function: sigmoid/tanh/relu

        :param tensor_board_logs_dir: where to locate the events file create from the summary operations.
        """
        if mode == 'TEST':
            # if we are in test mode we assume that we need to load an existing model. therefore the object will be
            # built empty
            self.session = None
            return
        if architecture_file is None:
            print("Error: Empty architecture file")
            return

        # 1. Clear and re-create the folders of the saved model and logs:
        print("Checking exsistance of logs directory", tensor_board_logs_dir)
        if os.path.exists(tensor_board_logs_dir):
            print("Logs directory exists, clearing its contents")
            for the_file in os.listdir(tensor_board_logs_dir):
                file_path = os.path.join(tensor_board_logs_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            print("Logs directory doesn't exist, creating it")
            os.makedirs(tensor_board_logs_dir)

        print("Checking exsistance of saved models directory", model_save_dir)
        if os.path.exists(model_save_dir):
            '''
            print("Saved model directory exists, clearing its contents")
            for the_file in os.listdir(model_save_dir):
                file_path = os.path.join(model_save_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
            '''
        else:
            print("Saved model directory doesn't exist, creating it")
            os.makedirs(model_save_dir)

        # Case we add feature at the last layer.
        self.aditional_data_placeholder = None

        self.dropout_prob = dropout_prob
        self.keep_prob = None  # this will be the placeholder.
        self.total_number_of_runs = 0
        self.merged_summaries = None
        self.train_writer = None
        self.val_writer = None
        self.test_writer = None

        self.model_save_dir = model_save_dir
        self.tensor_board_logs_dir = tensor_board_logs_dir
        # Clean old logs:
        self.delete_logs_folder()

        self.binary_predictions_tensor = None
        self.session = None
        self.logger = logging.getLogger(GenericNeuralNetwork.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.model_name = None
        self.input_dim = None
        self.num_of_layers = None
        self.soft_max_tensor = None
        self.layers_info = []

        self.parse_architecture_file(architecture_file)
        self.x_input_tf_placeholder, self.neural_network_tensor_tf = self.build_network_with_tf()

        # Statistical attributes:
        self.number_of_classes = self.layers_info[-1][1]
        self.number_of_positives_seen_during_train = 0
        self.number_negatives_seen_during_train = 0

    def parse_architecture_file(self, file_path):
        """
        opens the text file which represents the network archirecure and parses it.
        :param file_path: path to the text file
        :return:
        """
        architecture_file = file_path
        with open(architecture_file, "r") as f:
            data = f.readlines()

        i = 0
        while i < len(data):
            line = data[i]
            words = line.split()

            if i == 0:
                # First line:
                assert words[0] == "model_name:"
                self.model_name = words[1]
                i += 1
            elif i == 1:
                assert words[0] == "input_size:"
                if len(words) == 2:
                    self.input_dim = int(words[1])
                    self.number_of_dimensions_of_input = 1
                else:
                    self.number_of_dimensions_of_input = len(words) - 1
                    self.input_dim = ([int(x) for x in words[1:]])
                i += 1
            elif i == 2:
                assert words[0] == "number_of_layers:"
                self.num_of_layers = int(words[1])
                i += 1
            else:
                if words[0] == "layer":
                    assert words[0] == "layer"  # + str((i+1) / 3)
                    # current_layer = int(words[1])
                    if words[1] == 'with':
                        assert words[2] == 'additional'
                        assert words[3] == 'data'
                        i += 1
                        line = data[i]
                        words = line.split()

                        assert words[0] == 'additional'
                        assert words[1] == 'data'
                        assert words[2] == 'input'
                        assert words[3] == 'size:'
                        size_of_additional_data = int(words[4])
                        i += 1
                        line = data[i]
                        words = line.split()
                        assert words[0] == "number_of_neurons:"
                        num_of_neurons = int(words[1])
                        i += 1
                        line = data[i]
                        words = line.split()
                        assert words[0] == "activation_function:"
                        activation_function = words[1]
                        self.layers_info.append(('danse_with_additional_data', num_of_neurons, activation_function,
                                                 size_of_additional_data))
                        i += 1
                    else:
                        i += 1
                        line = data[i]
                        words = line.split()
                        assert words[0] == "number_of_neurons:"
                        num_of_neurons = int(words[1])
                        i += 1
                        line = data[i]
                        words = line.split()
                        assert words[0] == "activation_function:"
                        activation_function = words[1]
                        self.layers_info.append(('danse', num_of_neurons, activation_function))
                        i += 1
                elif words[0] == 'conv':
                    assert words[0] == 'conv'
                    assert words[1] == 'layer'
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'number_of_filters:'
                    number_of_filters = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'filter_size:'
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'rows:'
                    num_of_filter_rows = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'columns:'
                    num_of_filter_columns = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'channels:'
                    num_of_filter_channels = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == "activation_function:"
                    activation_function = words[1]
                    self.layers_info.append(('conv', number_of_filters, num_of_filter_rows, num_of_filter_columns,
                                             num_of_filter_channels, activation_function))
                    i += 1

                elif words[0] == 'cast_input_to_image:':
                    assert words[0] == 'cast_input_to_image:'
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'width:'
                    width = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'height:'
                    height = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'channels:'
                    channels = int(words[1])
                    self.layers_info.append(('reshape', width, height, channels))
                    i += 1
                elif words[0] == '1D':
                    assert words[0] == '1D'
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'number_of_filters:'
                    number_of_filters = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'filter_length:'
                    filter_length = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'channels:'
                    num_of_filter_channels = int(words[1])
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == "activation_function:"
                    activation_function = words[1]
                    self.layers_info.append(('1D_conv', number_of_filters, filter_length,
                                             num_of_filter_channels, activation_function))
                    i += 1
                else:
                    # SVM case:
                    assert words[0] == 'SVM'
                    i += 1
                    line = data[i]
                    words = line.split()
                    assert words[0] == 'kernel_type:'
                    kernel_type = words[1]
                    assert kernel_type in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
                    i += 1
                    line = data[i]
                    words = line.split()
                    kernel_degree = 1
                    if kernel_type == 'poly':
                        assert words[0] == 'degree:'
                        kernel_degree = int(words[1])
                        i += 1
                        line = data[i]
                        words = line.split()
                        assert words[0] == 'penalty:'
                        penalty = float(words[1])
                    elif kernel_type == 'linear':
                        assert words[0] == 'penalty:'
                        penalty = str(words[1])
                    else:
                        assert words[0] == 'penalty:'
                        penalty = float(words[1])
                    self.layers_info.append(('SVM', kernel_type, penalty, kernel_degree))
                    i += 1

    @staticmethod
    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def build_network_with_tf(self):
        """
        builds a tensorflow graph that represents the network.
        :return: tensorflow graph components
        """
        input_dim = self.input_dim
        if self.number_of_dimensions_of_input == 1:
            x = tf.placeholder(shape=[None, input_dim], dtype=tf.float32, name="input_placeholder")
        else:
            input_dims = [None] + input_dim
            x = tf.placeholder(shape=input_dims, dtype=tf.float32, name="input_placeholder")
        input_to_layer = x

        for layer_number, layer in enumerate(self.layers_info):
            if layer[0] == 'danse':
                with tf.name_scope("danse_layer_" + str(layer_number)):
                    # Check if it came from a conv layer:
                    shape_of_input = input_to_layer.get_shape().as_list()
                    # shape_of_input = tf.shape(input_to_layer)
                    if len(shape_of_input) == 4:
                        input_to_layer = tf.reshape(input_to_layer, [-1, shape_of_input[1] * shape_of_input[2] *
                                                                     shape_of_input[3]])
                    elif len(shape_of_input) == 3:
                        input_to_layer = tf.reshape(input_to_layer, [-1, shape_of_input[1] * shape_of_input[2]])

                    # Check if this is the last layer:
                    if layer_number == len(self.layers_info) - 1:
                        with tf.name_scope('dropout'):
                            self.keep_prob = tf.placeholder(tf.float32)
                            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
                            dropped = tf.nn.dropout(input_to_layer, self.keep_prob)
                            print("Added dropout of size: ", self.dropout_prob)
                        # Output layer: (without softmax)
                        # Adding a name scope ensures logical grouping of the layers in the graph:
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(dropped, layer[1], activation=None,
                                             name="layer_" + str(layer_number))
                        # Getting the weights and biases from the dense layer:
                        # This Variable will hold the state of the weights for the layer
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + "/kernel:0")
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    elif layer[2] == 'sigmoid':
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.sigmoid,
                                             name="layer_" + str(layer_number))
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    elif layer[2] == 'tanh':
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.tanh,
                                             name="layer_" + str(layer_number))
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    elif layer[2] == 'relu':
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.relu,
                                             name="layer_" + str(layer_number))

                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    else:
                        assert layer[2] == 'softmax'
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.softmax,
                                             name="layer_" + str(layer_number))
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    input_to_layer = nn
                    if layer_number == len(self.layers_info) - 1:
                        return x, nn

            elif layer[0] == 'danse_with_additional_data':
                # ' num_of_neurons, activation_function,size_of_additional_data'
                num_of_nuerons = layer[1]
                size_of_aditional_data = layer[3]
                with tf.name_scope("danse_layer_with_aditional_data_" + str(layer_number)):

                    # Build placeholder for the aditional data:
                    self.aditional_data_placeholder = tf.placeholder(shape=[None, size_of_aditional_data], dtype=tf.float32,
                                                                name="aditional_data_placeholder")

                    # Check if it came from a conv layer:
                    shape_of_input = input_to_layer.get_shape().as_list()
                    if len(shape_of_input) == 4:
                        # RGB case:
                        # [batch_size, w,h c = 3]  --> [batch_size, w*h*c]:
                        input_to_layer = tf.reshape(input_to_layer, [-1, shape_of_input[1] * shape_of_input[2] *
                                                                     shape_of_input[3]])
                    elif len(shape_of_input) == 3:
                        # Gray level case:
                        # [batch_size, w,h]  --> [batch_size, w*h]:
                        input_to_layer = tf.reshape(input_to_layer, [-1, shape_of_input[1] * shape_of_input[2]])

                    # Concatenate input_to_layer with additional_data_place_holder:
                    input_to_layer = tf.concat([input_to_layer, self.aditional_data_placeholder], 1)

                    # Check if this is the last layer, and if so add a dropout layer:
                    if layer_number == len(self.layers_info) - 1:
                        with tf.name_scope('dropout'):
                            self.keep_prob = tf.placeholder(tf.float32)
                            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
                            dropped = tf.nn.dropout(input_to_layer, self.keep_prob)
                            print("Added dropout of size: ", self.dropout_prob)
                        # Output layer: (without softmax)
                        # Adding a name scope ensures logical grouping of the layers in the graph:
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(dropped, num_of_nuerons, activation=None,
                                             name="layer_" + str(layer_number))
                        # Getting the weights and biases from the dense layer:
                        # This Variable will hold the state of the weights for the layer
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + "/kernel:0")
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    elif layer[2] == 'sigmoid':
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.sigmoid,
                                             name="layer_" + str(layer_number))
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    elif layer[2] == 'tanh':
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.tanh,
                                             name="layer_" + str(layer_number))
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    elif layer[2] == 'relu':
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.relu,
                                             name="layer_" + str(layer_number))

                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    else:
                        assert layer[2] == 'softmax'
                        # with tf.name_scope("layer_" + str(layer_number)):
                        nn = tf.layers.dense(input_to_layer, layer[1], activation=tf.nn.softmax,
                                             name="layer_" + str(layer_number))
                        with tf.name_scope('weights'):
                            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                                + '/kernel:0')
                            self.variable_summaries(weights)
                        with tf.name_scope('biases'):
                            bias = tf.get_default_graph().get_tensor_by_name(os.path.split(nn.name)[0].split("/")[1]
                                                                             + '/bias:0')
                            self.variable_summaries(bias)

                    input_to_layer = nn
                    if layer_number == len(self.layers_info) - 1:
                        return x, nn


            elif layer[0] == 'conv':
                assert layer[0] == 'conv'
                shape_of_input_tensor = input_to_layer.get_shape().as_list()
                # shape_of_input_tensor = tf.shape(input_to_layer)
                if len(shape_of_input_tensor) == 4:
                    # This means that the input if with a shape : [batch_size, width, height, channels]
                    # Which means that it came from a convolution layer.
                    # For Testing purpose:
                    if layer[5] == "sigmoid":
                        with tf.name_scope('conv2d_' + str(layer_number)):
                            conv_layer = tf.layers.conv2d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=[layer[2], layer[3]],
                                                          padding="same",
                                                          activation=tf.nn.sigmoid,
                                                          name="conv2d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv2d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv2d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)

                    elif layer[5] == "tanh":
                        with tf.name_scope('conv2d_' + str(layer_number)):
                            conv_layer = tf.layers.conv2d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=[layer[2], layer[3]],
                                                          padding="same",
                                                          activation=tf.nn.tanh,
                                                          name="conv2d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv2d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv2d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)

                    elif layer[5] == "relu":
                        with tf.name_scope('conv2d_' + str(layer_number)):
                            conv_layer = tf.layers.conv2d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=[layer[2], layer[3]],
                                                          padding="same",
                                                          activation=tf.nn.relu,
                                                          name="conv2d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv2d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv2d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)

                    else:
                        assert layer[5] == "softmax"
                        with tf.name_scope('conv2d_' + str(layer_number)):
                            conv_layer = tf.layers.conv2d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=[layer[2], layer[3]],
                                                          padding="same",
                                                          activation=tf.nn.softmax,
                                                          name="conv2d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv2d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv2d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)

                    pool_layer = tf.layers.max_pooling2d(inputs=conv_layer, pool_size=2, strides=2)
                    input_to_layer = pool_layer
                    if layer_number == len(self.layers_info) - 1:
                        return x, conv_layer
                else:
                    print("Error - shape to conv layer must be 4!")
                    return

            elif layer[0] == 'reshape':
                assert layer[0] == 'reshape'
                reshape_tensor = tf.reshape(input_to_layer, [-1, layer[1], layer[2], layer[3]])
                input_to_layer = reshape_tensor
                if layer_number == len(self.layers_info) - 1:
                    return x, reshape_tensor

            elif layer[0] == '1D_conv':
                assert layer[0] == '1D_conv'
                shape_of_input_tensor = input_to_layer.get_shape().as_list()
                if len(shape_of_input_tensor) == 2:
                    # This means that shape is of the form : [batch_size, length]
                    # TODO: I assume here that the 1-d signal always has 1 channel.
                    reshape_tensor = tf.reshape(input_to_layer, [-1,
                                                                 shape_of_input_tensor[1], 1])
                    input_to_layer = reshape_tensor
                    shape_of_input_tensor = input_to_layer.get_shape().as_list()
                if len(shape_of_input_tensor) == 3:
                    # This means that the input is with a shape : [batch_size, length, channels]
                    # Which means that it came from a convolution layer.
                    # For Testing purpose:
                    if layer[4] == "sigmoid":
                        with tf.name_scope('conv1d_' + str(layer_number)):
                            conv_layer = tf.layers.conv1d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=layer[2],
                                                          padding="same",
                                                          activation=tf.nn.sigmoid,
                                                          name="conv1d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv1d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv1d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)

                    elif layer[4] == "tanh":
                        with tf.name_scope('conv1d_' + str(layer_number)):
                            conv_layer = tf.layers.conv1d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=layer[2],
                                                          padding="same",
                                                          activation=tf.nn.tanh,
                                                          name="conv1d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv1d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv1d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)

                    elif layer[4] == "relu":
                        with tf.name_scope('conv1d_' + str(layer_number)):
                            conv_layer = tf.layers.conv1d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=layer[2],
                                                          padding="same",
                                                          activation=tf.nn.relu,
                                                          name="conv1d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv1d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv1d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)
                    else:
                        assert layer[4] == "softmax"
                        with tf.name_scope('conv1d_' + str(layer_number)):
                            conv_layer = tf.layers.conv1d(inputs=input_to_layer,
                                                          filters=layer[1],
                                                          kernel_size=layer[2],
                                                          padding="same",
                                                          activation=tf.nn.softmax,
                                                          name="conv1d_layer_" + str(layer_number))
                            with tf.name_scope("filter_weights"):
                                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           'conv1d_layer_' + str(layer_number) + '/kernel')[0]
                                self.variable_summaries(kernel)
                            with tf.name_scope("filter_biases"):
                                bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'conv1d_layer_' + str(layer_number) + '/bias')[0]
                                self.variable_summaries(bias)

                    pool_layer = tf.layers.max_pooling1d(inputs=conv_layer, pool_size=2, strides=2)
                    input_to_layer = pool_layer
                    if layer_number == len(self.layers_info) - 1:
                        return x, conv_layer
                else:
                    print("Error - shape to conv layer must be 3!")
                    return

            else:
                assert layer[0] == 'SVM'

    def get_shape_of_input(self):
        """
        returns the dimensions of the expected input into the network
        :return: the dimensions of the expected input into the network
        """
        return self.input_dim

    def forward_pass_single_example_tf(self, x):
        """
        feeds vector x into the network and returns the y prediction vector.
        also returns the predicted class.
        :param x: vector x with same dimensions as the expected input to the network
        :return: vector y - the output from the network.
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y = sess.run(self.neural_network_tensor_tf, feed_dict={self.x_input_tf_placeholder: x})
            predicted_class = np.argmax(y, axis=1)
            print("predicted class: %d", predicted_class)
            return y

    def calculate_loss_tf(self, loss_type, unbalanced_ratio=None):
        """

        :param loss_type:
        :return:
        """
        output_shape = self.get_shape_of_output()
        tags_placeholder = tf.placeholder(tf.float32, shape=[None, output_shape], name="tags_placeholder")
        with tf.name_scope("loss"):
            if loss_type == "cross_entropy":
                """
                Notes: 
                We could have used here :
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(tags_placeholder * tf.log(self.neural_network_tensor_tf), 
                    reduction_indices=[1]))
                First, tf.log computes the logarithm of each element of y (element wise). 
                Next, we multiply each element of y_tag with the corresponding element of tf.log(y) (element-wise).
                Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] 
                parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.
                Note that in the source code, we don't use this formulation, because it is numerically unstable. Instead, 
                we apply tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits (e.g., we call 
                softmax_cross_entropy_with_logits , because this more numerically stable function internally computes the 
                softmax activation. 
                """
                if unbalanced_ratio is not None:
                    cost = self.calculate_weighted_loss(tags_placeholder, unbalanced_ratio)
                else:
                    diff = tf.nn.softmax_cross_entropy_with_logits(labels=tags_placeholder,
                                                                   logits=self.neural_network_tensor_tf)
                    with tf.name_scope("total_loss"):
                        cost = tf.reduce_mean(diff)
                    tf.summary.scalar('cross_entropy_loss', cost)
            elif loss_type == "mean_square":
                if unbalanced_ratio is not None:
                    # TODO: add support.
                    cost = tf.reduce_mean(tf.squared_difference(self.neural_network_tensor_tf, tags_placeholder))
                else:
                    with tf.name_scope("total_loss"):
                        cost = tf.reduce_mean(tf.squared_difference(self.neural_network_tensor_tf, tags_placeholder),
                                              name='cost_tensor')
                    tf.summary.scalar('mean_square_loss', cost)
            else:
                print("Invalid loss type.")
                return
        '''
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print("total_cost_per_batch: ")
            print(sess.run(cost, feed_dict={self.x_input_tf_placeholder: x_batch, tags_placeholder: y_tags}))
        '''
        return cost, tags_placeholder

    def calculate_weighted_loss(self, labels, coefficients):
        """

        :param logits:
        :param labels:
        :param num_classes:
        :param coefficients:
        :return:
        """
        # logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = self.neural_network_tensor_tf + epsilon

        # consturct one-hot label array
        # label_flat = tf.reshape(labels, (-1, 1))
        # labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), coefficients), reduction_indices=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # tf.add_to_collection('losses', cross_entropy_mean)
        # loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return cross_entropy_mean

    def get_shape_of_output(self):
        return self.layers_info[-1][1]

    def train_tf_with_tf_datasets(self, train_dataset, length_of_train, length_of_val, validation_dataset, num_of_iterations,
                                  batch_size, optimizer_type, loss_type):
        """

        :param train_dataset:
        :param validation_dataset:
        :param num_iterations:
        :param batch_size:
        :param loss_type:
        :return:
        """
        # with tf.device("/gpu:0"):
        cost, tags_placeholder = self.calculate_loss_tf(loss_type, None)
        # TODO: We are assuming here that activation function of last layer is always softmax.
        self.soft_max_tensor = tf.nn.softmax(self.neural_network_tensor_tf, name="last_soft_max_tensor")
        self.binary_predictions_tensor = tf.argmax(self.soft_max_tensor, 1, name="binary_predictions")

        if optimizer_type == "gradient_descent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        elif optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(1e-4)
        else:
            # TODO: add more optimizers.
            print("Invalid optimizer")
            return

        with tf.name_scope("train"):
            updates = optimizer.minimize(cost, name="gradient_update_tensor")

        if loss_type == "cross_entropy":
            with tf.name_scope("accuracy"):
                with tf.name_scope("correct_prediction"):
                    correct_prediction = tf.equal \
                        (tf.argmax(self.soft_max_tensor, 1), tf.argmax(tags_placeholder, 1),
                         name="num_of_correct_predictions_tensor")
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy_tensor")
            tf.summary.scalar("accuracy", accuracy)

        # Create a saver object which will save all the variables
        saver = tf.train.Saver()

        # with tf.Session() as sess:
        # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        sess = tf.Session()
        # Merge all the summaries and write them out:
        self.merged_summaries = tf.summary.merge_all()

        # Create Writer object to visualize the graph later:
        self.train_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/train', sess.graph)
        self.val_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/val')
        self.test_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/test')

        # Initialize weights:
        init = tf.global_variables_initializer()
        sess.run(init)
        train_dataset = train_dataset.batch(batch_size)  # .shuffle(buffer_size=length_of_train)
        train_iterator = train_dataset.make_initializable_iterator()
        next_train_batch = train_iterator.get_next()
        sess.run(train_iterator.initializer)

        validation_dataset = validation_dataset.batch(300)
        validation_iterator = validation_dataset.make_initializable_iterator()
        next_validation_batch = validation_iterator.get_next()

        # data_iterator = generic_dataset.GenericDataSetIterator(x_train, y_train)
        for i in range(num_of_iterations):
            try:
                if self.aditional_data_placeholder is None:
                    x_batch, y_batch = sess.run(next_train_batch)
                else:
                    additional_data_batch, x_batch, y_batch = sess.run(next_train_batch)
            except tf.errors.OutOfRangeError:
                print("End of epoch")
                sess.run(train_iterator.initializer)
                if self.aditional_data_placeholder is None:
                    x_batch, y_batch = sess.run(next_train_batch)
                else:
                    additional_data_batch, x_batch, y_batch = sess.run(next_train_batch)

            # positives = [1 for x in y_batch if x == (1, 0)]
            if self.aditional_data_placeholder is None:
                summary, _ = sess.run([self.merged_summaries, updates],
                                      feed_dict={self.x_input_tf_placeholder: x_batch, tags_placeholder: y_batch,
                                                 self.keep_prob: self.dropout_prob})
            else:
                summary, _ = sess.run([self.merged_summaries, updates],
                                      feed_dict={self.x_input_tf_placeholder: x_batch, tags_placeholder: y_batch,
                                                 self.keep_prob: self.dropout_prob,
                                                 self.aditional_data_placeholder: additional_data_batch})

            self.train_writer.add_summary(summary, i)
            if i % 10 == 0:

                if loss_type == "cross_entropy":
                    if self.aditional_data_placeholder is None:
                            train_accuracy = sess.run(accuracy,
                                                      feed_dict={self.x_input_tf_placeholder: x_batch,
                                                                 tags_placeholder: y_batch,
                                                                 self.keep_prob: 1})
                    else:
                        train_accuracy = sess.run(accuracy,
                                                  feed_dict={self.x_input_tf_placeholder: x_batch,
                                                             tags_placeholder: y_batch,
                                                             self.keep_prob: 1,
                                                             self.aditional_data_placeholder: additional_data_batch})
                    print('Step %d, training accuracy %g' % (i, train_accuracy))

                    print("Calculating validation accuracy, Validation length is: ", length_of_val)
                    sess.run(validation_iterator.initializer)
                    val_acc = []
                    val_iter = 0
                    while True:
                        try:
                            val_iter += 1
                            print("Validation iteration: ", val_iter)
                            if self.aditional_data_placeholder is None:
                                val_x_batch, val_y_batch = sess.run(next_validation_batch)
                                # assert len(val_x_batch) == length_of_val
                                val_accuracy = sess.run(accuracy, feed_dict={self.x_input_tf_placeholder: val_x_batch,
                                                                              tags_placeholder: val_y_batch,
                                                                              self.keep_prob: 1})
                            else:
                                val_add_data_batch, val_x_batch, val_y_batch = sess.run(next_validation_batch)
                                # assert len(val_x_batch) == length_of_val
                                val_accuracy = sess.run(accuracy, feed_dict={self.x_input_tf_placeholder: val_x_batch,
                                                                             tags_placeholder: val_y_batch,
                                                                             self.keep_prob: 1,
                                                                             self.aditional_data_placeholder:
                                                                                 val_add_data_batch})
                            val_acc.append(val_accuracy)
                            # print('step %d, val accuracy %g' % (i, val_accuracy))
                            # self.val_writer.add_summary(summary, i)
                        except tf.errors.OutOfRangeError:
                            print("End of val")
                            print('step %d, val accuracy %g' % (i, (float(sum(val_acc))) / float(len(val_acc))))
                            # sess.run(validation_iterator.initializer)
                            break

                    '''
                    summary, val_accuracy = sess.run([self.merged_summaries, accuracy],
                                                     feed_dict=
                                                     {self.x_input_tf_placeholder: x_val, tags_placeholder: y_val,
                                                      self.keep_prob: 1})
                    print('step %d, val accuracy %g' % (i, val_accuracy))
                    self.val_writer.add_summary(summary, i)
                    '''
                elif loss_type == "mean_square":
                    train_cost = sess.run(cost, feed_dict={self.x_input_tf_placeholder: x_batch,
                                                           tags_placeholder: y_batch, self.keep_prob: 1})

                    print('step %d, training cost %g' % (i, train_cost))
                    '''
                    summary, val_cost = sess.run([self.merged_summaries, cost], feed_dict=
                    {self.x_input_tf_placeholder: x_val, tags_placeholder: y_val, self.keep_prob: 1})
                    print('step %d, val accuracy %g' % (i, val_cost))
                    self.val_writer.add_summary(summary, i)
                    '''
            if i % 100 == 0:
                saver.save(sess, os.path.join(self.model_save_dir, self.model_name + 'iter_' + str(i)),
                           global_step=num_of_iterations)

        # Now, save the graph
        # saver.save(sess, './' + self.model_name, global_step=num_of_iterations)
        # saver.save(sess, './' + self.model_name)
        self.total_number_of_runs = num_of_iterations
        self.session = sess
        return sess

    def train_tf(self, x_train, y_train, x_val, y_val, num_of_iterations, batch_size, optimizer_type, loss_type,
                 unbalanced_ratio=None):
        """
        :param x_data:
        :param y_tags:
        :param num_of_iterations:
        :param batch_size:
        :param optimizer_type:
        :param loss_type:
        :param unbalanced_ratio: allows one to trade off recall and precision by up- or down-weighting the cost of a
                positive error relative to a negative error.
        :return:
        """
        cost, tags_placeholder = self.calculate_loss_tf(loss_type, unbalanced_ratio)
        # TODO: We are assuming here that activation function of last layer is always softmax.
        self.soft_max_tensor = tf.nn.softmax(self.neural_network_tensor_tf, name="last_soft_max_tensor")
        self.binary_predictions_tensor = tf.argmax(self.soft_max_tensor, 1, name="binary_predictions")
        # self.cost_tensor = cost
        # self. tags_placeholder = tags_placeholder
        if optimizer_type == "gradient_descent":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        elif optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(1e-4)
        else:
            # TODO: add more optimizers.
            print("Invalid optimizer")
            return

        with tf.name_scope("train"):
            updates = optimizer.minimize(cost, name="gradient_update_tensor")

        if loss_type == "cross_entropy":
            with tf.name_scope("accuracy"):
                with tf.name_scope("correct_prediction"):
                    correct_prediction = tf.equal \
                        (tf.argmax(self.soft_max_tensor, 1), tf.argmax(tags_placeholder, 1),
                         name="num_of_correct_predictions_tensor")
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy_tensor")
            tf.summary.scalar("accuracy", accuracy)

        '''
        class_prediction = tf.argmax(self.neural_network_tensor_tf, 1, name="class_precition")
        ground_truth_classes = tf.argmax(tags_placeholder, 1, name="class_precition")
        '''
        # Create a saver object which will save all the variables
        saver = tf.train.Saver()

        # with tf.Session() as sess:
        sess = tf.Session()

        # Merge all the summaries and write them out:
        self.merged_summaries = tf.summary.merge_all()

        # Create Writer object to visualize the graph later:
        self.train_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/train', sess.graph)
        self.val_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/val')
        self.test_writer = tf.summary.FileWriter(self.tensor_board_logs_dir + '/test')

        # Initialize weights:
        init = tf.global_variables_initializer()
        sess.run(init)

        data_iterator = generic_dataset.GenericDataSetIterator(x_train, y_train)
        for i in range(num_of_iterations):
            x_batch, y_batch = data_iterator.next_batch(batch_size)
            # positives = [1 for x in y_batch if x == (1, 0)]
            summary, _ = sess.run([self.merged_summaries, updates],
                                  feed_dict={self.x_input_tf_placeholder: x_batch, tags_placeholder: y_batch,
                                             self.keep_prob: self.dropout_prob})

            self.train_writer.add_summary(summary, i)
            if i % 10 == 0:
                '''
                summary, train_accuracy = sess.run([self.merged_summaries, accuracy],
                                                   feed_dict=
                                                   {self.x_input_tf_placeholder: x_batch, tags_placeholder: y_batch})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                '''
                if loss_type == "cross_entropy":
                    train_accuracy = sess.run(accuracy,
                                              feed_dict={self.x_input_tf_placeholder: x_batch, tags_placeholder: y_batch,
                                                         self.keep_prob: 1})
                    print('step %d, training accuracy %g' % (i, train_accuracy))

                    summary, val_accuracy = sess.run([self.merged_summaries, accuracy],
                                                     feed_dict=
                                                     {self.x_input_tf_placeholder: x_val, tags_placeholder: y_val,
                                                      self.keep_prob: 1})
                    print('step %d, val accuracy %g' % (i, val_accuracy))
                    self.val_writer.add_summary(summary, i)

                elif loss_type == "mean_square":
                    train_cost = sess.run(cost, feed_dict={self.x_input_tf_placeholder: x_batch,
                                                           tags_placeholder: y_batch, self.keep_prob: 1})

                    print('step %d, training cost %g' % (i, train_cost))

                    summary, val_cost = sess.run([self.merged_summaries, cost], feed_dict=
                    {self.x_input_tf_placeholder: x_val, tags_placeholder: y_val, self.keep_prob: 1})
                    print('step %d, val accuracy %g' % (i, val_cost))
                    self.val_writer.add_summary(summary, i)

            if i % 100 == 0:
                saver.save(sess, os.path.join(self.model_save_dir, self.model_name + 'iter_' + str(i)), global_step=num_of_iterations)

        # Now, save the graph
        # saver.save(sess, './' + self.model_name, global_step=num_of_iterations)
        # saver.save(sess, './' + self.model_name)
        self.total_number_of_runs = num_of_iterations
        self.session = sess
        return sess

    def train_existing_net_tf(self, x_train, y_train, x_val, y_val, num_of_iterations, batch_size):
        """

        :param x_data:
        :param y_tags:
        :param num_of_iterations:
        :param batch_size:
        :return:
        """
        # with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('./' + self.model_name + '.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        restored_updates = graph.get_operation_by_name("train/gradient_update_tensor")
        tags_placeholder = graph.get_tensor_by_name("tags_placeholder:0")
        accuracy = graph.get_tensor_by_name("accuracy/accuracy/accuracy_tensor:0")
        data_iterator = generic_dataset.GenericDataSetIterator(x_train, y_train)
        for i in range(num_of_iterations):
            x_batch, y_batch = data_iterator.next_batch(batch_size)
            summary, _ = self.session.run([self.merged_summaries, restored_updates],
                                          feed_dict={self.x_input_tf_placeholder: x_batch, tags_placeholder: y_batch,
                                                     self.keep_prob: 0.7})

            self.train_writer.add_summary(summary, i + self.total_number_of_runs)

            if i % 10 == 0:
                train_accuracy = self.session.run(accuracy,
                                                  feed_dict={
                                                   self.x_input_tf_placeholder: x_batch, tags_placeholder: y_batch,
                                                      self.keep_prob: 1})
                print('step %d, training accuracy %g' % (i, train_accuracy))

                summary, val_accuracy = self.session.run([self.merged_summaries, accuracy],
                                                         feed_dict=
                                                         {self.x_input_tf_placeholder: x_val, tags_placeholder: y_val,
                                                          self.keep_prob: 1})
                print('step %d, val accuracy %g' % (i, val_accuracy))
                self.val_writer.add_summary(summary, i + self.total_number_of_runs)

        self.total_number_of_runs += num_of_iterations
        # Now, save the graph

        # saver.save(sess, './' + self.model_name)

    def eval_accuracy(self, x_test, y_test):
        """
        Calculate the accuracy over a given test set. the function will return the number true predicitions of the test
        set. This function is only valid for classification problems.
        :param x_test: The test set, a numpy ndarray with dims: nXd where n is the namber of samples and d is the
        dimension of each sample.
        :param y_test: True labels of the test set. numpy array of size n, where n is the number of samples.
        :return: number of true predictions over the test set / number of samples in the test set.
        """
        assert self.session is not None
        graph = tf.get_default_graph()
        tags_placeholder = graph.get_tensor_by_name("tags_placeholder:0")
        accuracy_tensor = graph.get_tensor_by_name("accuracy/accuracy/accuracy_tensor:0")

        test_accuracy = self.session.run(accuracy_tensor, feed_dict={
            self.x_input_tf_placeholder: x_test, tags_placeholder: y_test, self.keep_prob: 1})
        print("Test accuracy:", test_accuracy)
        # print(sess.run(self.full_predictions_tensor, feed_dict={self.x_input_tf_placeholder: x_test}))
        # self.close_session()
        return test_accuracy

    def eval_lost(self, x_test, y_test):
        """
        Calculate the lost over a test set. The function will return the mean cost over the test set.
        :param x_test: The test set, a numpy ndarray with dims: nXd where n is the namber of samples and d is the
        dimension of each sample.
        :param y_test: True labels of the test set. numpy array of size n, where n is the number of samples.
        :return: The mean cost over the test set.
        """
        assert self.session is not None
        graph = tf.get_default_graph()
        tags_placeholder = graph.get_tensor_by_name("tags_placeholder:0")
        cost_tensor = graph.get_tensor_by_name("loss/total_loss/cost_tensor:0")

        test_cost = self.session.run(cost_tensor, feed_dict={
            self.x_input_tf_placeholder: x_test, tags_placeholder: y_test, self.keep_prob: 1})
        print("Test cost:", test_cost)
        # print(sess.run(self.full_predictions_tensor, feed_dict={self.x_input_tf_placeholder: x_test}))
        # self.close_session()
        return test_cost

    def get_binary_predictions(self, x_data, one_hot=True):
        """
        Feeds all the examples in x_data to the network and returns a vector of binary predictions (not prbabilites),
        that is, with the class that got the highest probability. i.e [1,0,3,1,] ...
        :param x_data: matrix of examples : Number_of_examples X d
        :param one_hot: if True return each prediction as a one hot vector.
        :return:
        """

        if one_hot:
            one_hot_preds = tf.one_hot(self.binary_predictions_tensor, self.get_shape_of_output())
            # For debuging:
            # print(self.session.run(self.neural_network_tensor_tf, feed_dict={self.x_input_tf_placeholder: x_data,
            #                                                                 self.keep_prob: 1}))
            binary_preds = self.session.run(one_hot_preds, feed_dict={self.x_input_tf_placeholder: x_data,
                                                                      self.keep_prob: 1})
        else:
            binary_preds = self.session.run(self.binary_predictions_tensor, feed_dict=
            {self.x_input_tf_placeholder: x_data, self.keep_prob: 1})

        return binary_preds

    def get_probability_predictions(self, x_data):
        """
        Feeds all the examples in x_data to the network and returns a vector of full predictions i.e prbabilites,
        ...
        :param x_data: matrix of examples : Number_of_examples X d
        :param one_hot: if True return each prediction as a one hot vector.
        :return:
        """


        probability_preds = self.session.run(self.soft_max_tensor, feed_dict={self.x_input_tf_placeholder: x_data,
                                                                              self.keep_prob: 1})
        # print(probability_preds)
        return probability_preds

    def get_probability_predictions_of_tf_dataset_and_aditional_data(self, test_dataset, meta_file_name=None):
        """

        :param test_dataset:
        :param meta_file_name:
        :return:
        """
        sess = tf.Session()
        test_dataset = test_dataset.batch(100)  # .shuffle(buffer_size=length_of_train)
        test_iterator = test_dataset.make_initializable_iterator()
        next_test_batch = test_iterator.get_next()
        sess.run(test_iterator.initializer)

        if meta_file_name is None:
            # TODO: FIX.
            probability_preds = self.session.run(self.soft_max_tensor, feed_dict={self.x: test_dataset})
            # print(probability_preds)
        else:
            assert self.session is None  # just make sure session is closed.
            print("Restoring graph:")
            if not os.path.isfile(meta_file_name + '.meta'):
                raise AssertionError("Meta file %s doesn't exists" % meta_file_name)

            saver = tf.train.import_meta_graph(meta_file_name + '.meta')
            saver.restore(sess, save_path=meta_file_name)
            graph = tf.get_default_graph()
            input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
            probability_tensor = graph.get_tensor_by_name("last_soft_max_tensor:0")
            dropout_placeholder = graph.get_tensor_by_name("danse_layer_with_aditional_data_3/dropout/Placeholder:0")
            aditional_data_place_holder = graph.get_tensor_by_name("danse_layer_with_aditional_data_3/aditional_data_"
                                                                   "placeholder:0")

            probabiliity_arr = np.array([])
            flag = 0
            test_iter = 0
            while True:
                try:
                    test_iter += 1
                    print("run on Test set, iteration: ", test_iter)
                    additional_data_batch, test_x_batch, test_y_batch = sess.run(next_test_batch)

                    probability_preds = sess.run(probability_tensor,
                                            feed_dict={input_placeholder: test_x_batch, dropout_placeholder: 1,
                                                       aditional_data_place_holder: additional_data_batch})
                    if flag == 0:
                        probabiliity_arr = probability_preds
                        flag = 1
                    else:
                        probabiliity_arr = np.concatenate((probabiliity_arr, probability_preds), axis=0)

                except tf.errors.OutOfRangeError:
                    print("End of Test, First ten elements:")
                    print(probabiliity_arr[:10])
                    break
            sess.close()
            return probabiliity_arr

    def get_probability_predictions_of_tf_dataset(self, test_dataset, meta_file_name=None):
        """

        :param test_dataset:
        :return:
        """

        sess = tf.Session()
        test_dataset = test_dataset.batch(100)  # .shuffle(buffer_size=length_of_train)
        test_iterator = test_dataset.make_initializable_iterator()
        next_test_batch = test_iterator.get_next()
        sess.run(test_iterator.initializer)

        if meta_file_name is None:
            # TODO: FIX.
            probability_preds = self.session.run(self.soft_max_tensor, feed_dict={self.x: test_dataset})
            # print(probability_preds)
        else:
            assert self.session is None  # just make sure session is closed.
            print("Restoring graph:")
            if not os.path.isfile(meta_file_name + '.meta'):
                raise AssertionError("Meta file %s doesn't exists" % meta_file_name)

            saver = tf.train.import_meta_graph(meta_file_name + '.meta')
            saver.restore(sess, save_path=meta_file_name)
            graph = tf.get_default_graph()
            input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
            probability_tensor = graph.get_tensor_by_name("last_soft_max_tensor:0")
            dropout_placeholder = graph.get_tensor_by_name("danse_layer_3/dropout/Placeholder:0")

            # probability_preds = sess.run(probability_tensor, feed_dict={input_placeholder: x_data})
            probabiliity_arr = np.array([])
            flag = 0
            test_iter = 0
            while True:
                try:
                    test_iter += 1
                    print("run on Test set, iteration: ", test_iter)
                    test_x_batch, test_y_batch = sess.run(next_test_batch)
                    # assert len(val_x_batch) == length_of_val
                    probability_preds = sess.run(probability_tensor,
                                            feed_dict={input_placeholder: test_x_batch, dropout_placeholder: 1})
                    if flag == 0:
                        probabiliity_arr = probability_preds
                        flag = 1
                    else:
                        probabiliity_arr = np.concatenate((probabiliity_arr, probability_preds), axis=0)
                        print(probabiliity_arr.shape)
                    # probabiliity_arr.append(probability_preds)
                    # print('step %d, val accuracy %g' % (i, val_accuracy))
                    # self.val_writer.add_summary(summary, i)
                except tf.errors.OutOfRangeError:
                    print("End of Test")
                    print(probabiliity_arr)
                    # sess.run(validation_iterator.initializer)
                    break

            return probabiliity_arr


    @staticmethod
    def calculate_statistics(test_ground_truth, predictions, positive_name, negative_name, positive_index=0):
        """

        :param test_ground_truth: list of one-hot vectors with the ground truths : [(0,1), (0,1), ...(1,0),...]
        :param predictions: list of predictions that came from the network: [(0,1), (1,0), ...]
        :param positive_name: class name of the positives
        :param negative_name : class name of the negatives
        :param positive_index: the index of the class which is "positive"
        :return: number of positives (P) - among the labels, how much are positive.
                 number of negatives (N) - among the labels, how much are negative.
                 true positive (TP) -  among the data which was identified as positive how many are actualy positive.
                 false positive (FP) - among the data which was identified as positive how many are not positive.
                 true negative (TN) - among the data which was identified as negative how many are actually negative.
                 false negative (FN) - among the data which was identified as negative how many are not negative.
                 true positive rate (TPR) - TP/P = TP/(TP + FN)
                 true negative rate (TNR) - TN / N = TN/ (TN + FP)
                 false positive rate (FPR)
                 true positive rate (TPR)
        """
        statistic_dict = {}
        assert len(predictions) == len(test_ground_truth)
        idx_predicted_positives = []
        idx_predicted_negatives = []
        for i, e in enumerate(predictions):
            if e[positive_index] == 1:
                idx_predicted_positives.append(i)
            else:
                idx_predicted_negatives.append(i)

        num_false_positives = 0
        num_true_positives = 0

        for i in idx_predicted_positives:
            if test_ground_truth[i][positive_index] == 1:
                num_true_positives += 1
            else:
                num_false_positives += 1

        assert num_false_positives + num_true_positives == len(idx_predicted_positives)
        # tf.logging.info("Number of false positives (FP): %d" % num_false_positives)
        # tf.logging.info("Number of true positives (TP): %d" % num_true_positives)

        num_true_negative = 0
        num_false_negative = 0
        for i in idx_predicted_negatives:
            if test_ground_truth[i][positive_index] == 0:
                num_true_negative += 1
            else:
                num_false_negative += 1

        num_of_positives = 0
        num_of_negatives = 0
        for i, e in enumerate(test_ground_truth):
            if test_ground_truth[i][positive_index] == 1:
                num_of_positives += 1
            else:
                num_of_negatives += 1

        assert num_of_positives == num_true_positives + num_false_negative
        assert num_of_negatives == num_true_negative + num_false_positives

        statistic_dict["P"] = num_of_positives
        statistic_dict["N"] = num_of_negatives
        statistic_dict["TP"] = num_true_positives
        statistic_dict["FP"] = num_false_positives
        statistic_dict["TN"] = num_true_negative
        statistic_dict["FN"] = num_false_negative
        statistic_dict["TPR"] = float(num_true_positives) / float(num_of_positives)
        statistic_dict["TNR"] = float(num_true_negative) / float(num_of_negatives)
        statistic_dict["FNR"] = float(num_false_negative) / float(num_of_positives)
        statistic_dict["FPR"] = float(num_false_positives) / float(num_of_negatives)

        print("Number of %s (positives): %d" % (positive_name, statistic_dict["P"]))
        print("Number of %s (negatives): %d" % (negative_name, statistic_dict["N"]))
        print("Number of elements predicted %s and were correct (TP): %f" % (positive_name, statistic_dict["TP"]))
        print("Number of elements predicted %s and were wrong (FP): %f" % (positive_name, statistic_dict["FP"]))
        print("Number of elements predicted %s and were correct (TN): %f" % (negative_name, statistic_dict["TN"]))
        print("Number of elements predicted %s and were wrong (FN): %f" % (negative_name, statistic_dict["FN"]))
        print("True positive rate: %f" % statistic_dict["TPR"])
        print("True negative rate: %f" % statistic_dict["TNR"])
        print("False negative rate: %f" % statistic_dict["FNR"])
        print("False positive rate: %f" % statistic_dict["FPR"])

        return statistic_dict

    def calc_and_print_roc(self, test_ground_truth, probability_predictions):
        """

        :return:
        """
        model_name = self.model_name
        # Compute ROC curve and ROC area for each class
        fpr = dict()  # False positive rate
        tpr = dict()  # True positive rate
        y_test = np.array(test_ground_truth[:len(probability_predictions)])
        y_score = probability_predictions
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
        plt.plot(fpr[1], tpr[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f) when positive is normal' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic ' + self.model_name)
        plt.legend(loc="lower right")
        plt.savefig(self.model_name + "roc_normal.png")
        plt.show()

        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f) when positive is non normal' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(model_name + "roc_non_normal.png")
        plt.show()

    def calc_roc(self, test_ground_truth, probability_predictions, save_path):
        """
        1 - positive class
        0 - false class
        (0,1) - positive class
        (1,0) - negative class
        :param save_path: path where to save the roc curve (including file name)
        :return:
        """
        # Compute ROC curve and ROC area for each class
        fpr = dict()  # False positive rate
        tpr = dict()  # True positive rate
        assert len(probability_predictions) == len(test_ground_truth)
        y_test = np.array(test_ground_truth[:len(probability_predictions)])
        y_score = probability_predictions
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
        plt.plot(fpr[1], tpr[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic ')
        plt.legend(loc="lower right")
        # plt.savefig("roc_normal.png")
        plt.savefig(save_path)
        # plt.show()

        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        # plt.savefig("roc_non_normal.png")
        # plt.show()

        # print("Calc ROC manuaaly:")
        # performances.calc_roc_curve(test_ground_truth, probability_predictions, './manualy_roc.png')

    def calc_roc_for_multiclass_probelm(self, test_ground_truth, probability_predictions):
        """

        :param test_ground_truth:
        :param probability_predictions:
        :return:
        """

        n_classes = self.get_shape_of_output()
        assert n_classes > 2

        # Compute ROC curve and ROC area for each class
        fpr = dict()  # False positive rate
        tpr = dict()  # True positive rate
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(test_ground_truth[:, i], probability_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_ground_truth.ravel(), probability_predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot of a ROC curve for a specific class:
        print("Ploting ROC curve for class number 3:")
        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        # Plot ROC curves for the multiclass problem:

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    def calc_evaulations_for_multiclass(self, test_tags, probability_predictions_of_test_set,
                                        binary_predictions_of_test_set):
        """

        :param test_tags:
        :param probability_predictions_of_test_set:
        :return:
        """
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(test_tags,
                                                                    binary_predictions_of_test_set, average='micro')
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

    def print_precision_recall_graph(self, test_ground_truth, probability_predictions, save_path='.'):
        """
        We assume in this function - binary predictions.
        prints a precision recall graph.
        :param test_ground_truth: list of one-hot vectors with the ground truths : [(0,1), (0,1), ...(1,0),...]
        :param probability_predictions: list of predictions that came from the network after the softmax layer:
                [(0.2,0.8), (0.6,00.4), ...] probabilites.
        :return:
        """
        # The second element in each tuple in the test set is the score for the "positives":
        assert len(test_ground_truth) == len(probability_predictions)
        positives_ground_truth = [x[1] for x in test_ground_truth[:len(probability_predictions)]]
        positives_scores = [x[1] for x in probability_predictions]
        # positives_ground_truth = [x[0] for x in test_ground_truth[:len(probability_predictions)]]
        # positives_scores = [x[0] for x in probability_predictions]

        tf.logging.info("Precision recall fscore support : ")
        average_precision = average_precision_score(positives_ground_truth, positives_scores)
        tf.logging.info('Average precision-recall score: {0:0.2f}'.format(average_precision))

        precision, recall, _ = precision_recall_curve(positives_ground_truth, positives_scores)

        # plot precision recall graph :
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
        if not os.path.exists(save_path):
            print("Save path %s doesn't exists. Creating it" % save_path)
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "prec_rec.png"))
        plt.show()

        #tf.logging.info("Number of negatives: %d" % test_ground_truth.count(0))
        #tf.logging.info("Number of positives: %d" % test_ground_truth.count(1))

    def save_model(self):
        """
        saves the model into a .meta and closes the session
        :return:
        """
        print("Saving model:")
        # Create a saver object which will save all the variables
        saver = tf.train.Saver()
        saver.save(self.session, './' + self.model_name)
        print("Model saved successfully")
        self.close_session()
        print("Session closed")

    def restore_session(self):
        """
        Restores the model and session saved in the .meta file
        :return: session object
        """
        sess = tf.Session()
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('./' + self.model_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        self.session = sess
        return sess

    def close_session(self):
        """
        closess the session and changes self.session to None
        :return: None
        """
        self.session.close()
        self.session = None

    def delete_logs_folder(self):
        # 1. Clear and re-create the folders of the saved model and logs:
        print("Checking exsistance of logs directory", self.tensor_board_logs_dir)
        if os.path.exists(self.tensor_board_logs_dir):
            print("Logs directory exists, clearing its contents")
            for the_file in os.listdir(self.tensor_board_logs_dir):
                file_path = os.path.join(self.tensor_board_logs_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            print("Logs directory doesn't exist, creating it")
            os.makedirs(self.tensor_board_logs_dir)

        print("Checking exsistance of saved models directory", self.model_save_dir)
        if os.path.exists(self.model_save_dir):
            print("Saved model directory exists, clearing its contents")
            for the_file in os.listdir(self.model_save_dir):
                file_path = os.path.join(self.model_save_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            print("Saved model directory doesn't exist, creating it")
            os.makedirs(self.model_save_dir)

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

    def visualize_model(self):
        """

        :return:
        """
        with tf.Session() as sess:
            # sess.run(init)
            # First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph('./saved_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            # TODO: COMPLETE

    def test_on_mnist(self):
        # Get the sets of images and labels for training, validation, and
        # test on MNIST.
        mnist_data = input_data.read_data_sets("mnist_data", one_hot=True)
        train_set = mnist_data.train
        x = train_set.images
        y_tags = train_set.labels
        self.train_tf(x, y_tags, 2000, 100, "gradient_descent", "cross_entropy")

        '''
        test_set = mnist_data.test
        x_test = test_set.images
        y_tags_test = test_set.labels
        self.eval_accuracy(x_test, y_tags_test)
        '''
        return


if __name__ == "__main__":
    nn = GenericNeuralNetwork(os.path.join('architecture_files', 'conv_with_additional_features'),
                              tensor_board_logs_dir='./logs/examples')
    # nn = GenericNeuralNetwork(os.path.join('architecture_files', 'nn_example_2_layers'))
    # nn = GenericNeuralNetwork(os.path.join('architecture_files', 'conv_nn_example'))
    nn.test_on_mnist()


