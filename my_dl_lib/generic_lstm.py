import collections
import math
import os
import random
import shutil

from my_dl_lib import generic_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc


class GenericLSTM:

    def __init__(self, number_of_hidden_neurons, size_of_sequence, size_of_each_sample, output_length,
                 number_of_lstm_layers=1, tensor_board_dir='./logs', saved_models_dir='./saved_models', mode='TRAIN'):
        """
        initializes a new GenericLSTM object with all the meta configurations which defines an LSTM architecture
        :param number_of_hidden_neurons: int to define the size of the hidden layer, that is the number of neurons
        of each gate in the hidden layer.
        :param size_of_sequence: size of one sentence - number of words/samples we feed to the network at each step.
        :param size_of_each_sample: the size of each word/sample (for example voc size if it is a one-hot)
        :param output_length: length of the output vector we predict, (it will not always be the size of one sample)
        :param number_of_lstm_layers:
        """
        if mode != "TEST":
            # 1. Clear and re-create the folders of the saved model and logs:
            print("Checking exsistance of logs directory", tensor_board_dir)
            if os.path.exists(tensor_board_dir):
                print("Logs directory exists, clearing its contents")
                for the_file in os.listdir(tensor_board_dir):
                    file_path = os.path.join(tensor_board_dir, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)
            else:
                print("Logs directory doesn't exist, creating it")
                os.makedirs(tensor_board_dir)

            print("Checking exsistance of saved models directory", saved_models_dir)
            if os.path.exists(saved_models_dir):
                print("Saved model directory exists, clearing its contents")
                for the_file in os.listdir(saved_models_dir):
                    file_path = os.path.join(saved_models_dir, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)
            else:
                print("Saved model directory doesn't exist, creating it")
                os.makedirs(saved_models_dir)
        else:
            assert mode == "TEST"
            # if we are in test mode we assume that we need to load an existing model. therefore the object will be
            # built empty
            self.length_of_output_prediction = output_length
            self.session = None
            return

        self.merged_summaries = None
        self.tensor_board_dir = tensor_board_dir
        self.saved_models_dir = saved_models_dir

        self.train_writer = None
        self.val_writer = None
        self.test_writer = None

        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.size_of_sequence = size_of_sequence
        self.size_of_each_sample = size_of_each_sample
        self.length_of_output_prediction = output_length

        if number_of_lstm_layers > 1:
            # 2-layer LSTM, each layer has n_hidden units.
            # Average Accuracy= 95.20% at 50k iter
            self.lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.number_of_hidden_neurons)
                                                          for x in range(0, number_of_lstm_layers)])
        elif number_of_lstm_layers == 1:
            # 1-layer LSTM with n_hidden units but with lower accuracy.
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.number_of_hidden_neurons)

        else:
            print("Error: al least 1 lstm layer needed. %d given", number_of_lstm_layers)
            return

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.size_of_sequence, self.size_of_each_sample],
                                name='input_placeholder')
        # self.x = tf.placeholder("float", [self.size_of_sequence, None, self.size_of_each_sample])
        self.y = tf.placeholder("float", [None, self.length_of_output_prediction], name='tags_placeholder')

        # Before softmax. shape is [None, self.length_of_output_prediction]
        self.V = self.create_prediction_tensor()

        # Model evaluation
        self.soft_max_tensor = tf.nn.softmax(self.V, name="last_soft_max_tensor")
        self.full_predictions_tensor = tf.argmax(self.soft_max_tensor, 1, name="full_predictions")

        self.session = None

    def create_prediction_tensor(self):
        """

        :return:
        """
        # split the placeholder to a list of T tensors, each tensor is all the words of a single time step.
        x = tf.unstack(self.x, num=self.size_of_sequence, axis=1)

        # Generate prediction:

        outputs, states = tf.contrib.rnn.static_rnn(self.lstm_cell, x, dtype=tf.float32)

        # Just for testing:
        # shape_of_output = outputs.get_shape().as_list()
        # print("Shape of the output tensor from the lstm is ", shape_of_output)

        # Define weights and biases between the hidden and output layers:
        # weight = tf.Variable(tf.truncated_normal([self.lstm_cell, self.size_of_each_sample]))
        # bias = tf.Variable(tf.constant(0.1, shape=[targets_width]))
        # there are size_of_sequence outputs but
        # we only want the last output
        return tf.layers.dense(inputs=outputs[-1], units=self.length_of_output_prediction, activation=None)

    def my_lstm_runner(self, cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None):
        """
        defines how to run the LSTM cell. Currently this is just a copy from the tensorflow implementation.
        :param cell:
        :param inputs:
        :param initial_state:
        :param dtype:
        :param sequence_length:
        :param scope:
        :return:
        """
        pass
        '''
        if not _like_rnncell(cell):
            raise TypeError("cell must be an instance of RNNCell")
        if not nest.is_sequence(inputs):
            raise TypeError("inputs must be a sequence")
        if not inputs:
            raise ValueError("inputs must not be empty")

        outputs = []
        # Create a new scope in which the caching device is either
        # determined by the parent scope, or is set to place the cached
        # Variable using the same placement as for the rest of the RNN.
        with vs.variable_scope(scope or "rnn") as varscope:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

            # Obtain the first sequence of the input
            first_input = inputs
            while nest.is_sequence(first_input):
                first_input = first_input[0]

            # Temporarily avoid EmbeddingWrapper and seq2seq badness
            # TODO(lukaszkaiser): remove EmbeddingWrapper
            if first_input.get_shape().ndims != 1:

                input_shape = first_input.get_shape().with_rank_at_least(2)
                fixed_batch_size = input_shape[0]

                flat_inputs = nest.flatten(inputs)
                for flat_input in flat_inputs:
                    input_shape = flat_input.get_shape().with_rank_at_least(2)
                    batch_size, input_size = input_shape[0], input_shape[1:]
                    fixed_batch_size.merge_with(batch_size)
                    for i, size in enumerate(input_size):
                        if size.value is None:
                            raise ValueError(
                                "Input size (dimension %d of inputs) must be accessible via "
                                "shape inference, but saw value None." % i)
            else:
                fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                batch_size = array_ops.shape(first_input)[0]
            if initial_state is not None:
                state = initial_state
            else:
                if not dtype:
                    raise ValueError("If no initial_state is provided, "
                                     "dtype must be specified")
                state = cell.zero_state(batch_size, dtype)

            if sequence_length is not None:  # Prepare variables
                sequence_length = ops.convert_to_tensor(
                    sequence_length, name="sequence_length")
                if sequence_length.get_shape().ndims not in (None, 1):
                    raise ValueError(
                        "sequence_length must be a vector of length batch_size")

                def _create_zero_output(output_size):
                    # convert int to TensorShape if necessary
                    size = _concat(batch_size, output_size)
                    output = array_ops.zeros(
                        array_ops.stack(size), _infer_state_dtype(dtype, state))
                    shape = _concat(fixed_batch_size.value, output_size, static=True)
                    output.set_shape(tensor_shape.TensorShape(shape))
                    return output

                output_size = cell.output_size
                flat_output_size = nest.flatten(output_size)
                flat_zero_output = tuple(
                    _create_zero_output(size) for size in flat_output_size)
                zero_output = nest.pack_sequence_as(
                    structure=output_size, flat_sequence=flat_zero_output)

                sequence_length = math_ops.to_int32(sequence_length)
                min_sequence_length = math_ops.reduce_min(sequence_length)
                max_sequence_length = math_ops.reduce_max(sequence_length)

            for time, input_ in enumerate(inputs):
                if time > 0:
                    varscope.reuse_variables()
                # pylint: disable=cell-var-from-loop
                call_cell = lambda: cell(input_, state)
                # pylint: enable=cell-var-from-loop
                if sequence_length is not None:
                    (output, state) = _rnn_step(
                        time=time,
                        sequence_length=sequence_length,
                        min_sequence_length=min_sequence_length,
                        max_sequence_length=max_sequence_length,
                        zero_output=zero_output,
                        state=state,
                        call_cell=call_cell,
                        state_size=cell.state_size)
                else:
                    (output, state) = call_cell()

                outputs.append(output)

            return (outputs, state)
        '''

    def prepare_data(self, seq, convert_output_labels_func=None):
        """
        Gets an array of sequential data, ( each element is a float number ) and returns a new data structure with the
        following dims: [?, sequence_length, size_of_one_sample], each element in we pick from the first dimension,
        is excactly one input we can feed to the lstm cell.
        it also returns a data structure to correspond to the tags - [?, size_of_one_sample], the first row is the sample
        that comes after the first time series in the first data structure...
        :param seq:
        :param convert_output_labels_func: a function which gets as input batch of labels and converts to a format which
        matches to [?, self.length_of_output_prediction]
        :return:
                    x_data  - [?, sequence_length, size_of_one_sample]
                    y_labels - [?, length_of_output_prediction]
        """

        # split into items were each item is of size sample.
        seq_splited_by_size_of_sample = [np.array(seq[i * self.size_of_each_sample: (i + 1) * self.size_of_each_sample])
               for i in range(len(seq) // self.size_of_each_sample)]

        # split into groups of size_of_sequence
        # spceial case:
        if len(seq_splited_by_size_of_sample) - self.size_of_sequence == 0:
            x_data = np.array([seq_splited_by_size_of_sample[:self.size_of_sequence]])

            y_labels = np.array([seq_splited_by_size_of_sample[self.size_of_sequence]])
        else:
            x_data = np.array([seq_splited_by_size_of_sample[i: i + self.size_of_sequence]
                        for i in range(len(seq_splited_by_size_of_sample) - self.size_of_sequence)])

            # y_labels become [?, length_of_one_sample]
            y_labels = np.array([seq_splited_by_size_of_sample[i + self.size_of_sequence]
                        for i in range(len(seq_splited_by_size_of_sample) - self.size_of_sequence)])

            if convert_output_labels_func is None:
                assert self.length_of_output_prediction == self.size_of_each_sample
            else:
                # convert each sample to length_of_output_prediction:
                y_labels = convert_output_labels_func(y_labels)

        '''
        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y
        '''
        return x_data, y_labels

    def train(self, x_data, y_labels, x_val, y_val, num_of_iterations, batch_size, display_step=1000, learning_rate=0.001,
              loss_function='cross_entropy', optimizer='rms_prop', word_to_index=None, index_to_word=None):
        """

        :param x_data:
        :param y_labels:
        :param num_of_iterations:
        :param batch_size:
        :param display_step:
        :param learning_rate:
        :param loss_function:
        :param optimizer:
        :return:
        """
        predictor_tensor = self.V

        # Loss and optimizer
        if loss_function == "cross_entropy":
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictor_tensor, labels=self.y),
                                  name='cost')
        elif loss_function == "mse":
            cost = tf.reduce_mean(tf.squared_difference(predictor_tensor, self.y), name='cost')
        else:
            print("Undefined loss function")
            return -1
        tf.summary.scalar("loss", cost)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Only relevant for classification:
        correct_pred = tf.equal(tf.argmax(self.soft_max_tensor, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.accuracy = accuracy
        self.cost = cost

        tf.summary.scalar("accuracy", accuracy)
        # Initializing the variables
        init = tf.global_variables_initializer()

        # Create a saver object which will save all the variables
        saver = tf.train.Saver(max_to_keep=150)

        # Add configurations for early stopping:
        # Best validation accuracy seen so far.
        best_validation_accuracy = 0.0
        best_validation_loss = 100.0

        # Iteration-number for last improvement to validation accuracy.
        last_improvement = 0

        # Stop optimization if no improvement found in this many iterations.
        require_improvement = 1000

        # Launch the graph
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        session = self.session
        session.run(init)

        # Merge all the summaries and write them out:
        self.merged_summaries = tf.summary.merge_all()
        acc_total = 0
        loss_total = 0

        # Create Writer object to visualize the graph later:
        self.train_writer = tf.summary.FileWriter(self.tensor_board_dir + '/train', session.graph)
        self.val_writer = tf.summary.FileWriter(self.tensor_board_dir + '/val')

        data_iterator = generic_dataset.GenericDataSetIterator(x_data, y_labels)
        for i in range(num_of_iterations):
            x_batch, y_batch = data_iterator.next_batch(batch_size)

            summary, _, acc, loss, onehot_pred = session.run([self.merged_summaries, optimizer, accuracy, cost,
                                                              predictor_tensor],
                                                             feed_dict={self.x: x_batch, self.y: y_batch})
            self.train_writer.add_summary(summary, i)
            loss_total += loss
            acc_total += acc

            if (i + 1) % 10 == 0:
                print("iter " + str(i) + " , Train-Batch loss: " + "{:.2f}".format(loss) + ", Train-Batch Accuracy: " +
                      "{:.2f}".format(acc))
                summary, val_acc, val_loss, onehot_pred = session.run([self.merged_summaries, accuracy, cost, predictor_tensor],
                                                              feed_dict={self.x: x_val, self.y: y_val})
                print("iter " + str(i) + "  , validation loss " + "{:.2f}".format(val_loss) + ", val accuracy: " +
                      "{:.2f}".format(val_acc))
                self.val_writer.add_summary(summary, i)

            if loss_function == 'cross_entropy':
                if (i + 1) % display_step == 0:
                    print("Iter= " + str(i + 1) + ", Train Average Loss= " + \
                          "{:.6f}".format(loss_total / display_step) + ", Train Average Accuracy= " + \
                          "{:.2f}%".format(100 * acc_total / display_step))
                    acc_total = 0
                    loss_total = 0
                    # TODO: make generic!
                    if index_to_word is not None:
                        symbols_in = [index_to_word[x[0]] for x in x_batch[0]]
                        symbols_out = index_to_word[list(y_batch[0]).index(1)]
                        symbols_out_prediction = index_to_word[int(session.run(tf.argmax(onehot_pred, 1)))]
                        # symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                        # symbols_out = training_data[offset + n_input]
                        # symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                        print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_prediction))

            if loss_function == 'mse':
                if (i + 1) % display_step == 0:
                    tf.logging.info("Step:%d   train_loss:%.6f" % (i + 1, loss))
                    print("Step:%d   train_loss:%.6f" % (i+1, loss))

            if (i + 1) % 10 == 0:
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                if val_acc > best_validation_accuracy:
                    # Update the best-known validation accuracy.
                    best_validation_accuracy = val_acc

                    # Set the iteration for the last improvement to current.
                    last_improvement = i

                    # Save all variables of the TensorFlow graph to file.
                    saver.save(sess=session, save_path=os.path.join(self.saved_models_dir, 'iter_' + str(i)),
                               global_step=num_of_iterations)
                    print("Found improvement in validation accuracy - %f.2 , Saving graph" % val_acc)

            # If no improvement found in the required number of iterations.
            '''
            if i - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")
                print("The best validation acc is %f" % best_validation_accuracy)
                print("The best validation loss is %f" % best_validation_loss)
                print("Best validation acc found at iter %d" % last_improvement)
                # Break out from the for-loop.
                break
            '''
        print("The best validation acc is %f" % best_validation_accuracy)
        print("The best validation loss is %f" % best_validation_loss)
        print("Best validation acc found at iter %d" % last_improvement)
        # saver.save(sess=session, save_path=os.path.join(self.saved_models_dir, 'iter_' + str(num_of_iterations)),
        #           global_step=num_of_iterations)
        # TODO: Remove:
        print("Optimization Finished!")
        # print("Elapsed time: ", elapsed(time.time() - start_time))

    def eval_accuracy(self, test_beats, test_tags):
        """

        :param test_beats:
        :param test_tags:
        :return:
        """
        one_hot_preds = tf.one_hot(self.full_predictions_tensor, self.length_of_output_prediction)
        acc, cost, bin = self.session.run([self.accuracy, self.cost, one_hot_preds], feed_dict={self.x: test_beats,
                                                                                       self.y: test_tags})
        print("Test set acc: %f.2  Test set Cost: %f.2" % (acc, cost))
        return acc, cost, bin

    def get_probability_predictions(self, x_data, meta_file_name=None):
        """
        Feeds all the examples in x_data to the network and returns a vector of full predictions i.e prbabilites,
        ...
        :param x_data: matrix of examples : Number_of_examples X d
        :param meta_file_name:
        :return:
        """
        if meta_file_name is None:
            probability_preds = self.session.run(self.soft_max_tensor, feed_dict={self.x: x_data})
            # print(probability_preds)
        else:
            assert self.session is None  # just make sure session is closed.
            print("Restoring graph:")
            if not os.path.isfile(meta_file_name + '.meta'):
                raise AssertionError("Meta file %s doesn't exists" % meta_file_name)

            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(meta_file_name + '.meta')
                saver.restore(sess, save_path=meta_file_name)
                graph = tf.get_default_graph()
                input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
                probability_tensor = graph.get_tensor_by_name("last_soft_max_tensor:0")
                probability_preds = sess.run(probability_tensor, feed_dict={input_placeholder: x_data})

        return probability_preds

    def get_binary_predictions(self, x_data, meta_file_name=None):
        """
        Feeds all the examples in x_data to the network and returns a vector of full binary predictions as one hot
        vector
        ...
        :param x_data: matrix of examples : Number_of_examples X d
        :return:
        """
        if meta_file_name is None:
            one_hot_preds = tf.one_hot(self.full_predictions_tensor, self.length_of_output_prediction)
            bin_preds = self.session.run(one_hot_preds, feed_dict={self.x: x_data})
        else:
            assert self.session is None  # just make sure session is closed.
            print("Restoring graph:")
            print(os.path.isfile(meta_file_name + '.meta'))
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(meta_file_name + '.meta')
                # lateset_checkpoint = tf.train.latest_checkpoint(meta_file_name)
                # print("Latest checkpoint ", lateset_checkpoint)
                saver.restore(sess, save_path=meta_file_name)
                graph = tf.get_default_graph()

                # for op in graph.get_operations():
                #    print(str(op.name))

                input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
                binary_tensor = graph.get_tensor_by_name("full_predictions:0")
                one_hot_preds = tf.one_hot(binary_tensor, self.length_of_output_prediction)
                bin_preds = sess.run(one_hot_preds, feed_dict={input_placeholder: x_data})

        return bin_preds

    def calc_precision_recall_(self, test_ground_truth, probability_predictions, save_path):
        """
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

        print("Precision recall fscore support : ")
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
        plt.savefig(save_path)
        # plt.show()

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
        print("Saving Fig in :", save_path)
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

    def calc_statistics_under_certein_treshold(self, x_test, y_test, treshold, positive_name,
                                               positive_index, negative_name, meta_file_name=None):
        """
        Calculate the accuracy and the sensitivity (True positive rate) and FPR under specific treshold.
        :param x_test:
        :param y_test:
        :param treshold:
        :param meta_file_name
        :return:
        """
        probabilities_predictions = self.get_probability_predictions(x_data=x_test, meta_file_name=meta_file_name)
        shepe_of_preds = probabilities_predictions.shape
        # For now we only support classification of two classes:
        if len(shepe_of_preds) != 2:
            raise AssertionError("For now we only support classification of two classes, you enters %d" % shepe_of_preds)
        final_preds_according_to_treshold = [1 if pred[0] < treshold else 0 for pred in probabilities_predictions]
        predictions = final_preds_according_to_treshold
        test_ground_truth = y_test
        print("Calculating Statistics if the Positive class is %d", positive_index)
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

    def test_on_dummy_data(self):
        """
        taken from here : https://mapr.com/blog/deep-learning-tensorflow/
        :return:
        """
        random.seed(111)
        rng = pd.date_range(start='2000', periods=209, freq='M')
        ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()
        ts.plot(c='b', title='Example Time Series')
        plt.show()
        ts.head(10)

        TS = np.array(ts)
        train_set = TS[:200]
        x_train, y_train = self.prepare_data(train_set)
        test_set = TS[-41:]
        # test_set = test_set[:20]
        x_test, y_test = self.prepare_data(test_set)
        self.train(x_train, y_train, 1000, 200, 100, loss_function='mse')

        y_pred = self.session.run(self.V, feed_dict={self.x: x_test})
        print(y_pred)

        plt.title("Forecast vs Actual", fontsize=14)
        plt.plot(pd.Series(np.ravel(y_test)), "bo", markersize=10, label='Actual')
        plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label='Forecast')
        plt.legend(loc="upper left")
        plt.xlabel("Time periods")
        plt.show()

        self.session.close()

    def test_on_short_text_data(self):
        """
        taken from here: https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
        :return:
        """
        # Text file containing words for training
        training_file = 'belling_the_cat.txt'

        def read_data(fname):
            with open(fname) as f:
                content = f.readlines()
            content = [x.strip() for x in content]  # remove whitespaces from beginning and end of text
            content = [content[i].split() for i in range(len(content))]
            content = np.array(content)
            content = np.reshape(content, [-1, ])
            return content

        training_file_directory = os.path.join('tensor_flow_implementations', 'recurrent_networks', 'lstm_unofficial')

        # returns a list of all words in the text file.
        training_data = read_data(os.path.join(training_file_directory, training_file))
        print("Loaded training data...")

        def build_dataset(words):
            count = collections.Counter(words).most_common()
            dictionary = dict()
            for word, _ in count:
                dictionary[word] = len(dictionary)
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
            return dictionary, reverse_dictionary

        word_to_index, index_to_word = build_dataset(training_data)  # Give a unique number for each word in the text
        vocab_size = len(word_to_index)
        assert vocab_size == self.length_of_output_prediction
        # Parameters
        learning_rate = 0.001
        training_iters = 50000
        display_step = 1000

        sequence = [word_to_index[x] for x in training_data]
        assert max(sequence) == 111 and min(sequence) == 0

        def convert_to_one_hot(data):
            one_hot_tags = []
            for tag in data:
                one_hot = [0 for x in range(112)]
                one_hot[tag[0]] = 1
                one_hot_tags.append(one_hot)
            return np.array(one_hot_tags)

        x_data, y_labels = self.prepare_data(sequence, convert_to_one_hot)

        self.train(x_data, y_labels, training_iters, batch_size=1, display_step=display_step, learning_rate=learning_rate,
                   loss_function='cross_entropy', optimizer='rms_prop', word_to_index=word_to_index,
                   index_to_word=index_to_word)

        self.session.close()
        # print("Run on command line.")
        # print("\ttensorboard --logdir=%s" % (logs_path))
        # print("Point your web browser to: http://localhost:6006/")
        '''
        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")
        '''

    def test_on_stock_prices(self):
        """
        taken from here : https://github.com/lilianweng/stock-rnn
        :return:
        """

        training_file_directory = os.path.join('tensor_flow_implementations', 'recurrent_networks', 'stock_preds')
        # Read csv file
        raw_df = pd.read_csv(os.path.join(training_file_directory, 's&p500_historical_prices.csv'))
        raw_seq = raw_df['Close'].tolist()
        n = len(raw_seq)
        print("Number of points: %d" % n)

        def plot_samples(preds, targets):
            def _flatten(seq):
                return [x for y in seq for x in y]

            truths = _flatten(targets)
            preds = _flatten(preds)
            days = range(len(truths))

            plt.figure(figsize=(12, 6))
            plt.plot(days, truths, label='truth')
            plt.plot(days, preds, label='pred')
            plt.legend(loc='upper left', frameon=False)
            plt.xlabel("day")
            plt.ylabel("normalized price")
            plt.ylim((min(truths), max(truths)))
            plt.grid(ls='--')
            plt.show()
            #if stock_sym:
            #     plt.title(stock_sym + " | Last %d days in test" % len(truths))

            # plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
            # plt.close()

        # train_seq = raw_seq[:(n-200)]
        test_seq = raw_seq[(n-200):]
        days = range(len(test_seq))
        plt.figure(figsize=(12, 6))
        plt.plot(days, test_seq, label='truth')
        # plt.plot(days, preds, label='pred')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.ylim((min(test_seq), max(test_seq)))
        plt.grid(ls='--')
        plt.show()

        x_data, y_labels = self.prepare_data(raw_seq)
        n = len(x_data)
        x_train = x_data[:(n - 200)]
        y_train = y_labels[:(n - 200)]
        x_test = x_data[(n - 200):]
        y_test = y_labels[(n - 200):]

        self.train(x_train, y_train, 1000, 100, loss_function='mse')

        # x_test, y_test = self.prepare_data(test_seq)
        y_pred = self.session.run(self.V, feed_dict={self.x: x_test})
        plot_samples(y_pred, y_test)

        self.session.close()

    def restore_model_and_eval_acc(self, test_beats, test_tags, meta_file_name, meta_file_directory, val_beats=None,
                                   val_tags=None,
                                   best_val_acc_value=None):
        """
        :param best_val_acc_value:
        :param val_beats:
        :param meta_file_name:
        :param meta_file_directory: directory which contains the meta files.
        :param test_beats:
        :param test_tags:
        :return:
        """
        assert self.session is None  # just make sure session is closed.
        print("Restoring graph with the best validation acc:")
        print(os.path.isfile(meta_file_name + '.meta'))
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_file_name + '.meta')
            # lateset_checkpoint = tf.train.latest_checkpoint(meta_file_directory)
            # print("Latest checkpoint ", lateset_checkpoint)
            saver.restore(sess, save_path=meta_file_name)
            graph = tf.get_default_graph()

            # for op in graph.get_operations():
            #    print(str(op.name))

            input_placeholder = graph.get_tensor_by_name("input_placeholder:0")
            tags_placeholder = graph.get_tensor_by_name("tags_placeholder:0")

            accuracy = graph.get_tensor_by_name("accuracy:0")
            cost = graph.get_tensor_by_name("cost:0")
            full_preds_tensor = graph.get_tensor_by_name("full_predictions:0")

            # calculate validation acc just for debugging:
            one_hot_preds = tf.one_hot(full_preds_tensor, self.length_of_output_prediction)

            if val_beats is not None:
                acc_val, cost_val, bin_val = sess.run([accuracy, cost, one_hot_preds], feed_dict={
                    input_placeholder: val_beats, tags_placeholder: val_tags})
                print("Accuracy on validation set: ", acc_val)
                print("Expected accuracy on validation set: ", best_val_acc_value)
                assert math.isclose(acc_val, best_val_acc_value, abs_tol=0.001)

            acc_test, cost_test, bin_test = sess.run([accuracy, cost, one_hot_preds], feed_dict={
                input_placeholder: test_beats, tags_placeholder: test_tags})
            print("Test set acc: %f.2  Test set Cost: %f.2" % (acc_test, cost_test))

            return acc_test, cost_test, bin_test

if __name__ == '__main__':
    # my_lstm = GenericLSTM(512, 20, 1, 1)
    # my_lstm.test_on_dummy_data()

    # test_lstm_short_text = GenericLSTM(512, 3, 1, 112, 2)
    # test_lstm_short_text.test_on_short_text_data()

    lstm_stocks = GenericLSTM(512, 30, 5, 5, number_of_lstm_layers=1)
    lstm_stocks.test_on_stock_prices()