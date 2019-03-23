import tensorflow as tf
import numpy as np


# Implementation of feed-forward neural network with 1 hidden layer, 2 neurons at each layer.
def feed_forward_nn_two_layers(x_input, y_tags):
    batch_size = 1

    num_of_neurons_at_hidden_layer = 2

    sess = tf.Session()

    # Define input Placeholders:
    x_dim = 2
    y_dim = 2
    X = tf.placeholder(tf.float32, shape=[None, x_dim + 1])
    y = tf.placeholder(tf.float32, shape=[None, y_dim])

    # Weight initializations

    # Layer 1: We define the weight matrix for each layer :
    # hidden_layer_w_matrix = tf.Variable(tf.random_normal((x_dim + 1, num_of_neurons_at_hidden_layer), stddev=0.1))
    hidden_layer_w_matrix = tf.Variable([[0.15, 0.25], [0.2, 0.3], [0.35, 0.35]], dtype=tf.float32)

    # Printing for testing:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Hidden layer weights values: ")
    print(sess.run(hidden_layer_w_matrix))

    # Layer 2:
    # final_layer_w_matrix = tf.Variable(tf.random_normal((num_of_neurons_at_hidden_layer + 1, y_dim), stddev=0.1))
    final_layer_w_matrix = tf.Variable([[0.4, 0.5], [0.45, 0.55], [0.6, 0.6]], dtype=tf.float32)

    init = tf.global_variables_initializer()
    sess.run(init)
    print("Output layer weights values: ")
    print(sess.run(final_layer_w_matrix))

    # Forward Pass :
    hidden_layer_output_before_activation = tf.matmul(X, hidden_layer_w_matrix)
    print("Hidden_layer_output_before_activation: ")
    print(sess.run(hidden_layer_output_before_activation, feed_dict={X: x_input}))

    hidden_layer_output_after_activation = tf.nn.sigmoid(hidden_layer_output_before_activation)
    # Add biass for final layer:
    hidden_layer_output_after_activation = tf.concat([hidden_layer_output_after_activation,
                                                      np.ones((batch_size, 1),
                                                              float)], 1)
    print("Hidden_layer_output_after_activation: ")
    print(sess.run(hidden_layer_output_after_activation, feed_dict={X: x_input}))

    final_layer_output_before_activation = tf.matmul(hidden_layer_output_after_activation,
                                                     final_layer_w_matrix)

    print("Final_layer_output_before_activation: ")
    print(sess.run(final_layer_output_before_activation, feed_dict={X: x_input}))

    # softmax_activation_on_final_layer = tf.nn.softmax(final_layer_output_before_activation)
    sigmoid_activation_on_final_layer = tf.nn.sigmoid(final_layer_output_before_activation)

    # print("softmax_activation_on_final_layer: ")
    # print(sess.run(softmax_activation_on_final_layer, feed_dict={X: x_input}))
    print("sigmoid_activation_on_final_layer: ")
    print(sess.run(sigmoid_activation_on_final_layer, feed_dict={X: x_input}))

    # Backward propagation
    # cost_per_instance = -tf.reduce_sum((y) * tf.log(softmax_activation_on_final_layer), reduction_indices=[1])

    '''
    3 options for mean square errors in tensorflow:

        loss = tf.reduce_sum(tf.pow(prediction - Y,2))/(n_instances)
        loss = tf.reduce_mean(tf.squared_difference(prediction, Y))
        loss = tf.nn.l2_loss(prediction - Y)
    '''

    cost_per_instance = (tf.squared_difference(sigmoid_activation_on_final_layer, y))
    print("cost_per_instance: ")
    print(sess.run(cost_per_instance, feed_dict={X: x_input, y: y_tags}))

    total_cost_per_batch = tf.reduce_mean(cost_per_instance)

    print("total_cost_per_batch: ")
    print(sess.run(total_cost_per_batch, feed_dict={X: x_input, y: y_tags}))

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    updates = optimizer.minimize(total_cost_per_batch)
    predict = tf.argmax(sigmoid_activation_on_final_layer, axis=1)
    print("predictions: ")
    print(sess.run(predict, feed_dict={X: x_input, y: y_tags}))

    # Run session:
    # with tf.Session() as sess:

    # init = tf.global_variables_initializer()
    # sess.run(init)

    # sess.run(updates, feed_dict={X: x_input[i: i + 1], y: y_tags[i: i + 1]})
    train_accuracy = np.mean(np.argmax(y_tags, axis=1) ==
                             sess.run(predict, feed_dict={X: x_input, y: y_tags}))
    '''
    test_accuracy = np.mean(np.argmax(y_tags, axis=1) ==
                            sess.run(predict, feed_dict={X: x_input, y: y_tags}))
    '''
    print("E, train accuracy = %.2f%%, "
          % (100. * train_accuracy,))

    print(total_cost_per_batch)
    # grads_and_vars = optimizer.compute_gradients(xs=hidden_layer_w_matrix, ys=total_cost_per_batch)
    grads_and_vars = tf.gradients(xs=hidden_layer_w_matrix, ys=total_cost_per_batch)

    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(grads_and_vars, feed_dict={X: x_input, y: y_tags}))
    '''
    for gv in grads_and_vars:
        print(str(sess.run(gv[0])) + " - " + gv[1].name)
    '''
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(updates, feed_dict={X: x_input, y: y_tags})

    train_accuracy = np.mean(np.argmax(y_tags, axis=1) ==
                             sess.run(predict, feed_dict={X: x_input, y: y_tags}))
    '''
    test_accuracy = np.mean(np.argmax(y_tags, axis=1) ==
                            sess.run(predict, feed_dict={X: x_input, y: y_tags}))
    '''
    print("E, train accuracy = %.2f%%, "
          % (100. * train_accuracy))


def test_forward_pass():
    # Define two input samples_of_my_change: (1,2) , (5,6)
    # x = np.array([[0.05, 0.1], [0.03, 0.7]])
    x = np.array([[0.05, 0.1]])
    print("Input data:\n ", x)
    print("Input di:m ", x.shape)
    # Add bias:
    x = np.append(x, np.ones((1, 1), float), axis=1)
    print("Input data with bias:\n ", x)
    print("Input dim with bias ", x.shape)
    # x = tf.convert_to_tensor(x)

    # y_tags = np.array([[0, 1.0], [1.0, 0]])
    y_tags = np.array([[0.01, 0.99]])
    # y_tags = tf.convert_to_tensor(y_tags)
    feed_forward_nn_two_layers(x, y_tags)


def feed_forward_example(x_input, y_tags):
    """
    Implementation of a Fully connected NN in tensor flow with one hidden layer. 2 neurons at each layer.
    :param x_input: numpy matrix of size Ax3, each row is a sample with 2 features the last dim is 1 (will be
            multiplied with the trainable bias)
    :param y_tags: numby matrix of size AX2, each element is a tag corresponding to the same row in the x_input vector
    :return:
    """

    # A. Create the computational Graph:

    # 1. Define input variables:

    x_dim = 2
    y_dim = 2
    num_of_neurons_at_hidden_layer = 2
    batch_size = 2
    # Define input Placeholders:
    X = tf.placeholder(tf.float32, shape=[None, x_dim + 1])
    y = tf.placeholder(tf.float32, shape=[None, y_dim])

    # 2. Define weights matrices:

    # 2.1 hidden layer:
    # hidden_layer_w_matrix = tf.Variable(tf.random_normal((x_dim + 1, num_of_neurons_at_hidden_layer), stddev=0.1))
    hidden_layer_w_matrix = tf.Variable([[0.15, 0.25], [0.2, 0.3], [0.35, 0.35]], dtype=tf.float32)

    # 2.2: output Layer :
    # final_layer_w_matrix = tf.Variable(tf.random_normal((num_of_neurons_at_hidden_layer + 1, y_dim), stddev=0.1))
    final_layer_w_matrix = tf.Variable([[0.4, 0.5], [0.45, 0.55], [0.6, 0.6]], dtype=tf.float32)

    # 3. Forward Pass:

    # 3.1 : multiply input matrix with hidden layer weights matrix
    hidden_layer_output_before_activation = tf.matmul(X, hidden_layer_w_matrix)

    # 3.2 : Perform Activation function to the output of each neuron in the hidden layer:
    hidden_layer_output_after_activation = tf.nn.sigmoid(hidden_layer_output_before_activation)

    # 3.3 Add biases for final layer input:
    hidden_layer_output_after_activation = tf.concat([hidden_layer_output_after_activation,
                                                      np.ones((batch_size, 1),
                                                              float)], 1)

    # 3.4 : Multiply output of hidden layer with the weights matrix of the output layer:
    final_layer_output_before_activation = tf.matmul(hidden_layer_output_after_activation, final_layer_w_matrix)

    # 3.5: Perform Softmax activation fumction on the output of the final layer:
    softmax_activation_on_final_layer = tf.nn.softmax(final_layer_output_before_activation)
    # sigmoid_activation_on_final_layer = tf.nn.sigmoid(final_layer_output_before_activation)

    # 4. Calculate Cost:
    '''
        3 options for mean square errors in tensorflow:

            loss = tf.reduce_sum(tf.pow(prediction - Y,2))/(n_instances)
            loss = tf.reduce_mean(tf.squared_difference(prediction, Y))
            loss = tf.nn.l2_loss(prediction - Y)
    '''
    # Apply Cross-Entropy loss function:
    cost_per_instance = -tf.reduce_sum((y) * tf.log(softmax_activation_on_final_layer), reduction_indices=[1])
    # The output of cost_per_instance is a vector of all costs per each example fed to the network.

    # Calculate the sum of all errors acroos our batch and divide by the batch size:
    total_cost_per_batch = tf.reduce_mean(cost_per_instance)

    # The predicted tag of each sample is the highest element in each output vector:
    predictions = tf.argmax(softmax_activation_on_final_layer, axis=1)

    # 5. Train the weights: backpropagation using Gradient descent optimizer:
    # 5.1 Define the Gradient descent optimizer on our loss function:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    updates = optimizer.minimize(total_cost_per_batch)

    # B. Run The computational Graph:
    with tf.Session() as sess:
        # Initialize weights:
        init = tf.global_variables_initializer()
        sess.run(init)

        print("weights matrix hidden layer before training:", sess.run(hidden_layer_w_matrix))

        for i in range(3):
            sess.run(updates, feed_dict={X: x_input, y: y_tags})
            accuracy = np.mean(np.argmax(y_tags, axis=1) ==
                               sess.run(predictions, feed_dict={X: x_input, y: y_tags}))

            print("Iteration: %d Accuracy: %f" % (i, accuracy))
            print("weights matrix hidden layer:", sess.run(hidden_layer_w_matrix))


def run_example_nn():
    # Define two input samples_of_my_change:
    x = np.array([[0.05, 0.1], [0.1, 0.05]])
    # x = np.array([[0.05, 0.1]])
    print("Input data:\n ", x)
    print("Input dim ", x.shape)
    # Add bias:
    x = np.append(x, np.ones((2, 1), float), axis=1)
    print("Input data with bias:\n ", x)
    print("Input dim with bias ", x.shape)

    # Define tags:
    y_tags = np.array([[0, 1.0], [0.0, 1.0]])
    # y_tags = np.array([[0.01, 0.99]])
    feed_forward_example(x, y_tags)


# TODO: from a text file read architecture and design it.

if __name__ == "__main__":
    run_example_nn()

    # Example 1:

    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly
    print(node1, node2)
    with tf.Session() as sess:
        print("Value of node1:", end=" ")
        print(sess.run(node1), end='\n\n')
    '''
        # Example 2:

        a = tf.placeholder(tf.float32)
        b = tf.placeholder(tf.float32)
        adder_node = a + b
        print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
        print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
    '''




