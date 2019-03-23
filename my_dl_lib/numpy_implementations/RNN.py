"""
My implementation of an RNN Cell compenent in Numpy.
I took some code from here:

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

"""
import numpy as np


class RNNCell:
    def __init__(self, hidden_size, v_size):
        """
        Initializes the RNN weights. this is valid for any number of RNN cells because they all share the same weights.
        :param hidden_size: number of neruons at the hidden layer
        :param v_size: vocabulary dimension - also determins the size dim of a single word input.
        """
        self.hidden_layer_size = hidden_size
        self.vocabulary_size = v_size

        # Init the trainable weights with random values:
        self.U = np.random.uniform(low=-np.sqrt(1./v_size), high=np.sqrt(1./v_size), size=(v_size, hidden_size))

        self.V = np.random.uniform(low=-np.sqrt(1. / hidden_size), high=np.sqrt(1. / hidden_size),
                                   size=(hidden_size, v_size))

        self.W = np.random.uniform(low=-np.sqrt(1. / hidden_size), high=np.sqrt(1. / hidden_size),
                                   size=(hidden_size, hidden_size))

    def forward_pass(self, data_sqeuence):
        """
        Feed a sequence in the data to the RNN unit and calculates the next words probabilites.
        :param data_sqeuence: a sequence for which we want to predict the next word.
        :return:
                1. matrix O - each row is a vector of vobabulary size which predicts the i+1 word of the sqeuence.
                 number of rows as the length of the sequence.
                2. matrix S - the hidden state valus. ( starting from S(-1) ) : each row is the hidden state at time i.
        """

        # length of sequence:
        T = len(data_sqeuence)

        # During forward propagation we save all hidden states in s because wee need them for the gradient calculations.
        # We init S(-1) to 0
        S = np.zeros(shape=(T + 1, self.hidden_layer_size))
        S[-1] = np.zeros(shape=(1, self.hidden_layer_size))

        O = np.zeros(shape=(T, self.vocabulary_size))

        # calculate for each time step t the prediction of the t+1 word in the sequence:
        for t in range(T):
            word_index = self.get_word_index(data_sqeuence[t])
            S[t] = np.tanh(S[t-1].dot(self.W) + self.U[:, word_index])
            O[t] = np.softmax(S[t].dot(self.V))  # TODO: implement softmax ?

        return [O, S]

    @staticmethod
    def get_word_index(word_vector):
        for i, e in enumerate(word_vector):
            if e == 1:
                return i
            else:
                assert e == 0

    def cross_entropy(self, prediction, true_label):
        index_of_true_label = self.get_word_index(true_label)
        loss = -1. * np.log2(prediction[index_of_true_label])
        return loss

    def calc_cross_entropy_loss(self, x, y):
        """
        Calculates the mean loss over N examples. each example is a sequence of size T and has a corresponding label.
        We use the cross-entropy loss function.
        :param x: list of sequences - [[x1,x2,..,xT], [x1,x2,..,XT]...] where each x is a word vector.
        :param y:
        :return:
        """
        loss = 0
        N = len(y)
        for i, example in enumerate(x):
            O, S = self.forward_pass(example)
            loss_of_sequence = 0
            T = len(example)
            for t, _ in enumerate(O):
                local_loss = self.cross_entropy(O[t], y[t])
                loss_of_sequence += local_loss
            loss += (1. / T) * loss_of_sequence

        loss = (1. / N) * loss
        return loss