import tensorflow as tf


class GenericNeuron:

    def __init__(self, input_size, activation_function):
        """

        :param input_size:
        :param activation_function:
        """
        self.input_size = input_size
        self.activation_function = activation_function

    def build_neuron_tensor_flow(self):
        """
        Using tensorflow build a tensor to represent the neuron.
        :return:
        """