"""
Here are some module docstrings
"""
from typing import Callable, Dict, List, Tuple
import numpy as np
np.random.seed(123)


class Layer:
    """
    Class that represents a single layer in an Artifical Neural Network (ANN)
    """

    def __init__(self, a_func: Callable, d_func: Callable,
                 weights: np.ndarray, biases: np.ndarray,
                 softmax: bool = False) -> None:
        self.a_func = a_func
        self.d_func = d_func
        self.weights = weights
        self.biases = biases
        # Keep track of the number of nodes for ease of access
        self.m = weights.shape[1]
        # Keep track of the results of the layer for backprop
        self.activations = None
        self.softmax = softmax

    def calculate_input(self, node_values: np.ndarray) -> np.ndarray:
        """
        Function that takes in the outputs of the nodes in the current layer, and calculates the
        activations for the next layer
        Args:
            node_values (np.ndarray): Values (outputs) of the nodes in the current layer

        Returns:
            np.ndarray: Result of multiplying nodes by weights and applying activation function
        """
        # Get the transposed weight-matrix
        print(f"Weights shape: {self.weights.shape}")
        weights_t = np.einsum('ij->ji', self.weights)
        print(node_values.shape)
        # Multiply the weight-matrix by the node-values
        output = np.einsum('ij...,j...->i...', weights_t, node_values)
        print(f"Output shape: {output.shape}")
        # Now add the bias to the output
        output = output + self.biases
        # Pass the result into the activation function, and return it
        activations = self.a_func(output)
        return activations

    def save_input(self, activations: np.ndarray) -> None:
        """
        Method for setting the result output of the current layer

        Args:
            activations (np.ndarray): Result of act_func(weights * nodes)
        """
        self.activations = activations

    def get_output_jacobian(self, column: int = 0) -> np.ndarray:
        """This method will return the jacobian matrix that represents the derivatives of
        the inputs to this layer with respect to the activations from the previous layer.
        From the examples, this corresponds to J^Z_Y

        Args:
            column (int): Index that extracts the particular column of the result in the current layer.

        Returns:
            np.ndarray: Matrix where index (m,n) corresponds to the derivative of output of node m times
                        the weight going from node n (in the previous layer) to node m in the current layer
        """
        # Notes:
        # OUTPUT of a layer will be a vector with shape (m, 1) where m is the number of nodes in the layer
        # The jacobian matrix will have shape (m, n), where m is the number of nodes in the current layer,
        # and n is the number of nodes in the previous layer.
        if not self.softmax:
            J_sum = np.diag(self.d_func(self.activations[:, column]))
            return np.einsum('ij,kj->ik', J_sum, self.weights)
        diag = self.activations[:, column]
        s = diag.reshape(-1,1)
        return np.diagflat(s) - np.einsum('ij,kj->ik', s, s)

    def get_weight_jacobian(self, column: int, previous_output: np.ndarray) -> np.ndarray:
        diag_J_sum = self.d_func(self.activations[:, column])
        # Return object corresponds to J-hat; einsum is outer product
        return np.einsum('i,j->ij', previous_output, diag_J_sum)

    def get_soft_jacobian(self, column: int):

    def __str__(self) -> str:
        pass


class NeuralNetwork:

    def __init__(self, config: Dict, batch_size: int = 1) -> None:
        """
        Taking in parameters for neural network.

        Args:
            layers_config (list[tuple[int, Callable]]):
                List of tuples containing information of every layer. It has the following format:
                [(number of nodes, act. func.), (number of nodes, act. func.), ...],
                where the index of the tuple corresponds to the layer position

            batch_size (int):
                Integer describing the number of examples to operate on simultaneously

        """
        self.batch_size = batch_size
        self.hidden_layers = []
        n_hl = config['Hidden layers']
        hl_neurons = config['HL Neurons']
        hl_funcs = config['HL Activation functions']
        hl_d_funcs = config['HL Derivative functions']
        # Weights: nxm, m current, n previous
        n = config['Inputs']
        for i in range(n_hl):
            # Nodes in the current hidden layer
            m = hl_neurons[i]
            func = hl_funcs[i]
            d_func = hl_d_funcs[i]
            w_shape = (n, m)
            b_shape = (m, 1)
            weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
            biases = np.zeros(shape=b_shape)
            layer = Layer(func, d_func, weights, biases)
            self.hidden_layers.append(layer)
            # Set the next n value for correct dimensions in the next weight matrix
            n = m
        m = config['Outputs']
        func = config['Output function']
        d_func = config['Output derivative function']
        w_shape = (n, m)
        b_shape = (m, 1)
        weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
        biases = np.zeros(shape=b_shape)
        self.output_layer = Layer(func, d_func, weights, biases)

    def forward_pass(self, network_inputs: np.ndarray) -> np.ndarray:
        # Get the first layer activations for use in the later layers
        # current_input = self.hidden_layers[0].calculate_output(network_inputs)
        current_input = network_inputs
        # Iterate through the rest of the layers, feeding the previous output as the next input
        for layer in self.hidden_layers:
            current_input = layer.calculate_input(current_input)
            layer.save_input(current_input)
        final_input = self.output_layer.calculate_input(current_input)
        self.output_layer.save_input(final_input)
        return final_input
        # Note to self: to extract the column of a numpy matrix, simply use matrix[:,<index of column>]


# Here we will define all the activation functions we can use

def sigmoid(input_vector: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function, rest of docstring unnecessary seeing as it
    becomes larger than the function itself.
    """
    return 1 / (1 + np.exp(input_vector))


def relu(input_vector: np.ndarray) -> np.ndarray:
    """
    Classic ReLu function with (somewhat) optimized time usage.
    """
    return np.maximum(input_vector, 0, input_vector)


def tanh(input_vector: np.ndarray) -> np.ndarray:
    return np.tanh(input_vector)


def unit(input_vector: np.ndarray) -> np.ndarray:
    return input_vector


def softmax(input_vector: np.ndarray) -> np.ndarray:
    """
    Classic softmax function for probability distributions.
    """
    return np.exp(input_vector) / np.sum(np.exp(input_vector), axis=0)


def d_sigmoid(input_vector: np.ndarray) -> np.ndarray:
    def f(x): return x * (1 - x)
    result = f(input_vector)
    return result


def d_relu(input_vector: np.ndarray) -> np.ndarray:
    """
    ReLu derivative. Where elements are <= 0, return 0. Else return 1
    """
    return np.where(input_vector > 0, 1, 0)


def d_tanh(input_vector: np.ndarray) -> np.ndarray:
    """
    Tanh derivative = 1 - tanh^2
    """
    def f(x): return 1 - x ^ 2
    return f(input_vector)


def d_unit(input_vector: np.ndarray) -> np.ndarray:
    return np.ones(shape=input_vector.shape)
