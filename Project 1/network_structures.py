"""
Here are some module docstrings
"""
from tkinter.messagebox import NO
from typing import Callable
import numpy as np
np.random.seed(123)


class Layer:
    """
    Class that represents a single layer in an Artifical Neural Network (ANN)
    """

    def __init__(self, act_func: Callable, weights: np.ndarray, biases: np.ndarray) -> None:
        self.act_func = act_func
        self.weights = weights
        self.biases = biases
        # Keep track of the number of nodes for ease of access
        self.nodes = weights.shape[0]
        # Keep track of the results of the layer for backprop
        self.activations = None

    def get_activations(self, node_values: np.ndarray) -> np.ndarray:
        """
        Function that takes in the outputs of the nodes in the current layer, and calculates the
        activations for the next layer
        Args:
            node_values (np.ndarray): Values (outputs) of the nodes in the current layer

        Returns:
            np.ndarray: Result of multiplying nodes by weights and applying activation function
        """
        # Get the transposed weight-matrix
        weights_t = np.einsum('ij->ji', self.weights)
        # Multiply the weight-matrix by the node-values
        output = np.einsum('ij...,j...->i...', weights_t, node_values)
        # Now add the bias to the output
        output = output + self.biases
        # Pass the result into the activation function, and return it
        activations = self.act_func(output)
        return activations

    def set_activations(self, activations: np.ndarray) -> None:
        """
        Method for setting the result output of the current layer

        Args:
            activations (np.ndarray): Result of act_func(weights * nodes)
        """
        self.activations = activations

    def __str__(self) -> str:
        pass


class NeuralNetwork:

    def __init__(self, layers_config: list[tuple[int, Callable]], batch_size: int=1) -> None:
        """
        Taking in parameters for neural network.

        Args:
            layers_config (list[tuple[int, Callable]]):
                List of tuples containing information of every layer. It has the following format:
                [(number of nodes, act. func.), (number of nodes, act. func.), ...],
                where the index of the tuple corresponds to the layer position
            
            batch_size (int):
                Integer describing the number of batches (examples) to operate on simultaneously

        """
        self.batch_size = batch_size
        self.layers = []
        for i in range(len(layers_config) - 1):
            # If batch size is 1, then we must have a 1D bias matrix. Else we create a 2D bias matrix
            # It doesn't have to be broadcast, because numpy applies this automatically during addition
            bias_shape = (layers_config[i+1][0],1) if batch_size > 1 else (layers_config[i+1][0],)
            layer = Layer(
                act_func=layers_config[i][1],
                weights=
                    np.random.uniform(
                        low=-0.5,
                        high=0.5,
                        size=(layers_config[i][0], layers_config[i+1][0])
                    ),
                biases=
                    np.zeros(shape=bias_shape)
            )
        self.layers = [
            Layer(
                act_func=layers_config[i][1],
                weights=
                    np.random.uniform(
                        low=-0.5,
                        high=0.5,
                        size=(layers_config[i][0], layers_config[i+1][0])
                    ),
                biases=
                    np.zeros(shape=(layers_config[i+1][0],1))
            ) for i in range(len(layers_config) - 1)
        ]
        # To make the forward pass function work with the final layer, we just add a final
        # Layer-object with the identity matrix (to return the same vector)
        n_outputs = layers_config[-1][0]
        final_layer = Layer(act_func=layers_config[-1][1], weights=np.identity(n_outputs), biases=0)
        self.layers.append(final_layer)

    def forward_pass(self, network_inputs: np.ndarray) -> None:
        # Make sure that the number of inputs match the number of nodes in the input-layer
        assert len(network_inputs) == self.layers[0].nodes
        # Get the first layer activations for use in the later layers
        current_output = self.layers[0].get_activations(network_inputs)
        self.layers[0].set_activations(current_output)
        # Iterate through the rest of the layers, feeding the previous output as the next input
        for layer in self.layers[1:]:
            current_output = layer.get_activations(current_output)
            layer.set_activations(current_output)
        # Note to self: to extract the column of a numpy matrix, simply use matrix[:<index of column>]


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


def softmax(input_vector: np.ndarray) -> np.ndarray:
    """
    Classic softmax function for probability distributions.
    """
    return np.exp(input_vector) / np.sum(np.exp(input_vector), axis=0)


# User testing functions

def test_activations() -> None:
    test_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    test = Layer(act_func=sigmoid, weights=test_weights, biases=1)
    test_outputs = np.array([0.1, 0.4, 0.8])
    test_result = test.get_activations(test_outputs)
    print(test_result)


def test_final_layer_activations() -> None:
    test_weights = np.array([[1]])
    test = Layer(act_func=sigmoid, weights=test_weights, biases=1)
    test_outputs = np.array([0.1])
    test_result = test.get_activations(test_outputs)
    print(test_result)


def test_network_config() -> None:
    test_config = [(3, sigmoid), (2, sigmoid), (3, sigmoid)]
    test_object = NeuralNetwork(layers_config=test_config)
    print(len(test_object.layers))
    for layer in test_object.layers:
        print(layer.weights)


def test_network_forward_pass() -> None:
    test_config = [(3, relu), (2, relu), (10, softmax)]
    test_object = NeuralNetwork(layers_config=test_config)
    test_network_inputs_1 = np.array([1, 3, 4])
    for layer in test_object.layers:
        print(layer.biases)
    test_object.forward_pass(test_network_inputs_1)
    for layer in test_object.layers:
        print(layer.activations)
    print()

def test_forward_pass_with_batch() -> None:
    test_config = [(3, relu), (2, relu), (10, softmax)]
    test_object = NeuralNetwork(layers_config=test_config, batch_size=1)
    inputs = np.array([[1, 2, 3], [3, 5, 6], [4, 8, 7]])
    for layer in test_object.layers:
        print(layer.biases)
    test_object.forward_pass(inputs)
    for layer in test_object.layers:
        print(layer.activations)


if __name__ == '__main__':
    # test_activations()
    # test_final_layer_activations()
    # test_network_config()
    test_network_forward_pass()
    # test_forward_pass_with_batch()
