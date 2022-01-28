"""
Here are some module docstrings
"""
from ctypes import Union
from typing import Callable, Dict, List, Tuple
import numpy as np
np.random.seed(123)


class Layer:
    """
    Class that represents a single layer in an Artifical Neural Network (ANN)
    """

    def __init__(self, a_func: Callable, da_func: Callable,
                 weights: np.ndarray, biases: np.ndarray,
                 softmax: bool = False) -> None:
        self.a_func = a_func
        self.da_func = da_func
        self.weights = weights
        self.biases = biases
        # Keep track of the number of nodes for ease of access
        self.m = weights.shape[1]
        # Keep track of the results of the layer for backprop
        self.activations = None
        self.prv_layer_inputs = None
        self.softmax = softmax

    def calculate_input(self, node_values: np.ndarray) -> np.ndarray:
        """
        Function that takes in the outputs of the nodes in the previous layer, and calculates the
        activations for the current layer
        Args:
            node_values (np.ndarray): Values (outputs) of the nodes in the previous layer

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
        activations = self.a_func(output)
        # Now store the values for use in backprop
        self.activations, self.prv_layer_inputs = activations, node_values
        return activations

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
            J_sum = np.diag(self.da_func(self.activations[:, column]))
            # Einsum corresponds to inner product of J_sum * weights.T
            return np.einsum('ij,kj->ik', J_sum, self.weights)
        activations = self.activations[:, column]
        # If we ignore the diagonal for a second, the J_soft matrix corresponds to the outer product of the
        # activation vector with itself (with a negative prefix)
        J_soft = - np.einsum('i,j->ij', activations, activations)
        # The above matrix is basically -s_m * s_n for index (m,n). However for the diagonal, we now have
        # -s_n^2 (m=n), rather than s_n - s_n^2
        # Thus we are lacking one s_n, so all we need to do is add it back
        J_soft = J_soft + np.diag(activations)
        # Now the soft matrix is correct, and we can return it
        return J_soft

    def get_weight_jacobian(self, column: int = 0) -> np.ndarray:
        """This method will calculate the jacobian gradient matrix with respect to the weights of the current layer.
        More explicitly, with respect to the weights going INTO the current layer.
        This corresponds to J^Z_W (Z current layer; W going into layer Z)

        Args:
            column (int): Integer for indexing a particular result (used for minibatches)

        Returns:
            np.ndarray: Will return the jacobian matrix w.r.t. weights for the current layer
        """
        diag_J_sum = self.da_func(self.activations[:, column])
        # Return object corresponds to J-hat; einsum is outer product
        return np.einsum('i,j->ij', self.prv_layer_inputs, diag_J_sum)


class NeuralNetwork:

    def __init__(self, config: Dict, batch_size: int = 1) -> None:
        """
        Taking in parameters for neural network.

        Args:
            layers_config (Dict):
                Dictionary containing necessary parameters for network configuration

            batch_size (int):
                Integer describing the number of examples to operate on simultaneously

        """
        use_softmax = config['Softmax']
        self.batch_size: int = batch_size
        self.hidden_layers: List[Layer] = []
        n_hl: int = config['Hidden layers']
        hl_neurons: List[int] = config['HL Neurons']
        hl_funcs: List[Callable] = config['HL Activation functions']
        hl_d_funcs: List[Callable] = config['HL Derivative functions']
        self.l_func: Callable = config['Loss function']
        self.dl_func: Callable = config['Loss derivative function']
        # Weights: nxm, m current, n previous
        n: int = config['Inputs']
        for i in range(n_hl):
            # Nodes in the current hidden layer
            m = hl_neurons[i]
            func, d_func = hl_funcs[i], hl_d_funcs[i]
            w_shape, b_shape = (n, m), (m, 1)
            weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
            biases = np.zeros(shape=b_shape)
            self.hidden_layers.append(Layer(func, d_func, weights, biases))
            # Set the next n value for correct dimensions in the next weight matrix
            n = m
        m: int = config['Outputs']
        output_func: Callable = config['Output function']
        d_output_func: Callable = config['Output derivative function']
        w_shape, b_shape = (n, m), (m, 1)
        weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
        biases = np.zeros(shape=b_shape)
        final_layer = Layer(output_func, d_output_func, weights, biases)
        # If we use softmax, then treat the output layer as a hidden layer, and add a final layer
        # with softmax as the activation function and the identity matrix as weights
        if not use_softmax:
            self.output_layer = final_layer
        else:
            self.hidden_layers.append(final_layer)
            weights, biases = np.identity(m), 0
            func, d_func = output_func, d_output_func
            self.output_layer = Layer(
                func, d_func, weights, biases, softmax=True)

    def forward_pass(self, network_inputs: np.ndarray) -> np.ndarray:
        # Get the first layer activations for use in the later layers
        current_input = network_inputs
        # Iterate through the rest of the layers, feeding the previous output as the next input
        for layer in self.hidden_layers:
            current_input = layer.calculate_input(current_input)
        final_input = self.output_layer.calculate_input(current_input)
        return final_input
        # Note to self: to extract the column of a numpy matrix, simply use matrix[:,<index of column>]

    def get_loss_jacobian(self, targets: np.ndarray) -> np.ndarray:
        predictions = self.output_layer.activations
        return self.dl_func(predictions, targets)

    def backpropagation(self, targets: np.ndarray) -> None:
        # We start off by getting the jacobian matrix w.r.t. the loss function
        J_loss = self.get_loss_jacobian(targets)
        for i in range(len(self.hidden_layers), 0, -1):
            layer = self.hidden_layers[i]



