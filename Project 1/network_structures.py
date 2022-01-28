"""
Here are some module docstrings
"""
from ctypes import Union
from typing import Callable, Dict, List, Tuple
import numpy as np

from nn_functions import d_cross_entropy, d_identity, d_mse, d_relu, d_sigmoid, d_tanh
np.random.seed(123)

# Dict containing the mappings from functions to their derivatives
function_map = {
    "sigmoid": d_sigmoid,
    "relu": d_relu,
    "tanh": d_tanh,
    "identity": d_identity,
    "softmax": None,
    "cross_entropy": d_cross_entropy,
    "mse": d_mse
}


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
        self.n_nodes = weights.shape[1]
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
        # weights_t = np.einsum('ij->ji', self.weights) <- this is slower than just weight.T
        # Multiply the transposed weight-matrix by the node-values
        output = np.einsum('ij...,j...->i...', self.weights.T, node_values)
        # Now add the bias to the output
        output = output + self.biases
        # Pass the result into the activation function, and return it
        activations = self.a_func(output)
        # Check if we need to expand the dimensions of the results
        if len(activations.shape) == 1:
            activations = activations[:,np.newaxis]
        if len(node_values.shape) == 1:
            node_values = node_values[:,np.newaxis]
        # Now store the values for use in backprop
        self.activations, self.prv_layer_inputs = activations, node_values
        return activations

    def get_output_jacobian(self, column: int = 0) -> np.ndarray:
        """This method will return the jacobian matrix that represents the derivatives of
        the inputs to this layer with respect to the activations from the previous layer.
        From the examples, this corresponds to J^Z_Y.

        Notes:
        The jacobian matrix will have shape (m, n), where m is the number of nodes in the current layer,
        and n is the number of nodes in the previous layer.

        Args:
            column (int): Index that extracts the particular column of the result in the current layer.

        Returns:
            np.ndarray: Matrix where index (m,n) corresponds to the derivative of output of node m times
                        the weight going from node n (in the previous layer) to node m in the current layer
        """
        if not self.softmax:
            J_sum = np.diag(self.da_func(self.activations[:, column]))
            # Einsum corresponds to inner product of J_sum * weights.T
            return np.einsum('ij,kj->ik', J_sum, self.weights)
        activations = self.activations[:, column]
        # If we ignore the diagonal for a second, the J_soft matrix corresponds to the outer product of the
        # activation vector with itself (with a negative prefix)
        J_soft = - np.einsum('i,j->ij', activations, activations)
        # The above matrix is basically -s_m * s_n for index (m,n). However for the diagonal, we now have
        # -s_n^2 (m=n), rather than s_n - s_n^2. Thus we are lacking one s_n, so all we need to do is add it back
        J_soft = J_soft + np.diag(activations)
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
        # print(f"Input")
        diag_J_sum = self.da_func(self.activations[:, column])
        # Return object corresponds to J-hat; einsum is outer product
        return np.einsum('i,j->ij', self.prv_layer_inputs[:,column], diag_J_sum)

    def update_weights(self, jacobian: np.ndarray, column: int=0, lr: float=1e-2) -> None:
        prv = self.prv_layer_inputs[:,column]
        



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
        self.batch_size: int = batch_size
        self.hidden_layers: List[Layer] = []
        n_hl: int = config['Hidden layers']
        hl_neurons: List[int] = config['HL Neurons']
        hl_funcs: List[Callable] = config['HL Activation functions']
        self.l_func: Callable = config['Loss function']
        self.dl_func: Callable = function_map[self.l_func.__name__]
        # Weights: nxm, m current, n previous
        n: int = config['Inputs']
        for i in range(n_hl):
            # Nodes in the current hidden layer
            m = hl_neurons[i]
            func = hl_funcs[i]
            d_func = function_map[func.__name__]
            w_shape = (n, m)
            b_shape = (m,1)# (m,) if self.batch_size == 1 else (m,1)
            weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
            biases = np.zeros(shape=b_shape)
            self.hidden_layers.append(Layer(func, d_func, weights, biases))
            # Set the next n value for correct dimensions in the next weight matrix
            n = m
        m: int = config['Outputs']
        output_func: Callable = config['Output function']
        d_output_func: Callable = function_map[output_func.__name__]
        w_shape = (n, m)
        b_shape = (m,1) # (m,) if self.batch_size == 1 else (m,1)
        weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
        biases = np.zeros(shape=b_shape)
        final_layer = Layer(output_func, d_output_func, weights, biases)
        # If we use softmax, then treat the output layer as a hidden layer, and add a final layer
        # with softmax as the activation function and the identity matrix as weights
        if output_func.__name__ != 'softmax':
            self.output_layer = final_layer
        else:
            self.hidden_layers.append(final_layer)
            weights, biases = np.identity(m), 0
            func, d_func = output_func, d_output_func
            self.output_layer = Layer(
                func, d_func, weights, biases, softmax=True)

    def forward_pass(self, network_inputs: np.ndarray) -> None:
        # Get the first layer activations for use in the later layers
        current_input = network_inputs
        # Iterate through the rest of the layers, feeding the previous output as the next input
        for layer in self.hidden_layers:
            current_input = layer.calculate_input(current_input)
        self.output_layer.calculate_input(current_input)

    def get_loss_jacobian(self, targets: np.ndarray) -> np.ndarray:
        predictions = self.output_layer.activations
        return self.dl_func(predictions, targets).T

    def backpropagation(self, network_inputs: np.ndarray, targets: np.ndarray) -> None:
        # First we must start by passing the inputs forward in the network
        self.forward_pass(network_inputs)
        assert targets.shape == self.output_layer.activations.shape, "Targets must match the output layer mafakka"
        # We start off by getting the jacobian matrix of the loss function w.r.t. the output
        J_lz = self.get_loss_jacobian(targets)
        print(f"\nJ^L_Z:\n {J_lz}", end="\n\n")
        current_layer = self.output_layer
        # print(current_layer.activations, end="\n\n")
        J_hat_zw = current_layer.get_weight_jacobian()
        print(f"J^Z_W:\n {J_hat_zw}", end="\n\n")
        # Finally get the jacobian matrix for the actual weights in the current layer
        J_lw = J_lz * J_hat_zw
        print(f"J^L_W:\n {J_lw}", end="\n\n")
        assert J_lw.shape == current_layer.weights.shape
        self.update_weights(J_lw)
        J_zy = current_layer.get_output_jacobian()
        print(f"J^Z_Y:\n {J_zy}", end="\n\n")
        # Now start the actual backpropagation
        for current_layer in reversed(self.hidden_layers):
            J_lz = np.dot(J_lz, J_zy)
            print(f"\nNext layer J^L_hidden layer:\n {J_lz}", end="\n\n")
            # print(current_layer.activations, end="\n\n")
            J_hat_zw = current_layer.get_weight_jacobian()
            print(f"J^Z_W:\n {J_hat_zw}", end="\n\n")
            # Finally get the jacobian matrix for the actual weights in the current layer
            J_lw = J_lz * J_hat_zw
            print(f"J^L_W:\n {J_lw}", end="\n\n")
            assert J_lw.shape == current_layer.weights.shape
            self.update_weights(J_lw)
            J_zy = current_layer.get_output_jacobian()
            print(f"J^Z_Y:\n {J_zy}", end="\n\n")

    
    
        




