"""
Here are some module docstrings
"""
from ctypes import Union
from typing import Callable, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from nn_functions import d_cross_entropy, d_identity, d_mse, d_relu, d_sigmoid, d_tanh


# Dict containing the mappings from functions to their derivatives
function_map = {
    "sigmoid": d_sigmoid,
    "relu": d_relu,
    "tanh": d_tanh,
    "identity": d_identity,
    # We only need the derivative of the softmax function with respect to the same elements, which corresponds to the same functionality as the d_sigmoid function
    "softmax": d_sigmoid,
    "cross_entropy": d_cross_entropy,
    "mse": d_mse
}


class Layer:
    """
    Class that represents a single layer in an Artifical Neural Network (ANN)
    """

    def __init__(self, a_func: Callable, da_func: Callable,
                 weights: np.ndarray, biases: np.ndarray,
                 softmax: bool = False, is_final_layer: bool = False) -> None:
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
        self.is_final_layer = is_final_layer

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
        # Pass the result into the activation function
        activations = self.a_func(output)
        # Check if we need to expand the dimensions of the results
        if len(activations.shape) == 1:
            activations = activations[:, np.newaxis]
        if len(node_values.shape) == 1:
            node_values = node_values[:, np.newaxis]
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
        J_soft = np.einsum('i,j->ij', activations, activations) * -1
        # The above matrix is basically -s_m * s_n for index (m,n). However for the diagonal, we now have
        # -s_n^2 (m=n), rather than s_n - s_n^2. Thus we are lacking one s_n, so all we need to do is add it back
        J_soft = J_soft + np.diag(activations)
        return J_soft

    def get_J_hat_zw(self, column: int = 0) -> np.ndarray:
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
        return np.einsum('i,j->ij', self.prv_layer_inputs[:, column], diag_J_sum)

    def get_bias_jacobian(self, J_lz: np.ndarray) -> np.ndarray:
        """This method will return a vector with values to update the weights
        from the bias node in the previous layer.
        Since derivatives of the biases are always 1, this will simply correspond
        to multiplying the J_lz matrix by a column vector of 1's.
        This is simply summing the rows of the original matrix

        Args:
            J_lz (np.ndarray): The jacobian matrix containing the derivatives of the loss
            w.r.t. the output

        Returns:
            np.ndarray: A vector of derivatives of the loss w.r.t. the biases
        """
        return np.einsum('ij->i', J_lz).reshape(J_lz.shape[0], 1)

    def update_weights(self, jacobian: np.ndarray, lr: float) -> None:
        assert jacobian.shape == self.weights.shape, "Jacobian weight matrix and weights are not the same shape"
        self.weights = self.weights - lr * jacobian

    def update_biases(self, jacobian: np.ndarray, lr: float) -> None:
        assert jacobian.shape == self.biases.shape
        self.biases = self.biases - lr * jacobian

    def __eq__(self, __o: object) -> bool:
        return np.all(self.weights == __o.weights)


class NeuralNetwork:

    def __init__(self, inputs: int, outputs: int, lr: float,
                 wreg: str, wreg_lr: float,
                 n_hl: int, hl_neurons: List,
                 hl_funcs: List, output_func: Callable,
                 l_func: Callable, softmax: bool = False) -> None:
        """
        Taking in parameters for neural network.

        Args:
            layers_config (Dict):
                Dictionary containing necessary parameters for network configuration

            batch_size (int):
                Integer describing the number of examples to operate on simultaneously

        """
        self.inputs = inputs
        self.outputs = outputs
        self.lr = lr
        self.wreg = wreg
        self.wreg_func: Callable = self.L1_reg if wreg == 'L1' else self.L2_reg
        self.wreg_lr = wreg_lr
        n_hl = n_hl
        hl_neurons = hl_neurons
        hl_funcs = hl_funcs
        self.l_func = l_func
        self.dl_func: Callable = function_map[self.l_func.__name__]
        # Weights: nxm, m current, n previous
        n = self.inputs
        self.hidden_layers: List[Layer] = []
        for i in range(n_hl):
            # Nodes in the current hidden layer
            m = hl_neurons[i]
            func = hl_funcs[i]
            d_func = function_map[func.__name__]
            w_shape = (n, m)
            b_shape = (m, 1)
            weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
            biases = np.zeros(shape=b_shape)
            self.hidden_layers.append(Layer(func, d_func, weights, biases))
            # Set the next n value for correct dimensions in the next weight matrix
            n = m
        m: int = outputs
        d_output_func: Callable = function_map[output_func.__name__]
        w_shape = (n, m)
        b_shape = (m, 1)
        weights = np.random.uniform(low=-0.1, high=0.1, size=w_shape)
        biases = np.zeros(shape=b_shape)
        final_layer = Layer(output_func, d_output_func, weights, biases)
        # If we use softmax, then treat the output layer as a hidden layer, and add a final layer
        # with softmax as the activation function and the identity matrix as weights
        self.softmax: bool = output_func.__name__ == 'softmax'
        if not self.softmax:
            self.output_layer = final_layer
        else:
            self.hidden_layers.append(final_layer)
            weights, biases = np.identity(m), 0
            func, d_func = output_func, d_output_func
            self.output_layer = Layer(
                func, d_func, weights, biases, softmax=True)
        self.training_loss: List[float] = []

    
    def L1_reg(self) -> float:
        total = 0
        total += np.sum(self.output_layer.weights)
        for layer in self.hidden_layers:
            total += np.sum(layer.weights)
        return total

    
    def L2_reg(self) -> float:
        total = 0
        total += np.sum(self.output_layer.weights ** 2)
        for layer in self.hidden_layers:
            total += np.sum(layer.weights ** 2)
        return 1/2 * total

    def forward_pass(self, network_inputs: np.ndarray) -> None:
        # Get the first layer activations for use in the later layers
        current_input = network_inputs
        # Iterate through the rest of the layers, feeding the previous output as the next input
        for layer in self.hidden_layers:
            current_input = layer.calculate_input(current_input)
        self.output_layer.calculate_input(current_input)

    def backpropagation(self, targets: np.ndarray) -> None:
        # Debugging
        assert targets.shape == self.output_layer.activations.shape, "Targets must match the output layer mafakka"
        first_layer = self.output_layer
        # We start off by getting the jacobian matrix of the loss function w.r.t. the output
        J_lz = self.get_loss_jacobian(targets)
        # Something must be fixed here to take softmax into account
        if self.softmax:
            J_lz = np.dot(J_lz.T, first_layer.get_output_jacobian()).T
            first_layer = self.hidden_layers[-1]
        J_hat_zw = first_layer.get_J_hat_zw()
        # Finally get the jacobian matrix for the actual weights in the current layer
        J_lw = J_hat_zw * J_lz.T
        # Now take regularization into account
        if self.wreg == 'L1':
            J_lw = J_lw + self.wreg_lr * np.sign(first_layer.weights)
        else:
            J_lw = J_lw + self.wreg_lr * first_layer.weights
        # Simultaneously, get the jacobian matrix for the biases
        J_lb = first_layer.get_bias_jacobian(J_lz)
        # Debugging
        assert J_lw.shape == first_layer.weights.shape
        # Try to get the output jacobian BEFORE updating the weights
        J_zy = first_layer.get_output_jacobian()
        first_layer.update_weights(J_lw, self.lr)
        first_layer.update_biases(J_lb, self.lr)
        # Now propagate backwards through the hidden layers
        J_ly = np.dot(J_lz.T, J_zy)
        for current_layer in reversed(self.hidden_layers):
            # Just a workaround for taking softmax into account; skip if we've already dealt with it
            if current_layer == first_layer:
                continue
            J_hat_yw = current_layer.get_J_hat_zw()
            J_lv = J_ly * J_hat_yw
            if self.wreg == 'L1':
                J_lv = J_lv + self.wreg_lr * np.sign(current_layer.weights)
            else:
                J_lv = J_lv + self.wreg_lr * current_layer.weights
            J_lb = first_layer.get_bias_jacobian(J_ly.T)
            assert J_lv.shape == current_layer.weights.shape
            J_yx = current_layer.get_output_jacobian()
            current_layer.update_weights(J_lv, self.lr)
            current_layer.update_biases(J_lb, self.lr)
            J_ly = np.dot(J_ly, J_yx)

    def get_loss_jacobian(self, targets: np.ndarray) -> np.ndarray:
        predictions = self.output_layer.activations
        return self.dl_func(predictions, targets, self.output_layer.a_func.__name__ == "sigmoid")

    def train(self, training_data: List[np.ndarray], training_targets: List[np.ndarray]) -> None:
        epochs = 100
        # For some reason, storing np arrays in lists, and then getting them out from the list again
        # turns the arrays into lists. Sucks, but I gotta turn them back into arrays another time
        assert len(training_data) == len(
            training_targets), "Training data and solutions must have same length"
        assert np.array(
            training_data[0]).shape[0] == self.inputs, "Inputs to the network much match config parameter 'Inputs' (array shape does not match)"
        assert np.array(training_targets[0]).shape[0] == self.outputs
        for k in range(epochs):
            for i in range(len(training_data)):
                inputs = np.array(training_data[i])
                targets = np.array(training_targets[i])
                self.forward_pass(inputs)
                loss = self.l_func(self.output_layer.activations, targets) + self.wreg_lr * self.wreg_func()
                if i == 0 and k == 0:
                    self.training_loss.append(loss)
                self.backpropagation(targets)
            if (k+1) % int(epochs * 0.1) == 0:
                print(f"Progress: {int((k+1)/epochs * 100)}%")
            self.training_loss.append(loss)

    def predict(self, example: np.ndarray) -> np.ndarray:
        assert example.shape[0] == self.inputs, "Cannot predict data that doesn't match input shape"
        self.forward_pass(example)
        print(f"Output layer function: {self.output_layer.a_func.__name__}")
        return self.output_layer.activations

    def visualize_training_losses(self) -> None:
        x_axis = range(0, len(self.training_loss))
        plt.plot(x_axis, self.training_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.savefig('./Figures/TRAINING_LOSS_FIRST_ATTEMPT.png')
        plt.show()

# NOTES FOR STUDASS:
#   1. Gradient DESCENT is not working. ASCENT however is (+ rather than -)
#   2. Cross-entropy is fucked, crazy bad losses when using that function (MSE is fine) <- wtf is goin on
#   3. What are the required parameters for figure-generation? Confusing (flatten + height/width compared to nxn)
