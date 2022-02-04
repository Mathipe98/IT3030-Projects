"""
Here are some module docstrings
"""
from ctypes import Union
from typing import Callable, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from nn_functions import d_cross_entropy, d_identity, d_mse, d_relu, d_sigmoid, d_tanh
from nn_functions import softmax

np.set_printoptions(threshold=20)

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
                 softmax: bool = False) -> None:
        """Constructor for the Layer-object. Sets necessary parameters for use in
        forward pass, backpropagation, and weight-updates.

        Args:
            a_func (Callable): Activation function
            da_func (Callable): Derivative of activation function
            weights (np.ndarray): Weights from the previous layer going into this layer
            biases (np.ndarray): Bias-weights from previous layer into this layer
            softmax (bool, optional): Whether or not to use softmax. Defaults to False.
        """
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

    def get_output_jacobian(self) -> np.ndarray:
        """This method will return the jacobian matrix that represents the derivatives of
        the inputs to this layer with respect to the activations from the previous layer.
        From the examples, this corresponds to J^Z_Y.

        Returns:
            np.ndarray: Matrix where index (m,n) corresponds to the derivative of output of node m times
                        the weight going from node n (in the previous layer) to node m in the current layer
        """
        if not self.softmax:
            J_sum = np.diag(self.da_func(self.activations[:, 0]))
            # Einsum corresponds to inner product of J_sum * weights.T
            return np.einsum('ij,kj->ik', J_sum, self.weights)
        activations = self.activations[:, 0]
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
        """Method for updating the weights going into the current layer

        Args:
            jacobian (np.ndarray): Matrix with derivatives of weights w.r.t. the loss
            lr (float): Learning rate of the neural net
        """
        assert jacobian.shape == self.weights.shape, "Jacobian weight matrix and weights are not the same shape"
        self.weights = self.weights - lr * jacobian

    def update_biases(self, jacobian: np.ndarray, lr: float) -> None:
        """Same as weight-update method, only for biases

        Args:
            jacobian (np.ndarray): Matrix with derivatives of biases w.r.t. the loss
            lr (float): Learning rate of the neural net
        """
        assert jacobian.shape == self.biases.shape
        self.biases = self.biases - lr * jacobian

    def __eq__(self, __o: object) -> bool:
        """Overwritten equal-method. To check whether two layers are
        considered identical (for handling softmax).

        Args:
            __o (object): Object to compare with

        Returns:
            bool: Whether or not the two layers are considered equal
        """
        return np.all(self.weights == __o.weights)


class NeuralNetwork:

    def __init__(self, inputs: int, outputs: int, lr: float,
                 epochs: int, wreg: str, wreg_lr: float,
                 n_hl: int, hl_neurons: List,
                 hl_funcs: List, hl_wranges: List, output_func: Callable,
                 output_wrange: Tuple, l_func: Callable, use_softmax: bool = False,
                 verbose: bool = True) -> None:
        """Constructor for the entire neural network. Sets up all necessary parameters and
        hyperparameters for the network.

        Args:
            inputs (int): Number of neurons in the first layer
            outputs (int): Number of neurons in the last layer
            lr (float): Learning rate used in backprop
            epochs (int): Number of times to iterate through training data in backprop
            wreg (str): Type og weight-regularization
            wreg_lr (float): Regularization rate
            n_hl (int): Number of hidden layers
            hl_neurons (List): List containing the number of neurons in each hidden layer
            hl_funcs (List): List containing the activation function for each hidden layer
            hl_wranges (List): List containing the initial weight-ranges for each hidden layer
            output_func (Callable): Activation function of the output layer
            output_wrange (Tuple): Initial weight ranges for the output layer
            l_func (Callable): Loss function
            use_softmax (bool, optional): Whether or not to use softmax on the output. Defaults to False.
            verbose (bool, optional): Variable controlling whether or not to print details. Defaults to True.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.lr = lr
        self.wreg = wreg
        self.wreg_func: Callable = self.L1_reg if wreg == 'L1' else self.L2_reg
        self.wreg_lr = wreg_lr
        self.epochs = epochs
        n_hl = n_hl
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
            w_range = hl_wranges[i]
            weights = np.random.uniform(low=w_range[0], high=w_range[1], size=w_shape)
            biases = np.zeros(shape=b_shape)
            self.hidden_layers.append(Layer(func, d_func, weights, biases))
            # Set the next n value for correct dimensions in the next weight matrix
            n = m
        m: int = outputs
        d_output_func: Callable = function_map[output_func.__name__]
        w_shape = (n, m)
        b_shape = (m, 1)
        weights = np.random.uniform(low=output_wrange[0], high=output_wrange[1], size=w_shape)
        biases = np.zeros(shape=b_shape)
        final_layer = Layer(output_func, d_output_func, weights, biases)
        # If we use softmax, then treat the output layer as a hidden layer, and add a final layer
        # with softmax as the activation function and the identity matrix as weights
        self.use_softmax = use_softmax
        if not self.use_softmax:
            self.output_layer = final_layer
        else:
            self.hidden_layers.append(final_layer)
            weights, biases = np.identity(m), 0
            func, d_func = softmax, function_map["softmax"]
            self.output_layer = Layer(
                func, d_func, weights, biases, softmax=True)
        self.verbose = verbose
        self.training_loss: List[float] = []
        self.validation_loss: List[float] = []
        self.testing_loss: List[float] = []

    def L1_reg(self) -> float:
        """Method that returns the value of the weight-regularization L1

        Returns:
            float: The aforementioned regularization value
        """     
        total = 0
        total += np.sum(self.output_layer.weights)
        for layer in self.hidden_layers:
            total += np.sum(layer.weights)
        return total

    def L2_reg(self) -> float:
        """Method that returns the value of the weight-regularization L2

        Returns:
            float: The aforementioned regularization value
        """
        total = 0
        total += np.sum(self.output_layer.weights ** 2)
        for layer in self.hidden_layers:
            total += np.sum(layer.weights ** 2)
        return 1/2 * total

    def forward_pass(self, network_inputs: np.ndarray) -> None:
        """Method that takes an input and passes it forward through the network,
        where each successive value is calculated via matrix-vector multiplication.

        Args:
            network_inputs (np.ndarray): Array of values being fed into the network
        """
        # Get the first layer activations for use in the later layers
        current_input = network_inputs
        # Iterate through the rest of the layers, feeding the previous output as the next input
        for layer in self.hidden_layers:
            current_input = layer.calculate_input(current_input)
        self.output_layer.calculate_input(current_input)

    
    def backpropagation(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, float]:
        """Main algorithm of the neural network.
        This algorithm takes in a concrete training example, and a target for this example.
        It feeds the example forward in the network to generate an output, and then calculates
        the loss of this output. It uses this loss value to propagate its derivative w.r.t to
        various weights and layers backwards through the network.

        Args:
            inputs (np.ndarray): Array containing the inputs to the network
            targets (np.ndarray): Array containing the correct classification of the example

        Returns:
            Tuple[np.ndarray, float]: Tuple containing the predicted output, and the loss value (for logging purposes)
        """
        predictions = self.predict(inputs)
        predictions_loss = self.l_func(predictions,
                            targets) + self.wreg_lr * self.wreg_func()
        first_layer = self.output_layer
        # Get loss w.r.t. output, then layer w.r.t. weights, and finally calculate loss w.r.t. weights (J_lw)
        J_lz = self.get_loss_jacobian(targets)
        if self.use_softmax:
            J_lz = np.dot(J_lz.T, first_layer.get_output_jacobian()).T
            first_layer = self.hidden_layers[-1]
        J_hat_zw = first_layer.get_J_hat_zw()
        J_lw = J_hat_zw * J_lz.T
        # Now take regularization into account
        if self.wreg == 'L1':
            J_lw = J_lw + self.wreg_lr * np.sign(first_layer.weights)
        else:
            J_lw = J_lw + self.wreg_lr * first_layer.weights
        # Simultaneously, get the jacobian matrix for the biases
        J_lb = first_layer.get_bias_jacobian(J_lz)
        # Get the jacobian of the output before updating the weights
        J_zy = first_layer.get_output_jacobian()
        # Now update weights and biases
        first_layer.update_weights(J_lw, self.lr)
        first_layer.update_biases(J_lb, self.lr)
        # Now propagate backwards through the hidden layers and repeat
        J_ly = np.dot(J_lz.T, J_zy)
        for current_layer in reversed(self.hidden_layers):
            # Workaround for softmax; if softmax, then we've added 1 more layer, and already dealt with
            # the output layer. Thus skip it so we don't use it twice
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
        return predictions, predictions_loss
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Method that passes an input forward in the network, while 
        simultaneously returning the output of the network.

        Args:
            inputs (np.ndarray): Inputs into the neural network

        Returns:
            np.ndarray: The predicted produced by the network
        """
        self.forward_pass(inputs)
        return self.output_layer.activations

    def get_loss_jacobian(self, targets: np.ndarray) -> np.ndarray:
        """Method that returns a Jacobian matrix that corresponds to
        the derivative of the loss function w.r.t the outputs of the
        network (i.e. activations of the final layer)

        Args:
            targets (np.ndarray): Array containing the correct classification

        Returns:
            np.ndarray: Aforementioned matrix
        """
        predictions = self.output_layer.activations
        return self.dl_func(predictions, targets, self.output_layer.a_func.__name__ == "sigmoid")

    def train(self, training_data: List[np.ndarray], training_targets: List[np.ndarray],
              validation_data: List[np.ndarray], validation_targets: List[np.ndarray]) -> None:
        """Method that initiates the process of feed-forwarding and backpropagation.
        The method iterates through all training examples and performs this process for an
        epoch number of times. Simultaneously, the method also calculates the average validation
        loss to compare the results to previously unseen data.

        Args:
            training_data (List[np.ndarray]): Array of training inputs
            training_targets (List[np.ndarray]): Array of training solutions
            validation_data (List[np.ndarray]): Array of validation inputs
            validation_targets (List[np.ndarray]): Array of validation solutions
        """
        assert len(training_data) == len(
            training_targets), "Training data and solutions must have same length"
        assert np.array(
            training_data[0]).shape[0] == self.inputs, "Inputs to the network much match config parameter 'Inputs' (array shape does not match)"
        assert np.array(training_targets[0]).shape[0] == self.outputs
        if self.verbose:
            print("Starting epoch iteration")
        for k in range(self.epochs):
            loss_this_epoch = 0
            for i in range(len(training_data)):
                inputs = np.array(training_data[i])
                targets = np.array(training_targets[i])
                predictions, loss = self.backpropagation(inputs, targets)
                loss_this_epoch += loss
                if i == 0 and self.verbose:
                    print(f"Inputs:\n {inputs}")
                    print(f"Targets:\n {targets}")
                    print(f"Predictions:\n {predictions}")
                    print(f"Loss:\t {loss}")
            if (k+1) % int(self.epochs * 0.1) == 0:
                print(f"Progress: {int((k+1)/self.epochs * 100)}%")
            # Add the average training loss for this epoch
            self.training_loss.append(loss_this_epoch / len(training_data))
            # At the same time, get the average loss on the validation set
            self.calculate_validation_loss(validation_data, validation_targets)

    def calculate_validation_loss(self, validation_data: List[np.ndarray], validation_targets: List[np.ndarray]) -> None:
        """Method for calculating the average loss on the validation set during an epoch.

        Args:
            validation_data (List[np.ndarray]): Array of validation inputs
            validation_targets (List[np.ndarray]): Array of validation solutions
        """
        total = 0
        for i in range(len(validation_data)):
            inputs = validation_data[i]
            targets = validation_targets[i]
            predictions = self.predict(inputs)
            loss = self.l_func(predictions, targets) + self.wreg_lr * self.wreg_func()
            total += loss
        self.validation_loss.append(total / len(validation_data))
    
    def calculate_testing_loss(self, testing_data: List[np.ndarray], testing_targets: List[np.ndarray]) -> None:
        """Method for calculating the average loss on the testing set during an epoch.

        Args:
            testing_data (List[np.ndarray]): Array of testing inputs
            testing_targets (List[np.ndarray]): Array of testing solutions
        """
        for i in range(len(testing_data)):
            inputs = testing_data[i]
            targets = testing_targets[i]
            predictions = self.predict(inputs)
            loss = self.l_func(predictions, targets) + self.wreg_lr * self.wreg_func()
            self.testing_loss.append(loss)

    def visualize_losses(self) -> None:
        """Method for visualizing the various calculated losses during the main training
        cycle. Visualizes training loss, validation loss, and testing loss.
        (Also saves the figures to local directory.)
        """
        x_axis = range(0, self.epochs)
        test_x_axis = range(self.epochs, self.epochs + len(self.testing_loss))
        plt.plot(x_axis, self.training_loss, label="Training loss")
        plt.plot(x_axis, self.validation_loss, label="Validation loss")
        plt.plot(test_x_axis, self.testing_loss, label="Testing loss")
        plt.xlabel("Epochs")
        plt.ylabel(f"Loss ({self.l_func.__name__})")
        plt.legend()
        plt.savefig('./Figures/Network_losses.png')
        plt.show()
    
    def visualize_testing_accuracy(self, testing_data: List[np.ndarray], testing_targets: List[np.ndarray]) -> None:
        """Method for visualizing the general accuracy of the network at predicting the correct labels.
        Since we're dealing with one-hot vectors, we take the argmax of the network output and check
        if this corresponds to the solution.

        Args:
            testing_data (List[np.ndarray]): Array of testing inputs
            testing_targets (List[np.ndarray]): Array of testing solutions
        """
        correct = 0
        incorrect = 0
        for i in range(len(testing_data)):
            inputs = testing_data[i]
            targets = testing_targets[i]
            predictions = self.predict(inputs)
            if np.argmax(predictions) == np.argmax(targets):
                correct += 1
            else:
                incorrect += 1
        categories = ["Correct", "Incorrect"]
        values = [correct, incorrect]
        plt.bar(categories, values, width = 0.4)
        plt.xlabel("Classifications")
        plt.ylabel("N examples")
        plt.title("Neural Network classification on testing data")
        plt.savefig("./Figures/Network_accuracy.png")
        plt.show()
