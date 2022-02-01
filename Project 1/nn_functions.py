import numpy as np

# Here we will define all the activation functions we can use


def sigmoid(input_vector: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function, rest of docstring unnecessary seeing as it
    becomes larger than the function itself.
    """
    return 1 / (1 + np.exp(-input_vector))


def relu(input_vector: np.ndarray) -> np.ndarray:
    """
    Classic ReLu function with (somewhat) optimized time usage.
    """
    return np.maximum(input_vector, 0, input_vector)


def tanh(input_vector: np.ndarray) -> np.ndarray:
    return np.tanh(input_vector)


def identity(input_vector: np.ndarray) -> np.ndarray:
    return input_vector


def softmax(input_vector: np.ndarray) -> np.ndarray:
    """
    Classic softmax function for probability distributions.
    """
    return np.exp(input_vector) / np.sum(np.exp(input_vector), axis=0)

# And here are the derivative of the same activation functions


def d_sigmoid(input_vector: np.ndarray) -> np.ndarray:
    def f(x): return x * (1 - x)
    result = f(input_vector)
    return result


def d_relu(input_vector: np.ndarray) -> np.ndarray:
    """
    ReLu derivative. Where elements are <= 0, return 0. Else return 1.
    Note: technically undefined at 0, but we'll probably never get exactly 0 anyway.
    But if we do, then setting it to 0 won't make much difference.
    """
    return np.where(input_vector <= 0, 0, 1)


def d_tanh(input_vector: np.ndarray) -> np.ndarray:
    """
    Tanh derivative = 1 - tanh^2
    """
    def f(x): return 1 - x ** 2
    return f(input_vector)


def d_identity(input_vector: np.ndarray) -> np.ndarray:
    return np.ones(shape=input_vector.shape)

# Note: no single derivative function for softmax because it's more complex and not directly applicable in general

# Here we define the loss functions we will use (MSE and cross-entropy)

# IMPORTANT NOTE FOR ALL THESE FUNCTIONS:
#      For all the loss functions and their corresponding derivatives, the following logic is applied:
#           When a SINGLE case (example) is sent in, then it will be a vector with shape (m,) for m nodes
#           in the final output layer.
#           If SEVERAL cases (i.e. SGD) are sent in, then it will be a MATRIX with shape (m,n) where n is the
#           number of minibatches.
#     The output of the LOSS functions will be of size n, where n is the number of cases sent in. I.e. if 1 case
#     is sent in (i.e. input shape (m,)), then output is an array with shape (1,). Similarly, 5 in => (5,) out
#


def cross_entropy(predictions: np.ndarray, targets: np.ndarray, sigmoid: bool = False) -> np.ndarray:
    """Cross-entropy loss function.
    Uses einsum to compute column-wise dot products, and reshapes inputs to match this function.

    Args:
        input_vector (np.ndarray): (m,n) array of predictions where m is the nodes in the output layer, and n is the batch size
        target_vector (np.ndarray): (m,n) array of targets; same structure as predictions
        sigmoid (bool, optional): Boolean value to clip targets if sigmoid is used. Defaults to False.

    Returns:
        float: Array of cross-entropy loss of all predictions compared to target
    """
    assert predictions.shape == targets.shape, "Predictions and targets must have same shapes"
    epsilon = 1e-12
    def safe_log(x): return 0 if x == 0 else np.log(x)
    # # If sigmoid -> binary values are 0.9 and 0.1
    # if sigmoid:
    #     targets = np.clip(targets, 0.1, 0.9)
    # If our input is 1-dimensional:
    if len(predictions.shape) == 1:
        # Make the vectors 2D to work with einsum in the same way as SGD
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    loss = 0
    for col in range(predictions.shape[0]):
        for row in range(predictions.shape[1]):
            predicted_value = predictions[col][row]
            target_value = targets[col][row]
            v1 = target_value * safe_log(predicted_value)
            v2 = (1-target_value) * safe_log(1-predicted_value)
            loss += v1 + v2
    return -loss/predictions.shape[1]
    


def mse(predictions: np.ndarray, targets: np.ndarray, sigmoid: bool = False) -> np.ndarray:
    if sigmoid:
        targets = np.clip(targets, 0.1, 0.9)
    return np.mean(np.power(predictions-targets, 2))

# And now the corresponding derivatives


def d_cross_entropy(predictions: np.ndarray, targets: np.ndarray, sigmoid: bool = False) -> np.ndarray:
    """This function will calculate the derivative of the loss function with respect to the output, given
    that the output has already been calculated.

    Args:
        predictions (np.ndarray): Array of predictions (output of the network)
        target_vector (np.ndarray): Array containing the actual results
        sigmoid (bool, optional): Value describing whether or not we're using sigmoid. Defaults to False.

    Returns:
        np.ndarray: An array containing the loss value for each case run through.
                    Supports both batch GD (one example at a time), and SGD (minibatches)
    """
    def func(x): return 1 - x
    # Again we need to clip in order to resist divide by 0 shenanigans
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1-epsilon)
    targets = np.clip(targets, epsilon, 1-epsilon)
    # if sigmoid:
    #     targets = np.clip(targets, 0.1, 0.9)
    return - (targets / predictions) + (func(targets) / func(predictions))


def d_mse(predictions: np.ndarray, targets: np.ndarray, sigmoid: bool = False) -> np.ndarray:
    if sigmoid:
        targets = np.clip(targets, 0.1, 0.9)
    return (predictions - targets) * 2/predictions.shape[0]

if __name__ == "__main__":
    pred = np.array([0, 1, 1, 0]).reshape(4,1)
    true = np.array([0, 0, 1, 0]).reshape(4,1)
    print(cross_entropy(pred, true, True))
