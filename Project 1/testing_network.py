import numpy as np
from network_structures import *
from timeit import default_timer as timer

# User testing functions

def test_activations() -> None:
    test_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    test = Layer(a_func=sigmoid, d_func=d_sigmoid, weights=test_weights, biases=1)
    test_outputs = np.array([0.1, 0.4, 0.8])
    test_result = test.calculate_input(test_outputs)
    print(test_result)


def test_final_layer_activations() -> None:
    test_weights = np.array([[1]])
    test = Layer(a_func=sigmoid, d_func=sigmoid, weights=test_weights, biases=1)
    test_outputs = np.array([0.1])
    test_result = test.calculate_input(test_outputs)
    print(test_result)


def test_network_config() -> NeuralNetwork:
    test_config = {
        "Inputs": 3,
        "Outputs": 2,
        "Hidden layers": 2,
        "Softmax": False,
        "HL Activation functions": [unit, unit],
        "HL Derivative functions": [d_unit, d_unit],
        "HL Neurons": [5, 8],
        "Output function": unit,
        "Output derivative function": d_sigmoid
    }
    test_object = NeuralNetwork(config=test_config)
    return test_object


def test_network_forward_pass() -> None:
    test_object = test_network_config()
    inputs = np.array([100,100,100]).reshape(3, 1)
    test_object.forward_pass(inputs)
    print(f"Result: {test_object.output_layer.activations}")
    test_object.output_layer.get_output_jacobian()


def test_forward_pass_with_batch() -> None:
    test_object = test_network_config()
    inputs = np.arange(0,9,1).reshape(3, 3)
    print(inputs)
    test_object.forward_pass(inputs)
    for layer in test_object.hidden_layers:
        print(layer.activations)


def test_timing():
    Y = np.array([1,2,3,4,5])
    Z = np.array([1,2,3])
    start1 = timer()
    c = np.outer(Y, Z)
    end1 = timer()
    start2 = timer()
    d = np.einsum('i,j->ij', Y.ravel(), Z.ravel())
    end2 = timer()
    print(f"Outer time: {end1 - start1}")
    print(f"Einsum time: {end2 - start2}")
    print(c,d)

if __name__ == '__main__':
    # test_activations()
    # test_final_layer_activations()
    # test_network_config()
    # test_network_forward_pass()
    # test_forward_pass_with_batch()
    test_timing()