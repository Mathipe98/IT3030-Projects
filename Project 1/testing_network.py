import numpy as np
from network_structures import *
from nn_functions import *
from visualization import DrawNN
from timeit import default_timer as timer

config = {
        "Inputs": 3,
        "Outputs": 3,
        "Hidden layers": 1,
        "HL Activation functions": [sigmoid, relu],
        "HL Neurons": [2, 8],
        "Output function": sigmoid,
        "Loss function": mse,
        "Learning rate": 1,
        "Batch size": 1,
    }


# User testing functions

def test_activations() -> None:
    test_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    test = Layer(a_func=sigmoid, da_func=d_sigmoid, weights=test_weights, biases=1)
    test_outputs = np.array([0.1, 0.4, 0.8])
    test_result = test.calculate_input(test_outputs)
    print(test_result)


def test_final_layer_activations() -> None:
    test_weights = np.array([[1]])
    test = Layer(a_func=sigmoid, da_func=sigmoid, weights=test_weights, biases=1)
    test_outputs = np.array([0.1])
    test_result = test.calculate_input(test_outputs)
    print(test_result)


def test_network_config() -> None:
    test_object = NeuralNetwork(config)
    nodes = [config['Inputs']]
    for layer in test_object.hidden_layers:
        nodes.append(layer.n_nodes)
    nodes.append(test_object.output_layer.n_nodes)
    # print(nodes)
    network_vis = DrawNN(nodes)
    network_vis.draw()

def draw_network() -> None:
    obj = NeuralNetwork(config)
    nodes = [config['Inputs']]
    for layer in obj.hidden_layers:
        nodes.append(layer.n_nodes)
    nodes.append(obj.output_layer.n_nodes)
    network_vis = DrawNN(nodes)
    network_vis.draw()

def test_network_forward_pass() -> None:
    test_object = NeuralNetwork(config)
    inputs = np.array([100,100,100]).reshape(3, 1)
    test_object.forward_pass(inputs)
    print(f"Result: {test_object.output_layer.activations}")
    test_object.output_layer.get_output_jacobian()


def test_forward_pass_with_batch() -> None:
    test_object = NeuralNetwork(config)
    print("Batch size: 3")
    inputs = np.array([[100, 1, 1], [100, 1, 1], [100, 1, 1]])
    print(inputs)
    test_object.forward_pass(inputs)
    for layer in test_object.hidden_layers + [test_object.output_layer]:
        print(layer.activations)
    inputs = np.array([100,100,100]).reshape(3, 1)
    print(inputs)
    test_object.forward_pass(inputs)
    for layer in test_object.hidden_layers + [test_object.output_layer]:
        print(layer.activations)


def test_timing():
    Y = np.arange(0, 50*50).reshape(50,50)
    Z = np.array([1,2,3])
    start1 = timer()
    c = np.einsum('ij->ji', Y)
    end1 = timer()
    start2 = timer()
    d = Y.T
    end2 = timer()
    print(f"Einsum time: {end1 - start1}")
    print(f"T time: {end2 - start2}")
    print(c,d)







def test_backprop() -> None:
    batch_size = config["Batch size"]
    network = NeuralNetwork(config)
    # inputs =  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]).reshape(25,batch_size)
    inputs = np.array([1,0,0]).reshape(3,1)
    print(network.hidden_layers[0].weights)
    print(network.output_layer.weights)
    # inputs = np.zeros(shape=(25, batch_size))
    # inputs[18][0] = 1
    print(f"Inputs into the network:\n {inputs}", end="\n\n")
    print(f"Network structure:\n {[inputs.shape[0]] + [layer.n_nodes for layer in network.hidden_layers] + [network.output_layer.n_nodes]}")
    targets = np.array([1, 0, 0]).reshape(3,batch_size)
    print(f"Network targets:\n {targets}", end="\n\n")
    # print(f"Results from forward pass with batch size: " + str(3))
    # for i in range(network.output_layer.activations.shape[1]):
    #     col = network.output_layer.activations[:,i]
    #     print(col)
    # print(f"\nCorrect targets from above example:")
    # for i in range(targets.shape[1]):
    #     print(targets[:,i])
    network.backpropagation(inputs, targets)
    predictions = network.output_layer.activations
    print(network.l_func(predictions, targets))

if __name__ == '__main__':
    # draw_network()
    test_backprop()