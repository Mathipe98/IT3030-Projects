import numpy as np
from network_structures import *
from nn_functions import *
from visualization import DrawNN
from picture_generator import PictureGenerator
from timeit import default_timer as timer


# User testing functions

def draw_network(network: NeuralNetwork) -> None:
    nodes = [network.inputs]
    for layer in network.hidden_layers:
        nodes.append(layer.n_nodes)
    nodes.append(network.output_layer.n_nodes)
    network_vis = DrawNN(nodes)
    network_vis.draw()

def test_training() -> None:
    config = {
        "inputs": 2500,
        "outputs": 4,
        "lr": 0.01,
        "wreg": "L2",
        "wreg_lr": 0.001,
        "l_func": cross_entropy,
        "n_hl": 2,
        "hl_neurons": [10, 20],
        "hl_funcs": [relu, relu],
        "output_func": sigmoid,
    }
    network = NeuralNetwork(**config)
    pg_params = {
        "n": 50,
        "centered": False,
    }
    pg = PictureGenerator(**pg_params)
    dataset = pg.get_datasets()
    print("Dataset fetched.")
    training_data = dataset["training"][0:300]
    training_targets = dataset["training_targets"][0:300]
    print(f"Starting training with {len(training_data)} training examples.")
    network.train(training_data, training_targets)

    test_example = dataset["training"][220]
    test_solution = dataset["training_targets"][220]
    prediction = network.predict(test_example)
    print(f"Final network prediction:\n \tTarget: {test_solution}\n \tPrediction: {prediction}\n \tLoss: {network.l_func(prediction, test_solution, sigmoid=True)}")

    network.visualize_training_losses()



def debug_network():
    config = {
        "inputs": 3,
        "outputs": 2,
        "lr": 0.1,
        "wreg": "L2",
        "wreg_lr": 0.001,
        "n_hl": 1,
        "hl_neurons": [50, 10],
        "hl_funcs": [tanh, relu],
        "output_func": softmax,
        "l_func": mse,
    }
    network = NeuralNetwork(**config)
    test_input = np.array([1,0,0]).reshape(3,1)
    targets = np.array([1,0]).reshape(2,1)
    network.forward_pass(test_input)
    network.backpropagation(targets)

    


if __name__ == '__main__':
    # debug_network()
    test_training()