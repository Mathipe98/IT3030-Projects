import numpy as np
from network_structures import *
from nn_functions import *
from visualization import DrawNN
from picture_generator import PictureGenerator
from timeit import default_timer as timer


# User testing functions

def draw_network() -> None:
    obj = NeuralNetwork(config)
    nodes = [config['Inputs']]
    for layer in obj.hidden_layers:
        nodes.append(layer.n_nodes)
    nodes.append(obj.output_layer.n_nodes)
    network_vis = DrawNN(nodes)
    network_vis.draw()

def test_training() -> None:
    config = {
        "Inputs": 2500,
        "Outputs": 4,
        "Hidden layers": 1,
        "HL Activation functions": [relu, relu],
        "HL Neurons": [10, 20],
        "Output function": sigmoid,
        "Loss function": cross_entropy,
        "Learning rate": 0.1,
        "Batch size": 1,
        "Debug": False
    }
    network = NeuralNetwork(config)
    print(network.output_layer.weights)
    pg_params = {
        "n": 50,
        "centered": False,
    }
    pg = PictureGenerator(**pg_params)
    dataset = pg.get_datasets()
    print("Dataset fetched.")
    training_data = dataset["training"][0:100]
    training_targets = dataset["training_targets"][0:100]
    print(f"Starting training with {len(training_data)} training examples.")
    network.train(training_data, training_targets)

    test_example = dataset["training"][30]
    test_solution = dataset["training_targets"][30]
    prediction = network.predict(test_example)
    print(f"Final network prediction:\n \tTarget: {test_solution}\n \tPrediction: {prediction}\n \tLoss: {network.l_func(prediction, test_solution, sigmoid=True)}")

    # print(f"Shitty result targets:\n {network.debug_list}")
    network.visualize_training_losses()



def debug_network():
    config = {
        "Inputs": 3,
        "Outputs": 2,
        "Hidden layers": 0,
        "HL Activation functions": [relu],
        "HL Neurons": [2],
        "Output function": sigmoid,
        "Loss function": mse,
        "Learning rate": 0.01,
        "Batch size": 1,
        "Debug": True
    }
    network = NeuralNetwork(config)
    test_input = np.array([1,0,0]).reshape(3,1)
    targets = np.array([1,0]).reshape(2,1)
    network.forward_pass(test_input)
    network.backpropagation(targets)

    


if __name__ == '__main__':
    # debug_network()
    test_training()